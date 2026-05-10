import cv2
import numpy as np
import json
import argparse
import os
from pathlib import Path
from ultralytics import YOLO


DEFAULT_CONFIG = dict(
    n_frames        = 120,    
    clip_limit      = 2.0,    
    tile_size       = 8,      
    conf            = 0.50,
    iou             = 0.25,
    imgsz           = 640,
    min_dist_vp     = 20,     
    proj_tol        = 60,     
    min_dashes      = 2,      
    merge_bottom_tol= 40,     
    vp_conv_tol     = 80,     
    model_path      = "models/best_lane_seg.pt",
)

def extract_median_background(video_path: str, n_frames: int = 120) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[BG] Video: {total} frames | {fps:.1f} FPS | {w}x{h}")

    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames, prev = [], -1
    for idx in indices:
        if idx != prev + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        prev = idx
    cap.release()

    print(f"[BG] Loaded {len(frames)} frames — computing median...")
    stack = np.stack(frames, axis=0).astype(np.uint8)
    bg    = np.median(stack, axis=0).astype(np.uint8)
    print("[BG] Median done.")
    return bg



def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_enh   = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_enh, a, b]), cv2.COLOR_LAB2BGR)


def polyfit_mask(mask_bin):
    pts = np.column_stack(np.where(mask_bin > 0.5))
    if len(pts) < 10: return None
    return np.polyfit(pts[:, 0], pts[:, 1], 1)   # x = a*y + b

def mask_centroid(mask_bin):
    pts = np.column_stack(np.where(mask_bin > 0.5))
    if len(pts) < 5: return None
    cy, cx = pts.mean(axis=0)
    return float(cx), float(cy)

def mask_area(mask_bin):
    return float((mask_bin > 0.5).sum())

def line_intersects_frame(poly, y_top, y_bot, fw):
    a, b = poly
    x_top, x_bot = a * y_top + b, a * y_bot + b
    return max(x_top, x_bot) >= 0 and min(x_top, x_bot) <= fw

def single_line_intersection(p1, p2):
    a1, b1 = p1; a2, b2 = p2
    if abs(a1 - a2) < 1e-6: return None
    y = (b2 - b1) / (a1 - a2)
    return float(a1 * y + b1), float(y)

def compute_vp(solid_polys, w, h):
    candidates = []
    for i in range(len(solid_polys)):
        for j in range(i + 1, len(solid_polys)):
            pt = single_line_intersection(solid_polys[i], solid_polys[j])
            if pt is None: continue
            px, py = pt
            if -w < px < 2 * w and -h < py < h:
                candidates.append((px, py))
    if not candidates:
        print("[VP] No candidates — fallback to center-top")
        return (w / 2, -h * 0.1)
    vp = (float(np.median([p[0] for p in candidates])),
          float(np.median([p[1] for p in candidates])))
    print(f"[VP] {len(candidates)} candidates → ({vp[0]:.1f}, {vp[1]:.1f})")
    return vp

def project_to_bottom(cx, cy, vp, h):
    if abs(cy - vp[1]) < 1e-6:
        return float(cx)
    t = (h - vp[1]) / (cy - vp[1])
    return float(vp[0] + t * (cx - vp[0]))

def vp_convergence_ok(poly, vp, tol):
    a, b = poly
    delta = abs(a * vp[1] + b - vp[0])
    return delta <= tol, float(delta)

def get_frame_endpoints(poly, y_top, y_bot, fw):
    a, b = poly
    pts  = []
    for y in (y_top, y_bot):
        x = a * y + b
        if 0 <= x <= fw:
            pts.append((int(round(x)), y))
    if abs(a) > 1e-6:
        for x_edge in (0, fw):
            y = (x_edge - b) / a
            if y_top <= y <= y_bot:
                pts.append((x_edge, int(round(y))))
    if len(pts) < 2: return None
    pts.sort(key=lambda p: p[1])
    return pts[0], pts[-1]


def cluster_by_proj_x(data, tol):
    if not data: return []
    groups, cur = [], [data[0]]
    for d in data[1:]:
        mean_proj = np.mean([g['proj_x'] for g in cur])
        if abs(d['proj_x'] - mean_proj) <= tol:
            cur.append(d)
        else:
            groups.append(cur)
            cur = [d]
    groups.append(cur)
    return groups

def merge_close_lines(lines, tol):
    merged, i = [], 0
    while i < len(lines):
        if (i + 1 < len(lines) and
                abs(lines[i]['bottom_x'] - lines[i+1]['bottom_x']) < tol):
            if lines[i]['type'] == 'solid':
                keep = lines[i]
            elif lines[i+1]['type'] == 'solid':
                keep = lines[i+1]
            else:
                keep = lines[i] if lines[i]['n_pts'] >= lines[i+1]['n_pts'] else lines[i+1]
            merged.append(keep)
            i += 2
        else:
            merged.append(lines[i])
            i += 1
    return merged



def run_lane_detection(bg_image: np.ndarray, cfg: dict) -> tuple[list, dict, list]:
    """
    Returns:
        lane_lines : list of line dicts
        vp         : (x, y) vanishing point
        rejected   : list (để visualize)
    """
    model = YOLO(cfg['model_path'])
    H, W  = bg_image.shape[:2]
    Y_TOP = int(H * 0.20)

    results = model.predict(bg_image,
                            conf=cfg['conf'], iou=cfg['iou'],
                            imgsz=cfg['imgsz'])[0]
    print(f"[YOLO] Detected {len(results.boxes)} instances")

    dashes_raw, solid_polys, solid_areas = [], [], []
    for i, cls in enumerate(results.boxes.cls.int().tolist()):
        raw  = results.masks.data[i].cpu().numpy()
        mask = cv2.resize(raw, (W, H))
        if cls == 1:
            poly = polyfit_mask(mask)
            if poly is not None:
                solid_polys.append(poly)
                solid_areas.append(mask_area(mask))
        else:
            c = mask_centroid(mask)
            if c is not None:
                dashes_raw.append({'cx': c[0], 'cy': c[1]})

    print(f"[YOLO] Solids: {len(solid_polys)}, Dashes: {len(dashes_raw)}")

    # VP
    if len(solid_polys) < 2:
        print(f"[WARN] Only {len(solid_polys)} solid line(s) — VP fallback")
        vp = (W / 2, -H * 0.1)
    else:
        vp = compute_vp(solid_polys, W, H)
        if not (-W < vp[0] < 2 * W and -H < vp[1] < H * 0.9):
            print("[VP] Out of range — fallback")
            vp = (W / 2, -H * 0.1)

    dashes = []
    for d in dashes_raw:
        dist = float(np.hypot(d['cx'] - vp[0], d['cy'] - vp[1]))
        if dist < cfg['min_dist_vp'] or d['cy'] <= vp[1]:
            continue
        d['proj_x']  = project_to_bottom(d['cx'], d['cy'], vp, H)
        d['vp_dist'] = dist
        dashes.append(d)

    dashes_sorted = sorted(dashes, key=lambda d: d['proj_x'])
    all_groups    = cluster_by_proj_x(dashes_sorted, cfg['proj_tol'])
    dashed_groups = [g for g in all_groups if len(g) >= cfg['min_dashes']]

    def polyfit_cents(g):
        return np.polyfit([d['cy'] for d in g], [d['cx'] for d in g], 1)

    lane_lines, rejected = [], []

    for poly, area in zip(solid_polys, solid_areas):
        ok, delta = vp_convergence_ok(poly, vp, cfg['vp_conv_tol'])
        bottom_x  = poly[0] * H + poly[1]
        if ok:
            lane_lines.append({'type': 'solid', 'poly': poly,
                               'bottom_x': bottom_x, 'n_pts': 0, 'delta_vp': delta})
        else:
            rejected.append({'type': 'solid', 'group': [], 'reason': f'vp_delta={delta:.0f}'})

    for g in dashed_groups:
        poly      = polyfit_cents(g)
        ok, delta = vp_convergence_ok(poly, vp, cfg['vp_conv_tol'])
        bottom_x  = poly[0] * H + poly[1]
        if ok:
            lane_lines.append({'type': 'dashed', 'poly': poly,
                               'bottom_x': bottom_x, 'n_pts': len(g), 'delta_vp': delta})
        else:
            rejected.append({'type': 'dashed', 'group': g, 'reason': f'vp_delta={delta:.0f}'})

    lane_lines.sort(key=lambda x: x['bottom_x'])
    lane_lines = merge_close_lines(lane_lines, cfg['merge_bottom_tol'])
    lane_lines = [l for l in lane_lines
                  if line_intersects_frame(l['poly'], Y_TOP, H, W)]

    print(f"[DETECT] Final: {len(lane_lines)} lines | {max(0, len(lane_lines)-1)} lanes")
    return lane_lines, vp, rejected, dashes, dashed_groups


COLORS = [
    (0,255,0),(0,200,255),(255,120,0),(200,0,255),(0,255,180),
    (255,200,0),(180,255,0),(255,0,150),(0,150,255),(100,255,200)
]
SOLID_COLOR = (0, 215, 255)

def build_visualization(bg_image: np.ndarray, lane_lines: list, vp: tuple,
                         rejected: list, dashes: list, dashed_groups: list,
                         cfg: dict) -> np.ndarray:
    H, W  = bg_image.shape[:2]
    Y_TOP = int(H * 0.20)
    vis   = bg_image.copy()

    def polyfit_cents(g):
        return np.polyfit([d['cy'] for d in g], [d['cx'] for d in g], 1)

    # VP marker
    vp_px = (int(np.clip(vp[0], 0, W-1)), int(np.clip(vp[1], 0, H-1)))
    cv2.circle(vis, vp_px, 8, (255,255,255), -1)
    cv2.circle(vis, vp_px, 8, (0,0,0), 2)

    for r in rejected:
        if r['type'] == 'dashed':
            for d in r['group']:
                cv2.circle(vis, (int(d['cx']), int(d['cy'])), 4, (60,60,180), -1)

    valid_groups = [g for g in dashed_groups
                    if vp_convergence_ok(polyfit_cents(g), vp, cfg['vp_conv_tol'])[0]]
    for gi, g in enumerate(valid_groups):
        col = COLORS[gi % len(COLORS)]
        for d in g:
            cv2.circle(vis, (int(d['cx']), int(d['cy'])), 5, col, -1)
            cv2.circle(vis, (int(d['cx']), int(d['cy'])), 5, (0,0,0), 1)

    # Lane lines + Line IDs
    dash_ci = 0
    for i, line in enumerate(lane_lines):
        endpoints = get_frame_endpoints(line['poly'], Y_TOP, H, W)
        if endpoints is None:
            continue
        pt1, pt2 = endpoints
        if line['type'] == 'solid':
            col, thick = SOLID_COLOR, 3
        else:
            col, thick = COLORS[dash_ci % len(COLORS)], 2
            dash_ci += 1
        cv2.line(vis, pt1, pt2, col, thick)
        lx = int(np.clip(pt1[0], 5, W - 40))
        cv2.putText(vis, f"L{i+1}", (lx, pt1[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # Lane IDs (midpoint giữa 2 line)
    for i in range(len(lane_lines) - 1):
        a1, b1 = lane_lines[i]['poly']
        a2, b2 = lane_lines[i+1]['poly']
        ym = int(H * 0.75)
        xm = int(((a1*ym+b1) + (a2*ym+b2)) / 2)
        if 0 < xm < W:
            cv2.putText(vis, f"Ln{i+1}", (xm-15, ym),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,0), 2)

    # Info overlay
    n_dashes_total = sum(len(g) for g in dashed_groups)
    infos = [
        f"Lines:{len(lane_lines)} Lanes:{max(0,len(lane_lines)-1)}",
        f"VP:({vp[0]:.0f},{vp[1]:.0f}) tol:{cfg['vp_conv_tol']}px",
        f"Rejected:{len(rejected)} | Dashes:{len(dashes)}/{len(dashes)+len(rejected)}",
    ]
    for j, t in enumerate(infos):
        cv2.putText(vis, t, (10, H - 12 - j * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

    return vis



def detect_lanes(
    video_path: str,
    model_path: str = "best_lane_seg.pt",
    n_frames:   int   = 120,
    **kwargs
) -> tuple[np.ndarray, dict]:
    cfg = {**DEFAULT_CONFIG, 'model_path': model_path, 'n_frames': n_frames, **kwargs}

    bg_raw   = extract_median_background(video_path, cfg['n_frames'])

    #CLAHE
    bg_clahe = apply_clahe(bg_raw, cfg['clip_limit'], cfg['tile_size'])

    lane_lines, vp, rejected, dashes, dashed_groups = run_lane_detection(bg_clahe, cfg)

    vis = build_visualization(bg_clahe, lane_lines, vp, rejected, dashes, dashed_groups, cfg)

    #Build config dict
    H, W = bg_clahe.shape[:2]

    def nat(v):
        if isinstance(v, (np.integer,)):  return int(v)
        if isinstance(v, (np.floating,)): return float(v)
        return v

    lane_config = {
        "frame_width":     W,
        "frame_height":    H,
        "vanishing_point": [nat(vp[0]), nat(vp[1])],
        "lines": [
            {
                "id":      f"Line {i+1}",
                "type":    l['type'],
                "poly":    [nat(l['poly'][0]), nat(l['poly'][1])],
                "bottom_x": nat(l['bottom_x']),
            }
            for i, l in enumerate(lane_lines)
        ],
        "lanes": [
            {
                "id":         f"Lane {i+1}",
                "left_line":  f"Line {i+1}",
                "right_line": f"Line {i+2}",
            }
            for i in range(len(lane_lines) - 1)
        ],
    }

    return vis, lane_config



#quick test
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lane & Line Detection Pipeline")
    parser.add_argument("--video",      required=True,  help="Path to input video")
    parser.add_argument("--model",      default="best_lane_seg.pt")
    parser.add_argument("--n_frames",   type=int,   default=120)
    parser.add_argument("--output",     default="lane_result.jpg")
    parser.add_argument("--save_config",default="lane_config.json")
    args = parser.parse_args()

    vis, config = detect_lanes(
        video_path  = args.video,
        model_path  = args.model,
        n_frames    = args.n_frames,
    )

    cv2.imwrite(args.output, vis)
    with open(args.save_config, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nDone!")
    print(f"   Visualization : {args.output}")
    print(f"   Config JSON   : {args.save_config}")
    print(f"   Lines detected: {len(config['lines'])}")
    print(f"   Lanes inferred: {len(config['lanes'])}")