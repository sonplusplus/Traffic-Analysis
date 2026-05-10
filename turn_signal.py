import cv2
import numpy as np
from collections import defaultdict, deque
from typing import Optional

try:
    from ultralytics import YOLO as _YOLO
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _ULTRALYTICS_AVAILABLE = False

_TS_DEVICE = "cpu"
print("[TurnSignal] Running on CPU (GPU disabled to prevent CUDA illegal memory access)")


MIN_ROI_PX      = 40
REF_ALPHA       = 0.85
WINDOW_FRAMES   = 40
BASELINE_FRAMES = 28
MIN_ZC          = 6
MIN_SWING       = 8.0
SILENCE_FRAMES  = 30
COOLDOWN_FRAMES = 25

SEG_CONF        = 0.25
SEG_IOU         = 0.45
SEG_IMGSZ       = 640  

MIN_CONTAINMENT = 0.50

CLASS_ROI_CONFIG = {
    2: dict(y_skip=0.12, y_height=0.38, corner_x_frac=0.28),
    3: dict(y_skip=0.00, y_height=0.40, corner_x_frac=0.32),
    5: dict(y_skip=0.00, y_height=0.28, corner_x_frac=0.22),
    7: dict(y_skip=0.05, y_height=0.28, corner_x_frac=0.22),
}
DEFAULT_ROI_CONFIG = dict(y_skip=0.05, y_height=0.30, corner_x_frac=0.25)

ROI_Y_FRAC    = 0.28
CORNER_X_FRAC = 0.22


def fmt_time(sec: float) -> str:
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m:02d}:{s:02d}"


def mean_gray(roi_bgr: np.ndarray) -> Optional[float]:
    if roi_bgr is None or roi_bgr.size == 0:
        return None
    if roi_bgr.shape[0] * roi_bgr.shape[1] < MIN_ROI_PX:
        return None
    return float(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY).mean())


def mask_mean_brightness(frame_gray: np.ndarray, mask_bin: np.ndarray) -> Optional[float]:
    active = mask_bin > 0.5
    if active.sum() < MIN_ROI_PX:
        return None
    return float(frame_gray[active].mean())


def count_zero_crossings(arr: np.ndarray) -> int:
    signs = np.sign(arr)
    signs = signs[signs != 0]
    if len(signs) < 2:
        return 0
    return int(np.sum(signs[:-1] != signs[1:]))


def bbox_iou(b1, b2) -> float:
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter)


def signal_containment(sig_bbox, veh_bbox) -> float:
    sx1, sy1, sx2, sy2 = sig_bbox
    vx1, vy1, vx2, vy2 = veh_bbox
    ix1 = max(sx1, vx1)
    iy1 = max(sy1, vy1)
    ix2 = min(sx2, vx2)
    iy2 = min(sy2, vy2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    sig_area = max(1, (sx2 - sx1) * (sy2 - sy1))
    return inter / sig_area


def get_signal_rois(frame, x1, y1, x2, y2, cls=5):
    cfg      = CLASS_ROI_CONFIG.get(cls, DEFAULT_ROI_CONFIG)
    bh       = y2 - y1
    bw       = x2 - x1
    if bh < 12 or bw < 12:
        return None, None, None, y2, y2, 4
    y_end    = y2 - int(bh * cfg["y_skip"])
    y_start  = y_end - max(int(bh * cfg["y_height"]), 6)
    y_start  = max(y_start, y1)
    corner_w = max(int(bw * cfg["corner_x_frac"]), 4)
    ctr_x1   = x1 + corner_w
    ctr_x2   = x2 - corner_w
    left_roi   = frame[y_start:y_end, x1            : x1 + corner_w]
    right_roi  = frame[y_start:y_end, x2 - corner_w : x2           ]
    center_roi = frame[y_start:y_end, ctr_x1:ctr_x2] if ctr_x2 > ctr_x1 + 4 else None
    return left_roi, right_roi, center_roi, y_start, y_end, corner_w

class BrightnessSignalState:
    def __init__(
        self,
        window_frames  : int   = WINDOW_FRAMES,
        baseline_frames: int   = BASELINE_FRAMES,
        min_zc         : int   = MIN_ZC,
        min_swing      : float = MIN_SWING,
        silence_frames : int   = SILENCE_FRAMES,
        cooldown_frames: int   = COOLDOWN_FRAMES,
    ):
        self.window_frames   = window_frames
        self.baseline_frames = baseline_frames
        self.min_zc          = min_zc
        self.min_swing       = min_swing
        self.silence_frames  = silence_frames
        self.cooldown_frames = cooldown_frames

        self.signal_buf       : deque = deque(maxlen=window_frames)
        self.frames_no_signal : int   = 0
        self.is_blinking      : bool  = False
        self.cooldown_left    : int   = 0

    def _oscillation_score(self) -> tuple[int, float]:
        arr = np.array(self.signal_buf)
        if len(arr) < self.baseline_frames:
            return 0, 0.0
        detrended = arr - np.mean(arr)
        zc    = count_zero_crossings(detrended)
        swing = float(detrended.max() - detrended.min())
        return zc, swing

    def update(self, brightness_signal: float) -> Optional[str]:
        self.signal_buf.append(brightness_signal)
        self.frames_no_signal = 0
        if self.cooldown_left > 0:
            self.cooldown_left -= 1
            return None
        zc, swing = self._oscillation_score()
        blink_now = (zc >= self.min_zc) and (swing >= self.min_swing)
        if not self.is_blinking and blink_now:
            self.is_blinking = True
            return "start"
        if self.is_blinking and not blink_now:
            self.is_blinking = False
            self.cooldown_left = self.cooldown_frames
            return "end"
        return None

    def force_reset(self, cooldown: int = 0) -> None:
        self.is_blinking   = False
        self.cooldown_left = cooldown

    def mark_missing(self) -> Optional[str]:
        self.frames_no_signal += 1
        if self.frames_no_signal >= self.silence_frames and self.is_blinking:
            self.is_blinking = False
            self.cooldown_left = self.cooldown_frames
            return "end"
        return None

    def accumulation_level(self) -> float:
        if len(self.signal_buf) < self.baseline_frames:
            return 0.0
        arr       = np.array(self.signal_buf)
        detrended = arr - np.mean(arr)
        signs     = np.sign(detrended)
        signs     = signs[signs != 0]
        zc        = int(np.sum(signs[:-1] != signs[1:])) if len(signs) >= 2 else 0
        swing     = float(detrended.max() - detrended.min())
        return min((zc / self.min_zc + swing / self.min_swing) / 2.0, 1.0)

class SignalMaskDetector:
    def __init__(self, model_path: str, device: Optional[str] = None):
        if not _ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics not installed. pip install ultralytics")

        # Always CPU regardless of what device param says
        self.device = "cpu"
        self.half   = False

        self.model = _YOLO(model_path)
        print(f"[TurnSignal] Loaded seg model: {model_path}  device=cpu  half=False  imgsz={SEG_IMGSZ}")

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Returns list of detected signal masks. Each: {'bbox','mask','conf','cx','cy'}"""
        H, W = frame.shape[:2]
        results = self.model.predict(
            frame,
            conf    = SEG_CONF,
            iou     = SEG_IOU,
            imgsz   = SEG_IMGSZ,
            device  = "cpu",
            half    = False,
            verbose = False,
        )[0]

        if results.masks is None or len(results.masks) == 0:
            return []

        detected  = []
        xyxy_list = results.boxes.xyxy.cpu().numpy().astype(int)
        conf_list = results.boxes.conf.cpu().numpy()
        mask_data = results.masks.data.cpu().numpy()

        for i in range(len(xyxy_list)):
            x1, y1, x2, y2 = xyxy_list[i]
            conf            = float(conf_list[i])

            raw_mask  = mask_data[i]
            mask_full = cv2.resize(raw_mask, (W, H))
            mask_bool = mask_full > 0.5

            ys, xs = np.where(mask_bool)
            if len(xs) < MIN_ROI_PX:
                continue
            cx = float(xs.mean())
            cy = float(ys.mean())

            detected.append({
                'bbox': (x1, y1, x2, y2),
                'mask': mask_bool,
                'conf': conf,
                'cx':   cx,
                'cy':   cy,
            })

        return detected


class TurnSignalDetector:
    SIDES = ("left", "right")

    def __init__(
        self,
        seg_model_path : Optional[str] = None,
        window_frames  : int   = WINDOW_FRAMES,
        min_zc         : int   = MIN_ZC,
        min_swing      : float = MIN_SWING,
        silence_frames : int   = SILENCE_FRAMES,
        cooldown_frames: int   = COOLDOWN_FRAMES,
        ref_alpha      : float = REF_ALPHA,
        device         : Optional[str] = None,   # kept for API compat, ignored
    ):
        self.window_frames   = window_frames
        self.min_zc          = min_zc
        self.min_swing       = min_swing
        self.silence_frames  = silence_frames
        self.cooldown_frames = cooldown_frames
        self.ref_alpha       = ref_alpha
        self._device         = "cpu"

        self._use_seg = False
        if seg_model_path is not None:
            try:
                self._seg_detector = SignalMaskDetector(seg_model_path, device="cpu")
                self._use_seg = True
                print("[TurnSignal] Using YOLO-seg mask mode (CPU)")
            except Exception as e:
                print(f"[TurnSignal] Could not load seg model ({e}), falling back to heuristic")

        if not self._use_seg:
            print("[TurnSignal] Using corner-ROI heuristic (CPU)")

        self.states           : dict = defaultdict(dict)
        self.open_events      : dict = {}
        self.completed_events : list = []

    def _get_state(self, vid: str, side: str) -> BrightnessSignalState:
        if side not in self.states[vid]:
            self.states[vid][side] = BrightnessSignalState(
                window_frames   = self.window_frames,
                min_zc          = self.min_zc,
                min_swing       = self.min_swing,
                silence_frames  = self.silence_frames,
                cooldown_frames = self.cooldown_frames,
            )
        return self.states[vid][side]

    def _handle_signal(self, signal, vid, side, ts, new_events):
        key = f"{vid}__{side}"
        if signal == "start":
            self.open_events[key] = {
                "event_type": "turn_signal",
                "vehicle_id": vid,
                "signal":     side,
                "start_time": fmt_time(ts),
                "end_time":   None,
            }
        elif signal == "end" and key in self.open_events:
            ev = self.open_events.pop(key)
            ev["end_time"] = fmt_time(ts)
            new_events.append(ev)
            self.completed_events.append(ev)

    def _apply_mutual_exclusion(self, vid: str, raw_signals: dict, new_events: list):
        left_st  = self._get_state(vid, "left")
        right_st = self._get_state(vid, "right")
        if left_st.is_blinking and right_st.is_blinking:
            for side in self.SIDES:
                self.open_events.pop(f"{vid}__{side}", None)
                self._get_state(vid, side).force_reset(cooldown=self.cooldown_frames)
                raw_signals[side] = None
            new_events[:] = [
                e for e in new_events
                if not (e.get("vehicle_id") == vid and e.get("event_type") == "turn_signal")
            ]
            return True
        return False

    def _match_masks_to_vehicles(
        self,
        detected_signals : list[dict],
        bboxes           : list,
        debug            : bool = False,
        frame_idx        : int  = 0,
    ) -> dict[str, dict[str, list]]:
        matches = {}
        for item in bboxes:
            vid, vx1, vy1, vx2, vy2 = item[0], item[1], item[2], item[3], item[4]
            v_cx    = (vx1 + vx2) / 2.0
            matched = {'left': [], 'right': []}

            for di, sig in enumerate(detected_signals):
                containment = signal_containment(sig['bbox'], (vx1, vy1, vx2, vy2))

                if debug and containment > 0.01:
                    side_label = "L" if sig['cx'] < v_cx else "R"
                    result     = "MATCH" if containment >= MIN_CONTAINMENT else "skip"
                    print(
                        f"  [DBG f{frame_idx}] sig{di}(conf={sig['conf']:.2f}) <-> {vid}: "
                        f"contain={containment:.3f} {result} side={side_label}"
                    )

                if containment < MIN_CONTAINMENT:
                    continue

                side = 'left' if sig['cx'] < v_cx else 'right'
                matched[side].append(sig['mask'])

            matches[vid] = matched
        return matches

    def _process_frame_v6(
        self,
        frame    : np.ndarray,
        bboxes   : list,
        frame_idx: int,
        fps      : float,
        debug    : bool = False,
    ) -> list[dict]:
        ts         = frame_idx / fps
        new_events = []
        seen_vids  = set()

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected   = self._seg_detector.detect(frame)

        if debug and (frame_idx % 15 == 0 or len(detected) > 0):
            print(f"  [DBG f{frame_idx}] seg -> {len(detected)} signal(s) | {len(bboxes)} vehicle(s)")

        matches = self._match_masks_to_vehicles(detected, bboxes, debug=debug, frame_idx=frame_idx)

        for item in bboxes:
            vid = item[0]
            seen_vids.add(vid)
            vid_matches = matches.get(vid, {'left': [], 'right': []})

            raw_signals: dict[str, Optional[str]] = {}

            for side in self.SIDES:
                masks = vid_matches[side]
                if not masks:
                    state             = self._get_state(vid, side)
                    raw_signals[side] = state.mark_missing()
                else:
                    combined = np.zeros(frame_gray.shape, dtype=bool)
                    for m in masks:
                        combined |= m
                    brightness = mask_mean_brightness(frame_gray, combined)

                    if brightness is None:
                        state             = self._get_state(vid, side)
                        raw_signals[side] = state.mark_missing()
                    else:
                        state             = self._get_state(vid, side)
                        raw_signals[side] = state.update(brightness)

            suppressed = self._apply_mutual_exclusion(vid, raw_signals, new_events)
            if suppressed:
                continue

            for side in self.SIDES:
                self._handle_signal(raw_signals[side], vid, side, ts, new_events)

        for vid in list(self.states.keys()):
            if vid not in seen_vids:
                for side in self.SIDES:
                    if side in self.states[vid]:
                        self._handle_signal(
                            self.states[vid][side].mark_missing(),
                            vid, side, ts, new_events,
                        )

        return new_events

    def _process_frame_v5(
        self,
        frame    : np.ndarray,
        bboxes   : list,
        frame_idx: int,
        fps      : float,
        debug    : bool = False,
    ) -> list[dict]:
        ts         = frame_idx / fps
        new_events = []
        seen_vids  = set()

        for item in bboxes:
            if len(item) == 6:
                vid, x1, y1, x2, y2, cls = item
            else:
                vid, x1, y1, x2, y2 = item
                cls = 5
            seen_vids.add(vid)

            left_roi, right_roi, center_roi, *_ = get_signal_rois(frame, x1, y1, x2, y2, cls=cls)
            ref_brightness = mean_gray(center_roi)
            rois = {"left": left_roi, "right": right_roi}

            raw_signals: dict[str, Optional[str]] = {}
            for side in self.SIDES:
                corner_brightness = mean_gray(rois[side])
                if corner_brightness is None:
                    raw_signals[side] = self._get_state(vid, side).mark_missing()
                else:
                    adjusted = (
                        corner_brightness - self.ref_alpha * ref_brightness
                        if ref_brightness is not None else corner_brightness
                    )
                    raw_signals[side] = self._get_state(vid, side).update(adjusted)

            suppressed = self._apply_mutual_exclusion(vid, raw_signals, new_events)
            if suppressed:
                continue

            for side in self.SIDES:
                self._handle_signal(raw_signals[side], vid, side, ts, new_events)

        for vid in list(self.states.keys()):
            if vid not in seen_vids:
                for side in self.SIDES:
                    if side in self.states[vid]:
                        self._handle_signal(
                            self.states[vid][side].mark_missing(),
                            vid, side, ts, new_events,
                        )

        return new_events

    def process_bboxes(
        self,
        frame    : np.ndarray,
        bboxes   : list,
        frame_idx: int,
        fps      : float,
        debug    : bool = False,
    ) -> list[dict]:
        if self._use_seg:
            return self._process_frame_v6(frame, bboxes, frame_idx, fps, debug=debug)
        else:
            return self._process_frame_v5(frame, bboxes, frame_idx, fps, debug=debug)

    def get_active_signals(self, vid: str) -> list:
        """Returns list of currently blinking sides for vid (for annotation)."""
        return [
            s for s in self.SIDES
            if self.states.get(vid, {}).get(s) is not None
            and self.states[vid][s].is_blinking
        ]

    def finalize(self, total_frames: int, fps: float) -> list[dict]:
        """Close any still-open events at end of video."""
        end_time = fmt_time(total_frames / fps)
        leftover = []
        for key, ev in list(self.open_events.items()):
            ev["end_time"] = end_time
            leftover.append(ev)
            self.completed_events.append(ev)
        self.open_events.clear()
        return leftover

    def get_all_events(self) -> list[dict]:
        return list(self.completed_events)


if __name__ == "__main__":
    import argparse, json, sys, os
    from ultralytics import YOLO

    parser = argparse.ArgumentParser(description="Turn Signal Detector — standalone test (CPU)")
    parser.add_argument("--video",         required=True)
    parser.add_argument("--model",         default="models/best_signal_seg.pt")
    parser.add_argument("--vehicle_model", default="yolov8n.pt")
    parser.add_argument("--output",        default="debug_ts_cpu.mp4")
    parser.add_argument("--stride",        type=int,  default=2)
    parser.add_argument("--watch",         default=None)
    parser.add_argument("--debug",         action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        sys.exit(f"Video not found: {args.video}")

    seg_model_path = args.model if os.path.exists(args.model) else None
    if seg_model_path is None:
        print(f"[WARN] {args.model} not found — falling back to heuristic")

    ts_detector   = TurnSignalDetector(seg_model_path=seg_model_path)
    vehicle_model = YOLO(args.vehicle_model)
    VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    cap     = cv2.VideoCapture(args.video)
    fps     = cap.get(cv2.CAP_PROP_FPS)
    W       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_fps = max(1, int(fps) // args.stride)
    writer  = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*'mp4v'), out_fps, (W, H)
    )

    print(f"[INFO] {total} frames | {fps:.1f} FPS | stride={args.stride} | device=cpu")

    all_events = []
    frame_idx  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % args.stride != 0:
            frame_idx += 1
            continue

        res = vehicle_model.track(
            frame, persist=True,
            classes=list(VEHICLE_CLASSES.keys()),
            conf=0.30, verbose=False,
            device="cpu",
        )[0]

        bboxes = []
        if res.boxes.id is not None:
            for tid, box, cls in zip(
                res.boxes.id.int().tolist(),
                res.boxes.xyxy.cpu().numpy().astype(int),
                res.boxes.cls.int().tolist(),
            ):
                vid = f"V{tid:03d}"
                x1, y1, x2, y2 = box
                bboxes.append((vid, x1, y1, x2, y2, cls))

        new_evs = ts_detector.process_bboxes(frame, bboxes, frame_idx, fps, debug=args.debug)
        all_events.extend(new_evs)
        for ev in new_evs:
            print(f"  [{ev['start_time']}] turn_signal {ev['signal']:>5} | {ev['vehicle_id']}")

        annotated = frame.copy()
        for item in bboxes:
            vid, x1, y1, x2, y2, cls = item
            if args.watch and vid != args.watch:
                continue
            active  = ts_detector.get_active_signals(vid)
            arrows  = {"left": "<", "right": ">"}
            sig_str = " ".join(arrows.get(s, s) for s in active)
            label   = f"{vid} {sig_str}" if sig_str else vid
            color   = (0, 80, 255) if active else (50, 220, 50)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.putText(
            annotated,
            f"[CPU  f={frame_idx}]",
            (10, H - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1,
        )

        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()
    all_events.extend(ts_detector.finalize(frame_idx, fps))

    print(f"\n[DONE] {len(all_events)} turn signal events")
    for ev in all_events:
        print(f"  {ev['vehicle_id']} | {ev['signal']:>5} | {ev['start_time']} -> {ev['end_time']}")
    print(f"  Video: {args.output}")