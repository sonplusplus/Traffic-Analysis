# event_tracker.py — CPU-only (GPU removed to avoid CUDA memory conflicts)

import cv2
import numpy as np
from collections import defaultdict, deque
from typing import Optional

from ultralytics import YOLO
from turn_signal import TurnSignalDetector


# ─────────────────────────────────────────────────────────────────────────────
# Device — hardcoded CPU to avoid FP16/FP32 CUDA memory conflicts
# ─────────────────────────────────────────────────────────────────────────────

DEVICE   = "cpu"
USE_HALF = False

print("[Device] Running on CPU (GPU disabled to prevent CUDA illegal memory access)")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def line_x_at_y(poly: tuple, y: float) -> float:
    a, b = poly
    return a * y + b


def point_to_line_dist(px: float, py: float, poly: tuple) -> float:
    a, b = poly
    return abs(px - a * py - b) / np.sqrt(1.0 + a * a)


def fmt_time(sec: float) -> str:
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m:02d}:{s:02d}"


# ─────────────────────────────────────────────────────────────────────────────
# Vectorized geometry — precomputed numpy arrays for speed
# ─────────────────────────────────────────────────────────────────────────────

class LaneGeometry:
    """
    Caches lane-line poly arrays as numpy for fast vectorized ops.
    Created once from lane_config, reused every frame.

    Line equation:  x = a*y + b
    polys_arr shape: (N_lines, 2)  — [[a0,b0], [a1,b1], ...]
    """

    def __init__(self, lane_config: dict):
        lines = lane_config.get("lines", [])
        self.line_ids  : list[str]   = [l["id"]   for l in lines]
        self.polys_list: list[tuple] = [tuple(l["poly"]) for l in lines]
        self.n_lines   : int         = len(lines)

        if self.n_lines > 0:
            arr = np.array([[l["poly"][0], l["poly"][1]] for l in lines], dtype=np.float64)
            self.A = arr[:, 0]   # shape (N,)
            self.B = arr[:, 1]   # shape (N,)
        else:
            self.A = np.empty(0)
            self.B = np.empty(0)

    def touching_lines(self, ax: float, ay: float, thr: float) -> list[str]:
        """Return IDs of lines whose distance to anchor (ax, ay) < thr."""
        if self.n_lines == 0:
            return []
        dists = np.abs(ax - self.A * ay - self.B) / np.sqrt(1.0 + self.A * self.A)
        return [self.line_ids[i] for i in np.where(dists < thr)[0]]

    def lane_at(self, ax: float, ay: float) -> Optional[int]:
        """Return 1-based lane index containing anchor point, or None."""
        if self.n_lines < 2:
            return None
        xs = self.A * ay + self.B
        for i in range(self.n_lines - 1):
            if xs[i] <= ax <= xs[i + 1]:
                return i + 1
        return None


# ─────────────────────────────────────────────────────────────────────────────
# State machines
# ─────────────────────────────────────────────────────────────────────────────

class TouchLineState:
    """Debounced touch detector for one vehicle × one lane line."""

    def __init__(self, debounce: int = 4):
        self.debounce   = debounce
        self.in_touch   = False
        self.hit_count  = 0
        self.miss_count = 0

    def update(self, touching: bool) -> Optional[str]:
        if touching:
            self.miss_count = 0
            if not self.in_touch:
                self.hit_count += 1
                if self.hit_count >= self.debounce:
                    self.in_touch  = True
                    self.hit_count = 0
                    return "start"
        else:
            self.hit_count = 0
            if self.in_touch:
                self.miss_count += 1
                if self.miss_count >= self.debounce:
                    self.in_touch   = False
                    self.miss_count = 0
                    return "end"
        return None


class LaneChangeState:
    """
    Tracks per-vehicle lane-change events via sliding-window mode detection.
    STABLE → (lane changes) → CROSSING → (new lane stable N frames) → STABLE + emit event
    """

    def __init__(self, stable_frames: int = 5, cooldown_frames: int = 20):
        self.stable_frames   = stable_frames
        self.cooldown_frames = cooldown_frames
        self.history: deque  = deque(maxlen=stable_frames * 4)
        self.confirmed_lane : Optional[int] = None
        self.crossing       : bool          = False
        self.cross_from     : Optional[int] = None
        self.cross_start_idx: int           = 0
        self.cooldown_left  : int           = 0

    def _mode_of(self, vals) -> Optional[int]:
        valid = [v for v in vals if v is not None]
        if len(valid) < self.stable_frames:
            return None
        counts: dict[int, int] = {}
        for v in valid:
            counts[v] = counts.get(v, 0) + 1
        best_val, best_cnt = max(counts.items(), key=lambda x: x[1])
        return best_val if best_cnt >= self.stable_frames else None

    def update(self, lane_id: Optional[int], frame_idx: int) -> Optional[tuple]:
        self.history.append(lane_id)

        if self.cooldown_left > 0:
            self.cooldown_left -= 1
            return None

        hist = list(self.history)

        if self.confirmed_lane is None:
            mode = self._mode_of(hist[-self.stable_frames:])
            if mode is not None:
                self.confirmed_lane = mode
            return None

        if not self.crossing:
            if lane_id is not None and lane_id != self.confirmed_lane:
                self.crossing        = True
                self.cross_from      = self.confirmed_lane
                self.cross_start_idx = frame_idx
                return ("crossing_start", self.cross_from, frame_idx)
            return None

        recent     = hist[-self.stable_frames:]
        new_stable = self._mode_of(recent)

        if new_stable is not None and new_stable != self.cross_from:
            result = ("crossing_end", self.cross_from, new_stable,
                      self.cross_start_idx, frame_idx)
            self.confirmed_lane = new_stable
            self.crossing       = False
            self.cross_from     = None
            self.cooldown_left  = self.cooldown_frames
            return result

        if new_stable == self.cross_from:
            self.crossing   = False
            self.cross_from = None
            return None

        return None

    @property
    def is_crossing(self) -> bool:
        return self.crossing


class EventTracker:


    def __init__(
        self,
        model_path             : str,
        lane_config            : dict,
        touch_thr              : float = 14.0,
        touch_debounce         : int   = 4,
        min_touch_duration_sec : float = 0.5,
        stable_frames          : int   = 5,
        cooldown_frames        : int   = 20,
        conf                   : float = 0.30,
        iou                    : float = 0.45,
        imgsz                  : int   = 640,
        turn_signal_detector          = None,
        device                 : Optional[str] = None,   # kept for API compat, ignored
    ):
        # Always CPU — no GPU to avoid CUDA illegal memory access
        self.device   = "cpu"
        self.use_half = False

        print(f"[Tracker] Loading YOLO model: {model_path}  →  device=cpu  half=False")
        self.model = YOLO(model_path)

        # ── Config ────────────────────────────────────────────────────────
        self.lane_config = lane_config
        self.geom        = LaneGeometry(lane_config)
        self.touch_thr   = touch_thr
        self.touch_debounce = touch_debounce
        self.min_touch_duration_sec = min_touch_duration_sec
        self.conf        = conf
        self.iou         = iou
        self.imgsz       = imgsz

        # ── State machines ────────────────────────────────────────────────
        self.touch_states : dict[str, dict[str, TouchLineState]] = defaultdict(dict)
        self.change_states: dict[str, LaneChangeState] = defaultdict(
            lambda: LaneChangeState(stable_frames, cooldown_frames)
        )

        # ── Event accumulators ────────────────────────────────────────────
        self.open_touch_events : dict[str, dict] = {}
        self.open_change_events: dict[str, dict] = {}
        self.completed_events  : list[dict]      = []
        self._active_labels    : dict[str, str]  = {}

        self._line_ids = self.geom.line_ids

        # ── Turn signal detector ──────────────────────────────────────────
        self.ts_detector = turn_signal_detector
        self.last_bboxes : list = []

    # ─────────────────────────────────────────────────────────────────────────
    # Main per-frame method
    # ─────────────────────────────────────────────────────────────────────────

    def process_frame(
        self, frame: np.ndarray, frame_idx: int, fps: float
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Run tracking + event detection on one frame.

        Returns
        -------
        annotated  : BGR frame with bounding boxes and labels drawn
        new_events : list of event dicts emitted this frame
        """

        # ── YOLO track (CPU) ─────────────────────────────────────────────
        results = self.model.track(
            frame,
            persist  = True,
            classes  = list(VEHICLE_CLASSES.keys()),
            conf     = self.conf,
            iou      = self.iou,
            imgsz    = self.imgsz,
            device   = "cpu",
            half     = False,
            verbose  = False,
        )[0]

        annotated  = frame.copy()
        new_events : list[dict] = []
        seen_vids  : set[str]   = set()
        ts_sec     = frame_idx / fps

        # ── Per-vehicle processing ────────────────────────────────────────
        if results.boxes.id is not None:
            self.last_bboxes = []

            track_ids = results.boxes.id.int().tolist()
            xyxy_list = results.boxes.xyxy.cpu().numpy().astype(int)
            cls_list  = results.boxes.cls.int().tolist()

            for tid, box, cls in zip(track_ids, xyxy_list, cls_list):
                vid = f"V{tid:03d}"
                seen_vids.add(vid)
                x1, y1, x2, y2 = box

                self.last_bboxes.append((vid, x1, y1, x2, y2, cls))

                anchor_x = float((x1 + x2) / 2)
                anchor_y = float(y2)

                # ── Touch-line detection (vectorized) ─────────────────────
                touching_ids    = self.geom.touching_lines(anchor_x, anchor_y, self.touch_thr)
                touching_set    = set(touching_ids)
                active_touch_labels: list[str] = []

                for line_id in self._line_ids:
                    if line_id not in self.touch_states[vid]:
                        self.touch_states[vid][line_id] = TouchLineState(
                            debounce=self.touch_debounce
                        )
                    sm     = self.touch_states[vid][line_id]
                    signal = sm.update(line_id in touching_set)
                    key    = f"{vid}__touch__{line_id}"

                    if signal == "start":
                        self.open_touch_events[key] = {
                            "event_type" : "touch_line",
                            "vehicle_id" : vid,
                            "line_id"    : line_id,
                            "start_frame": frame_idx,
                            "start_time" : fmt_time(ts_sec),
                            "end_time"   : None,
                        }
                    elif signal == "end" and key in self.open_touch_events:
                        ev               = self.open_touch_events.pop(key)
                        ev["end_time"]   = fmt_time(ts_sec)
                        duration_sec     = (frame_idx - ev["start_frame"]) / fps
                        if duration_sec >= self.min_touch_duration_sec:
                            new_events.append(ev)
                            self.completed_events.append(ev)

                    if sm.in_touch:
                        active_touch_labels.append(line_id)

                # ── Lane-change detection (vectorized) ────────────────────
                current_lane = self.geom.lane_at(anchor_x, anchor_y)
                signal       = self.change_states[vid].update(current_lane, frame_idx)
                change_emit  = None
                change_key   = f"{vid}__change"

                if signal is not None:
                    sig_type = signal[0]
                    if sig_type == "crossing_start":
                        _, from_lane, start_idx = signal
                        self.open_change_events[change_key] = {
                            "event_type": "change_lane",
                            "vehicle_id": vid,
                            "from_lane" : from_lane,
                            "to_lane"   : None,
                            "start_time": fmt_time(start_idx / fps),
                            "end_time"  : None,
                        }
                    elif sig_type == "crossing_end":
                        _, from_lane, to_lane, start_idx, end_idx = signal
                        ev = self.open_change_events.pop(change_key, {
                            "event_type": "change_lane",
                            "vehicle_id": vid,
                            "from_lane" : from_lane,
                            "start_time": fmt_time(start_idx / fps),
                        })
                        ev["to_lane"]  = to_lane
                        ev["end_time"] = fmt_time(end_idx / fps)
                        new_events.append(ev)
                        self.completed_events.append(ev)
                        change_emit = ev

                # ── Annotation ────────────────────────────────────────────
                is_crossing  = self.change_states[vid].is_crossing
                signal_sides = (
                    self.ts_detector.get_active_signals(vid)
                    if self.ts_detector else []
                )

                color, label = self._build_label(
                    vid, cls, current_lane,
                    active_touch_labels, change_emit, is_crossing,
                    signal_sides=signal_sides,
                )
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, label, (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)
                cv2.circle(annotated, (int(anchor_x), int(anchor_y)), 3, (255, 50, 50), -1)

                if current_lane is not None:
                    cv2.putText(
                        annotated, f"Ln{current_lane}",
                        (x1, min(y2 + 16, frame.shape[0] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1,
                    )

        # ── Stale-vehicle cleanup (vehicles that left the frame) ──────────
        ts_exit = ts_sec
        for vid in list(self.touch_states.keys()):
            if vid in seen_vids:
                continue
            for line_id in self._line_ids:
                if line_id not in self.touch_states[vid]:
                    continue
                signal = self.touch_states[vid][line_id].update(False)
                key    = f"{vid}__touch__{line_id}"
                if signal == "end" and key in self.open_touch_events:
                    ev               = self.open_touch_events.pop(key)
                    ev["end_time"]   = fmt_time(ts_exit)
                    duration_sec     = (frame_idx - ev["start_frame"]) / fps
                    if duration_sec >= self.min_touch_duration_sec:
                        new_events.append(ev)
                        self.completed_events.append(ev)

        for ck in list(self.open_change_events.keys()):
            vid = ck.replace("__change", "")
            if vid not in seen_vids:
                ev               = self.open_change_events.pop(ck)
                ev["end_time"]   = fmt_time(ts_exit)
                if ev.get("to_lane") is None:
                    ev["to_lane"] = "?"
                if ev.get("from_lane") is not None:
                    new_events.append(ev)
                    self.completed_events.append(ev)

        # ── Turn signal processing ────────────────────────────────────────
        if self.ts_detector is not None:
            ts_evs = self.ts_detector.process_bboxes(
                frame, self.last_bboxes, frame_idx, fps
            )
            new_events.extend(ts_evs)

        return annotated, new_events

    # ─────────────────────────────────────────────────────────────────────────
    # Label builder
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_label(vid, cls, lane, touch_lines, change_emit,
                     is_crossing=False, signal_sides=None):
        base  = VEHICLE_CLASSES.get(cls, "vehicle")
        label = f"{vid} [{base}]"
        if touch_lines:
            color  = (0, 0, 220)
            label += " | TOUCH " + "+".join(touch_lines)
        elif change_emit is not None:
            color  = (0, 140, 255)
            label += f" | Ln{change_emit['from_lane']}->Ln{change_emit['to_lane']} confirmed"
        elif is_crossing:
            color  = (0, 200, 255)
            label += " | CHANGING LANE..."
        else:
            color = (50, 220, 50)
        if signal_sides:
            arrows = {"left": "<", "right": ">"}
            label += " | SIG " + "".join(arrows.get(s, s.upper()) for s in signal_sides)
        return color, label

    # ─────────────────────────────────────────────────────────────────────────
    # Finalize — close any open events at end of video
    # ─────────────────────────────────────────────────────────────────────────

    def finalize(self, total_frames: int, fps: float) -> list[dict]:
        end_time = fmt_time(total_frames / fps)
        leftover : list[dict] = []

        for key, ev in list(self.open_touch_events.items()):
            ev["end_time"] = end_time
            duration_sec   = (total_frames - ev["start_frame"]) / fps
            if duration_sec >= self.min_touch_duration_sec:
                leftover.append(ev)
                self.completed_events.append(ev)
        self.open_touch_events.clear()

        for key, ev in list(self.open_change_events.items()):
            ev["end_time"] = end_time
            if ev.get("to_lane") is None:
                ev["to_lane"] = "?"
            leftover.append(ev)
            self.completed_events.append(ev)
        self.open_change_events.clear()

        if self.ts_detector is not None:
            leftover.extend(self.ts_detector.finalize(total_frames, fps))

        return leftover

    def get_all_events(self) -> list[dict]:
        return list(self.completed_events)


# ─────────────────────────────────────────────────────────────────────────────
# CLI — quick standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json
    from video_handler import VideoHandler

    parser = argparse.ArgumentParser(description="EventTracker standalone test (CPU)")
    parser.add_argument("--video",         required=True)
    parser.add_argument("--lane_config",   required=True)
    parser.add_argument("--model",         default="yolov8n.pt")
    parser.add_argument("--output_video",  default="output_tracked.mp4")
    parser.add_argument("--output_json",   default="events.json")
    parser.add_argument("--stride",        type=int,   default=2)
    parser.add_argument("--touch_thr",     type=float, default=14.0)
    parser.add_argument("--stable",        type=int,   default=5)
    parser.add_argument("--min_touch_dur", type=float, default=0.5)
    args = parser.parse_args()

    with open(args.lane_config) as f:
        lane_config = json.load(f)

    tracker = EventTracker(
        model_path             = args.model,
        lane_config            = lane_config,
        touch_thr              = args.touch_thr,
        stable_frames          = args.stable,
        min_touch_duration_sec = args.min_touch_dur,
        turn_signal_detector   = TurnSignalDetector(),
    )

    all_events: list[dict] = []
    with VideoHandler(args.video, args.output_video, stride=args.stride) as vh:
        meta  = vh.get_metadata()
        fps   = meta["fps"]
        total = meta["total_frames"]
        print(f"[INFO] {total} frames | {fps} FPS | stride={args.stride} | device=cpu")

        for frame_idx, frame in vh.read_frames():
            annotated, new_evs = tracker.process_frame(frame, frame_idx, fps)
            vh.write_frame(annotated)
            if new_evs:
                for ev in new_evs:
                    print(f"  [{ev['start_time']}] {ev['event_type']:12s} | {ev['vehicle_id']}")
                all_events.extend(new_evs)

        all_events.extend(tracker.finalize(total, fps))

    with open(args.output_json, "w") as f:
        json.dump({"events": all_events}, f, indent=2)

    print(f"\n[DONE] {len(all_events)} events → {args.output_json}")
    print(f"       Video  → {args.output_video}")