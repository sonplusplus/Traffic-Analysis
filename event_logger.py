import json
from datetime import datetime
from typing import Optional


def _time_to_seconds(t: str) -> float:
    """'MM:SS' → float seconds, used for sort key."""
    try:
        parts = t.split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except Exception:
        return 0.0


def _event_sort_key(ev: dict) -> tuple:
    """Primary: start_time. Secondary: event_type (stable sort within same second)."""
    ORDER = {"touch_line": 0, "change_lane": 1, "turn_signal": 2}
    return (
        _time_to_seconds(ev.get("start_time", "00:00")),
        ORDER.get(ev.get("event_type", ""), 9),
        ev.get("vehicle_id", ""),
    )



def build_event_log(
    events : list[dict],
    meta   : Optional[dict] = None,
    source : str = "lane_violation_detection_v6",
) -> dict:
    sorted_events = sorted(events, key=_event_sort_key)

    # Per-type counts
    type_counts: dict[str, int] = {}
    for ev in sorted_events:
        t = ev.get("event_type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    summary = {
        "total_events"  : len(sorted_events),
        "touch_line"    : type_counts.get("touch_line",  0),
        "change_lane"   : type_counts.get("change_lane", 0),
        "turn_signal"   : type_counts.get("turn_signal", 0),
        "vehicles_seen" : len({ev.get("vehicle_id", "") for ev in sorted_events}),
    }

    log: dict = {
        "source"    : source,
        "generated" : datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "summary"   : summary,
        "events"    : sorted_events,
    }

    if meta:
        log["video_meta"] = {
            "fps"         : meta.get("fps"),
            "resolution"  : f"{meta.get('width', '?')}x{meta.get('height', '?')}",
            "duration"    : meta.get("duration"),
            "total_frames": meta.get("total_frames"),
            "stride"      : meta.get("stride"),
            "output_fps"  : meta.get("output_fps"),
        }

    return log


def save_event_log(
    events      : list[dict],
    output_path : str,
    meta        : Optional[dict] = None,
    source      : str = "lane_violation_detection_v6",
) -> dict:
    """
    Build event log and write to JSON file.

    Returns the log dict (same as build_event_log).
    """
    log = build_event_log(events, meta=meta, source=source)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(
        f"[EventLogger] {log['summary']['total_events']} events → {output_path}  "
        f"(touch={log['summary']['touch_line']} "
        f"change={log['summary']['change_lane']} "
        f"signal={log['summary']['turn_signal']})"
    )
    return log


if __name__ == "__main__":
    sample = [
        {
            "event_type": "change_lane", "vehicle_id": "V001",
            "from_lane": 2, "to_lane": 3,
            "start_time": "00:08", "end_time": "00:11",
        },
        {
            "event_type": "touch_line", "vehicle_id": "V001",
            "line_id": "Line 2",
            "start_time": "00:06", "end_time": "00:07",
        },
        {
            "event_type": "turn_signal", "vehicle_id": "V003",
            "signal": "right",
            "start_time": "00:25", "end_time": "00:40",
        },
    ]
    sample_meta = {
        "fps": 25, "width": 1920, "height": 1080,
        "duration": "01:00", "total_frames": 1500,
        "stride": 2, "output_fps": 12,
    }
    log = build_event_log(sample, meta=sample_meta)
    print(json.dumps(log, indent=2))