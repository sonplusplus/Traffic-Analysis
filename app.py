import os
import json
import tempfile
import subprocess
import cv2
import streamlit as st
from lane_detector import detect_lanes
from video_handler import VideoHandler
from event_tracker import EventTracker
from turn_signal   import TurnSignalDetector
from event_logger  import build_event_log, save_event_log



DEFAULT_VIDEO     = "input_videos/video4.mp4"
LANE_MODEL        = "models/best_lane_seg.pt"
VEHICLE_MODEL     = "models/yolov8n.pt"
SIGNAL_MODEL      = "models/best_signal_seg.pt"

STRIDE            = 2
N_BG_FRAMES       = 120
TOUCH_THR         = 14.0
STABLE_FRAMES     = 5
COOLDOWN_FRAMES   = 20
MIN_TOUCH_DUR     = 0.5
CONF              = 0.30
LIVE_UPDATE_EVERY = 10   # update live preview every N processed frames

def _try_ffmpeg(src: str, dst: str) -> bool:
    """Re-encode with ffmpeg → H.264. Returns True on success."""
    try:
        r = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", src,
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "23",
                "-preset", "fast",
                "-movflags", "+faststart",
                "-an",
                dst,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=300,
        )
        return r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 1000
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _try_cv2_avc1(src: str, dst: str) -> bool:
    """Re-encode with OpenCV avc1 codec. Returns True on success."""
    try:
        cap    = cv2.VideoCapture(src)
        fps    = cap.get(cv2.CAP_PROP_FPS) or 12
        w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(dst, fourcc, fps, (w, h))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        cap.release()
        writer.release()
        return os.path.exists(dst) and os.path.getsize(dst) > 1000
    except Exception:
        return False


def make_browser_video(src: str) -> str:
    """
    Try to produce a browser-playable H.264 mp4.
    Priority: ffmpeg → cv2 avc1 → original (may not play in browser).
    """
    dst_ffmpeg = src.replace(".mp4", "_h264.mp4")
    dst_avc1   = src.replace(".mp4", "_avc1.mp4")

    if _try_ffmpeg(src, dst_ffmpeg):
        return dst_ffmpeg
    if _try_cv2_avc1(src, dst_avc1):
        return dst_avc1
    return src



st.set_page_config(
    page_title = "Lane Violation Detection",
    page_icon  = "🚦",
    layout     = "wide",
)

st.title("🚦 Lane Violation Detection System")
st.caption("Candidate code: CGS26_A100 — Cyber Gate Shields Technical Assessment")
st.divider()




for key in ("results", "error"):
    if key not in st.session_state:
        st.session_state[key] = None



up_col, btn_col = st.columns([3, 1], vertical_alignment="bottom")

with up_col:
    uploaded_file = st.file_uploader(
        "Upload traffic video (mp4 / avi / mov)",
        type=["mp4", "avi", "mov"],
        help=f"Leave empty to use default demo video ({DEFAULT_VIDEO})",
    )

with btn_col:
    run_clicked = st.button("▶ Run Analysis", type="primary", use_container_width=True)



if run_clicked:
    st.session_state.results = None
    st.session_state.error   = None

    tmp_input_path = None
    if uploaded_file is not None:
        tmp_in = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp_in.write(uploaded_file.read())
        tmp_in.flush()
        tmp_in.close()
        video_path     = tmp_in.name
        tmp_input_path = tmp_in.name
    else:
        video_path = DEFAULT_VIDEO
        if not os.path.exists(video_path):
            st.error(f"`{DEFAULT_VIDEO}` not found. Please upload a video.")
            st.stop()

    tmp_vid  = tempfile.NamedTemporaryFile(suffix=".mp4",  delete=False, prefix="out_")
    tmp_json = tempfile.NamedTemporaryFile(suffix=".json", delete=False, prefix="out_", mode="w")
    output_video_path = tmp_vid.name
    output_json_path  = tmp_json.name
    tmp_vid.close()
    tmp_json.close()


    progress_bar = st.progress(0, text="Initializing…")
    status_box   = st.empty()

    st.markdown("---")
    st.markdown("#### Live Processing")
    live_col1, live_col2 = st.columns(2, gap="medium")

    with live_col1:
        st.caption(" Annotated frame (updates every 10 frames)")
        frame_ph = st.empty()

    with live_col2:
        st.caption(" Events detected so far")
        event_ph = st.empty()

    end_live_marker = st.empty()   

    try:
        status_box.info("**Stage 1 / 3** — Lane geometry detection…  "
                        "(median background → CLAHE → YOLO-seg → VP clustering)")
        vis_img, lane_config = detect_lanes(
            video_path = video_path,
            model_path = LANE_MODEL,
            n_frames   = N_BG_FRAMES,
        )
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        n_lines = len(lane_config.get("lines", []))
        n_lanes = len(lane_config.get("lanes", []))
        progress_bar.progress(20, text=f"Lane detection done — {n_lines} lines / {n_lanes} lanes")

        status_box.info("**Stage 2 / 3** — Vehicle tracking + event detection…")

        seg_path    = SIGNAL_MODEL if os.path.exists(SIGNAL_MODEL) else None
        ts_detector = TurnSignalDetector(seg_model_path=seg_path)
        tracker     = EventTracker(
            model_path             = VEHICLE_MODEL,
            lane_config            = lane_config,
            touch_thr              = TOUCH_THR,
            stable_frames          = STABLE_FRAMES,
            cooldown_frames        = COOLDOWN_FRAMES,
            min_touch_duration_sec = MIN_TOUCH_DUR,
            conf                   = CONF,
            turn_signal_detector   = ts_detector,
        )

        all_events = []
        meta       = {}
        proc_count = 0

        with VideoHandler(video_path, output_video_path, stride=STRIDE) as vh:
            meta  = vh.get_metadata()
            fps   = meta["fps"]
            total = meta["total_frames"]

            for frame_idx, frame in vh.read_frames():
                annotated, new_evs = tracker.process_frame(frame, frame_idx, fps)
                vh.write_frame(annotated)
                all_events.extend(new_evs)
                proc_count += 1

                if proc_count % LIVE_UPDATE_EVERY == 0 or new_evs:
                    pct = 20 + int(68 * frame_idx / max(total, 1))
                    progress_bar.progress(
                        min(pct, 88),
                        text=f"Frame {frame_idx} / {total}   ·   {len(all_events)} events detected",
                    )

                    h_disp = 360
                    w_disp = int(annotated.shape[1] * h_disp / annotated.shape[0])
                    preview = cv2.cvtColor(
                        cv2.resize(annotated, (w_disp, h_disp)),
                        cv2.COLOR_BGR2RGB,
                    )
                    frame_ph.image(preview, use_container_width=True)

                    recent = list(reversed(all_events[-15:])) if all_events else []
                    event_ph.json(
                        {"events_so_far": len(all_events), "recent": recent},
                        expanded=bool(recent),
                    )

            all_events.extend(tracker.finalize(total, fps))

        frame_ph.empty()
        event_ph.empty()
        end_live_marker.divider()

        status_box.info(" **Stage 3 / 3** — Encoding video for browser + exporting JSON…")
        progress_bar.progress(90, text="Re-encoding video…")
        playback_path = make_browser_video(output_video_path)

        progress_bar.progress(96, text="Saving event log…")
        event_log = save_event_log(all_events, output_json_path, meta=meta)

        progress_bar.progress(100, text=" Done!")
        s = event_log["summary"]
        status_box.success(
            f" Analysis complete — **{s['total_events']} events**  "
            f"(touch **{s['touch_line']}** · "
            f"change **{s['change_lane']}** · "
            f"signal **{s['turn_signal']}**)"
        )

        st.session_state.results = {
            "vis_img"      : vis_img,
            "video_path"   : video_path,
            "playback_path": playback_path,
            "raw_out_path" : output_video_path,
            "event_log"    : event_log,
            "json_path"    : output_json_path,
            "meta"         : meta,
        }

    except Exception as exc:
        progress_bar.empty()
        st.session_state.error = str(exc)
        st.error(f"Pipeline failed: {exc}")
        raise

if st.session_state.results:
    r = st.session_state.results
    st.divider()

    # Metadata strip
    m = r.get("meta", {})
    if m:
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Resolution",   f"{m.get('width')}×{m.get('height')}")
        mc2.metric("Duration",     m.get("duration", "—"))
        mc3.metric("Source FPS",   m.get("fps", "—"))
        mc4.metric("Output FPS",   m.get("output_fps", "—"))
        mc5.metric("Total Events", r["event_log"]["summary"]["total_events"])
        st.divider()

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.subheader("① Input Video")
        try:
            with open(r["video_path"], "rb") as f:
                st.video(f.read())
        except Exception:
            st.warning("Input video preview unavailable.")

    with col2:
        st.subheader("② Lane & Line Detection")
        st.image(r["vis_img"][:, :, ::-1], use_container_width=True)
        st.caption("Line IDs (red) · Lane IDs (yellow) · Vanishing point (white dot)")

    st.divider()

    col3, col4 = st.columns(2, gap="medium")

    with col3:
        st.subheader("③ Processed Event Video")

        playback_ok = False
        try:
            video_bytes = open(r["playback_path"], "rb").read()
            if len(video_bytes) > 5000:
                st.video(video_bytes)
                playback_ok = True
                st.caption("🟢 Normal · 🟠 Crossing · 🔴 Touch line · 🔵 Signal active")
        except Exception:
            pass

        if not playback_ok:
            st.warning(
                " Browser playback failed (codec issue).  \n"
                "Download the video below and open with **VLC** or any media player."
            )

        try:
            with open(r["raw_out_path"], "rb") as f:
                st.download_button(
                    label     = "⬇ Download processed video (.mp4)",
                    data      = f,
                    file_name = "processed_events.mp4",
                    mime      = "video/mp4",
                    use_container_width = True,
                )
        except Exception:
            pass

    with col4:
        st.subheader("④ JSON Event Output")

        s = r["event_log"]["summary"]
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Touch Line",  s["touch_line"])
        sc2.metric("Change Lane", s["change_lane"])
        sc3.metric("Turn Signal", s["turn_signal"])
        sc4.metric("Vehicles",    s["vehicles_seen"])

        st.json(r["event_log"], expanded=False)

        with open(r["json_path"], "rb") as f:
            st.download_button(
                label     = "⬇ Download events.json",
                data      = f,
                file_name = "events.json",
                mime      = "application/json",
                use_container_width = True,
            )


st.divider()
st.caption(
    "Pipeline: Temporal Median BG → CLAHE → YOLOv8-seg (lane) → VP clustering → "
    "YOLOv8n + ByteTrack (vehicle) → Touch / Change state machines → "
    "YOLOv8s-seg + Brightness Oscillation (turn signal)"
)