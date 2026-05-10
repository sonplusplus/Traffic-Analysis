# Traffic Analysis System

A computer vision pipeline that detects **lane-line touches**, **lane changes**, and **turn signals** from dashcam / CCTV traffic video — wrapped in a Streamlit web UI.

---


## 📁 Project Structure

```
lane-violation-detection/
│
├── app.py                  
├── lane_detector.py        
├── event_tracker.py        
├── turn_signal.py          
├── video_handler.py        
├── event_logger.py         
├── extract_frames.py       
├── split_data.py           
│
├── models/
│   ├── best_lane_seg.pt    
│   ├── yolov8n.pt          
│   └── best_signal_seg.pt 
│
├── input_videos/           
├── requirements.txt
└── README.md
```

---

##  Setup

### 1 — Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.10 |

Install **ffmpeg** (used for H.264 re-encoding):

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install -y ffmpeg

# macOS
brew install ffmpeg

# Windows — download from https://ffmpeg.org/download.html and add to PATH
```

---

### 2 — Clone & install Python dependencies

```bash
git clone https://github.com/<your-username>/lane-violation-detection.git
cd lane-violation-detection

# Create virtual env (recommend)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

> **GPU users:** replace the `torch` lines in `requirements.txt` with the CUDA build matching your driver:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```
> Note: the pipeline runs **CPU-only by default** to avoid CUDA memory conflicts between simultaneous YOLO models.

---

### 3 — Add model weights

Place the following files inside the `models/` folder:

| File | Purpose |
|---|---|
| `models/best_lane_seg.pt` | Lane segmentation |
| `models/yolov8n.pt` | Vehicle detection |
| `models/best_signal_seg.pt` | Turn signal detection |

```bash
mkdir -p models
# copy your .pt files into models/
```

---

### 4 — Add a test video

```bash
mkdir -p input_videos
cp /path/to/your/traffic_video.mp4 input_videos/your_video.mp4
```


---

##  Running the App

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### UI Workflow

1. **Upload** a traffic video (mp4 / avi / mov) — or leave blank to use the default demo.
2. Click **▶ Run Analysis**.
3. Watch the live annotated frame preview and running event count update in real time.
4. When processing finishes, a 2×2 results grid appears:
   - **① Input Video** — original playback
   - **② Lane & Line Detection** — static image with detected lines, lanes, and vanishing point
   - **③ Processed Event Video** — annotated output with colour-coded bounding boxes
   - **④ JSON Event Output** — structured event log with download button

---

## 🖥️ Command-Line Usage

Each module can also be run standalone without the UI.

### Lane detection only
```bash
python lane_detector.py \
    --video input_videos/video4.mp4 \
    --model models/best_lane_seg.pt \
    --output lane_result.jpg \
    --save_config lane_config.json
```

### Full event tracking (requires a pre-built lane config JSON)
```bash
python event_tracker.py \
    --video input_videos/video4.mp4 \
    --lane_config lane_config.json \
    --model models/yolov8n.pt \
    --output_video output_tracked.mp4 \
    --output_json events.json
```

### Turn signal detector (standalone debug)
```bash
python turn_signal.py \
    --video input_videos/video4.mp4 \
    --model models/best_signal_seg.pt \
    --output debug_ts.mp4 \
    --debug
```

### Extract frames for labelling
```bash
python extract_frames.py \
    --input_dir  input_videos \
    --output_dir frames_output \
    --every_n 10 \
    --max_per_video 200
```

---

## JSON Output Format

```json
{
  "source": "lane_violation_detection_v6",
  "generated": "2025-01-15T14:32:07",
  "summary": {
    "total_events": 12,
    "touch_line": 5,
    "change_lane": 4,
    "turn_signal": 3,
    "vehicles_seen": 6
  },
  "video_meta": {
    "fps": 25,
    "resolution": "1920x1080",
    "duration": "01:00",
    "total_frames": 1500,
    "stride": 2,
    "output_fps": 12
  },
  "events": [
    {
      "event_type": "touch_line",
      "vehicle_id": "V001",
      "line_id": "Line 2",
      "start_time": "00:06",
      "end_time": "00:07"
    },
    {
      "event_type": "change_lane",
      "vehicle_id": "V001",
      "from_lane": 2,
      "to_lane": 3,
      "start_time": "00:08",
      "end_time": "00:11"
    },
    {
      "event_type": "turn_signal",
      "vehicle_id": "V003",
      "signal": "right",
      "start_time": "00:25",
      "end_time": "00:40"
    }
  ]
}
```

---


## Key Configuration Parameters

All tunable in `app.py`:

| Parameter | Default | Description |
|---|---|---|
| `STRIDE` | `2` | Process every Nth frame (higher = faster, less accurate) |
| `N_BG_FRAMES` | `120` | Frames sampled for median background estimation |
| `TOUCH_THR` | `14.0` px | Pixel distance threshold for touch-line detection |
| `STABLE_FRAMES` | `5` | Frames needed to confirm a new lane assignment |
| `COOLDOWN_FRAMES` | `20` | Frames to suppress repeated events after one fires |
| `MIN_TOUCH_DUR` | `0.5` s | Minimum touch duration required to log an event |
| `CONF` | `0.30` | YOLO detection confidence threshold |

---


