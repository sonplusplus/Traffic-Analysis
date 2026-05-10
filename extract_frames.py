import cv2
import os
import argparse
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".ts"}


def extract_frames(
    input_dir     : str,
    output_dir    : str,
    every_n       : int  = 10,    
    max_per_video : int  = 200,   
    uniform       : bool = True, 
):
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted([f for f in input_dir.iterdir() if f.suffix.lower() in VIDEO_EXTS])
    if not videos:
        print(f"[ERROR] No video files found in '{input_dir}'")
        return

    print(f"[INFO] Found {len(videos)} video(s) in '{input_dir}'")
    total_saved = 0

    for vid_path in videos:
        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            print(f"  [SKIP] Cannot open: {vid_path.name}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS)
        stem         = vid_path.stem  # dùng làm prefix tên file

        print(f"\n  [{vid_path.name}] {total_frames} frames | {fps:.1f} FPS")

        # Tính indices cần lấy
        if uniform:
            # Sample đều trong toàn video
            import numpy as np
            n_samples  = min(max_per_video, total_frames // every_n)
            n_samples  = max(1, n_samples)
            indices    = set(np.linspace(0, total_frames - 1, n_samples, dtype=int).tolist())
        else:
            # Lấy mỗi every_n frame liên tục
            indices = set(range(0, total_frames, every_n))
            if max_per_video:
                indices = set(sorted(indices)[:max_per_video])

        saved = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in indices:
                filename = output_dir / f"{stem}_f{frame_idx:06d}.jpg"
                cv2.imwrite(str(filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved += 1
            frame_idx += 1

        cap.release()
        total_saved += saved
        print(f"    → Saved {saved} frames to '{output_dir}'")

    print(f"\n[DONE] Total: {total_saved} frames saved to '{output_dir}'")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir",      default="input_videos",  help="Folder chứa video")
    p.add_argument("--output_dir",     default="frames_output", help="Folder lưu frames")
    p.add_argument("--every_n",        type=int,  default=10,   help="Lấy 1 frame mỗi N frame")
    p.add_argument("--max_per_video",  type=int,  default=200,  help="Tối đa frame mỗi video")
    p.add_argument("--sequential",     action="store_true",     help="Sequential thay vì uniform sampling")
    args = p.parse_args()

    extract_frames(
        input_dir     = args.input_dir,
        output_dir    = args.output_dir,
        every_n       = args.every_n,
        max_per_video = args.max_per_video,
        uniform       = not args.sequential,
    )