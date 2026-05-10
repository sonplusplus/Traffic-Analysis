import cv2
import os
from typing import Generator, Tuple
import numpy as np


class VideoHandler:
    def __init__(self, input_path: str, output_path: str = "output_video.mp4", stride: int = 2):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Cannot find video: '{input_path}'")

        self.input_path  = input_path
        self.output_path = output_path
        self.stride      = stride

        self.cap = cv2.VideoCapture(input_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: '{input_path}' (corrupt or unsupported codec)")

        # Metadata
        self.fps          = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width        = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height       = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_sec = self.total_frames / self.fps if self.fps > 0 else 0


        output_fps = max(1, self.fps // self.stride)
        fourcc     = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, output_fps, (self.width, self.height))

    def get_metadata(self) -> dict:
        """Trả về metadata của video để hiển thị trong UI."""
        mins = int(self.duration_sec // 60)
        secs = int(self.duration_sec % 60)
        return {
            "fps":          self.fps,
            "width":        self.width,
            "height":       self.height,
            "total_frames": self.total_frames,
            "duration":     f"{mins:02d}:{secs:02d}",
            "stride":       self.stride,
            "output_fps":   max(1, self.fps // self.stride),
        }

    def read_frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if frame_idx % self.stride == 0:
                yield frame_idx, frame
            frame_idx += 1

    def frame_to_timestamp(self, frame_idx: int) -> str:
        """Convert frame index → timestamp string MM:SS."""
        sec  = frame_idx / self.fps if self.fps > 0 else 0
        mins = int(sec // 60)
        secs = int(sec % 60)
        return f"{mins:02d}:{secs:02d}"

    def write_frame(self, frame: np.ndarray):
        """Ghi frame đã xử lý vào output video."""
        self.writer.write(frame)

    def release(self):
        """Giải phóng tài nguyên."""
        self.cap.release()
        self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()