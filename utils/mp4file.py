import os
import cv2
import numpy as np
from utils.threads import Thread_With_Return_Value

class Mp4File():
    def __init__(self, filename):
        self.filename = filename
        cap = cv2.VideoCapture(self.filename)
        if not cap.isOpened():
            raise ValueError("Failed to open video file")
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if not os.path.isdir(self.filename.replace('.mp4', '')):
            os.makedirs(self.filename.replace('.mp4', ''))
            self.extract_frames()
            
    def extract_frames(self):
        # excute ffmpeg command to extract frames
        os.system(f"ffmpeg -i {self.filename} {self.filename.replace('.mp4', '')}/%d.jpg")

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx) -> np.ndarray:
        if isinstance(idx, int):
            if idx < 0 or idx >= self.total_frames:
                raise IndexError("Frame index out of range")
            toread = range(idx, idx + 1)
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(self.total_frames)
            toread = range(start, stop, step)
        elif isinstance(idx, list) or isinstance(idx, tuple) or isinstance(idx, np.ndarray):
            if any([i < 0 or i >= self.total_frames for i in idx]):
                raise IndexError("Frame index out of range")
            toread = idx
        else:
            raise ValueError("Invalid index type")
        frames = [Thread_With_Return_Value(target=self.read_frame, args=(i,)) for i in toread] 
        [frame.start() for frame in frames]
        frames = np.stack([frame.join() for frame in frames], axis=0)
        if len(frames) == 1:
            return frames[0]
        else:
            return np.array(frames)

    def read_frame(self, idx):
        frame = cv2.imread(f"{self.filename.replace('.mp4', '')}/{idx}.jpg")
        if frame is None:
            raise ValueError("Failed to read frame")
        frame = np.array(frame)
        return frame