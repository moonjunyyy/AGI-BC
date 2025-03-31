import os
import cv2
import numpy as np
from m00nny_utils.threads import Thread_With_Return_Value
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)

class _Mp4File:
    def __init__(self, filename):
        self.filename = filename
        if not os.path.isfile(self.filename): raise FileNotFoundError(f"{self.filename} not found")
        cap = cv2.VideoCapture(self.filename)
        if not cap.isOpened(): raise ValueError("Failed to open video file")
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if not os.path.isdir(self.filename.replace('.mp4', '')):
            os.makedirs(self.filename.replace('.mp4', ''))
            self._extract_frames()
            
    def _extract_frames(self):
        # excute ffmpeg command to extract frames
        os.system(f"ffmpeg -i {self.filename} {self.filename.replace('.mp4', '')}/%d.png")

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
        frames = [Thread_With_Return_Value(target=self._read_frame, args=(i,), daemon=True) for i in toread] 
        [frame.start() for frame in frames]
        frames = np.stack([frame.join() for frame in frames], axis=0)
        frames = frames.astype(np.float32) / 255.0
        frames = (frames - IMAGENET_MEAN) / IMAGENET_STD
        if len(frames) == 1: return frames[0]
        return frames

    def _read_frame(self, idx):
        frame_img = Image.open(f"{self.filename.replace('.mp4', '')}/{idx}.png")
        if frame_img is None: raise ValueError("Failed to read frame")
        frame = np.array(frame_img)
        del frame_img
        return frame

class _Mp4FileManager():
    def __new__(cls):
        if not hasattr(cls, '_instance'):
            cls._instance = super().__new__(cls)
            cls._instance._files = {}
            cls._instance._ref_count = {}
        return cls._instance
    
    def _get_file(self, filename):
        _cls = type(self)
        if filename not in _cls._instance._files:
            _cls._instance._files[filename] = _Mp4File(filename)
            _cls._instance._ref_count[filename] = 0
        _cls._instance._ref_count[filename] += 1
        return _cls._instance._files[filename]
    
    def _release_file(self, filename):
        _cls = type(self)
        if filename not in _cls._instance._files: return
        _cls._instance._ref_count[filename] -= 1
        if _cls._instance._ref_count[filename] <= 0:
            del _cls._instance._files[filename]
            del _cls._instance._ref_count[filename]

class Mp4File():
    def __init__(self, filename):
        self.filename = filename
        self._manager = _Mp4FileManager()
        self._file = self._manager._get_file(self.filename)

        self.total_frames = self._file.total_frames
        self.frame_rate = self._file.frame_rate
        self.width = self._file.width
        self.height = self._file.height
    def __len__(self): return len(self._file)
    def __getitem__(self, idx): return self._file[idx]
    def __del__(self): self._manager._release_file(self.filename)