import os
import torch
import ffmpeg
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)


class _Mp4File:
    def __init__(self, filename):
        self.filename = filename
        if not os.path.isfile(self.filename):
            raise FileNotFoundError(f"{self.filename} not found")
        probe = ffmpeg.probe(self.filename)
        video_stream = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
            None,
        )
        self.width = int(video_stream["width"])
        self.frame_rate = eval(video_stream["r_frame_rate"])
        self.total_frames = int(video_stream["nb_frames"])
        self.height = int(video_stream["height"])

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
        elif (
            isinstance(idx, list)
            or isinstance(idx, tuple)
            or isinstance(idx, np.ndarray)
        ):
            if any([i < 0 or i >= self.total_frames for i in idx]):
                raise IndexError("Frame index out of range")
            toread = idx
        else:
            raise ValueError("Invalid index type")
        ffmpeg_query = " + ".join([f"eq(n,{i})" for i in toread])
        out, _ = (
            ffmpeg
            .input(self.filename)
            .filter(
                "select",
                ffmpeg_query,
            )
            .output(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
            )
            .run(
                capture_stdout=True,
                capture_stderr=True,
            )
        )
        video = (
            np
            .frombuffer(out, np.uint8)
            .reshape([-1, self.height, self.width, 3])
        ).astype(np.float32) / 255.0
        video = (video - IMAGENET_MEAN) / IMAGENET_STD
        video = torch.from_numpy(video)
        return video


class _Mp4FileManager:
    def __new__(cls):
        if not hasattr(cls, "_instance"):
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
        if filename not in _cls._instance._files:
            return
        _cls._instance._ref_count[filename] -= 1
        if _cls._instance._ref_count[filename] <= 0:
            del _cls._instance._files[filename]
            del _cls._instance._ref_count[filename]


class Mp4File:
    def __init__(self, filename):
        self.filename = filename
        self._manager = _Mp4FileManager()
        self._file = self._manager._get_file(self.filename)

        self.total_frames = self._file.total_frames
        self.frame_rate = self._file.frame_rate
        self.width = self._file.width
        self.height = self._file.height

    def __len__(self):
        return len(self._file)

    def __getitem__(self, idx):
        return self._file[idx]

    def __del__(self):
        self._manager._release_file(self.filename)
