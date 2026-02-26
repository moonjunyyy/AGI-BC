"""
Real-time streaming inference pipeline.
"""
import queue
import threading
from typing import Optional

import torch

from ..lm.base import S2SModel


class StreamingInferencePipeline:
    """Streaming inference pipeline wrapping an S2SModel.

    Audio chunks are read from audio_queue, inference runs in a background thread,
    and results are put into output_queue.

    Args:
        model: An S2SModel instance.
        device: Device string.
        chunk_ms: Chunk size in milliseconds (default 80ms @ 24kHz = 1920 samples).
        sample_rate: Audio sample rate (default 24000).
    """

    CHUNK_MS = 80
    SAMPLE_RATE = 24000

    def __init__(
        self,
        model: S2SModel,
        device: str = "cuda",
        chunk_ms: int = 80,
        sample_rate: int = 24000,
    ):
        self.model = model
        self.device = device
        self.chunk_ms = chunk_ms
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_ms / 1000)
        self._stop_event = threading.Event()

    def run(
        self,
        audio_queue: "queue.Queue[Optional[torch.Tensor]]",
        output_queue: "queue.Queue[dict]",
    ) -> None:
        """Start streaming inference loop.

        Reads PCM chunks from audio_queue (None = end of stream).
        Puts {"text": str, "audio": Tensor|None} into output_queue.
        Runs in the calling thread (wrap in threading.Thread if needed).
        """
        self._stop_event.clear()
        buffer = []

        def _frame_iter():
            """Yield accumulated audio frames."""
            while buffer:
                yield buffer.pop(0)

        while not self._stop_event.is_set():
            try:
                chunk = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if chunk is None:
                # End of stream: flush remaining buffer
                if buffer:
                    with torch.no_grad():
                        for result in self.model.generate_stream(_frame_iter()):
                            output_queue.put(result)
                break

            # Accumulate chunks; run inference when we have enough context
            if not isinstance(chunk, torch.Tensor):
                chunk = torch.from_numpy(chunk).float()
            if chunk.dim() == 1:
                chunk = chunk.unsqueeze(0).unsqueeze(0)
            elif chunk.dim() == 2:
                chunk = chunk.unsqueeze(0)
            buffer.append(chunk.to(self.device))

            # Run inference every N chunks (simple batching strategy)
            # Here we accumulate until we have at least 1 second of audio
            total_samples = sum(c.shape[-1] for c in buffer)
            if total_samples >= self.sample_rate:
                frames = list(buffer)
                buffer.clear()
                with torch.no_grad():
                    for result in self.model.generate_stream(iter(frames)):
                        output_queue.put(result)

        output_queue.put(None)  # Signal end

    def stop(self) -> None:
        """Stop the inference loop."""
        self._stop_event.set()

    def run_async(
        self,
        audio_queue: "queue.Queue[Optional[torch.Tensor]]",
        output_queue: "queue.Queue[dict]",
    ) -> threading.Thread:
        """Run inference in a background thread. Returns the thread."""
        t = threading.Thread(target=self.run, args=(audio_queue, output_queue), daemon=True)
        t.start()
        return t
