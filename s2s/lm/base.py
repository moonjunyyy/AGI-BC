"""
Abstract S2SModel interface.
"""
import abc
from typing import Iterator, Optional

import torch
import torch.nn as nn


class S2SModel(nn.Module, abc.ABC):
    """Abstract base class for speech-to-speech models."""

    @abc.abstractmethod
    def generate_stream(
        self,
        audio_frames: Iterator[torch.Tensor],
        temperature: float = 1.0,
        max_new_tokens: int = 512,
    ) -> Iterator[dict]:
        """Stream inference: consume audio frames, yield {"text": str, "audio": Tensor | None}.

        Args:
            audio_frames: Iterator of [1, 1, T] audio tensors (one frame at a time).
            temperature: Sampling temperature.
            max_new_tokens: Maximum new tokens to generate.

        Yields:
            Dicts with "text" (str, possibly empty) and "audio" (Tensor [1,1,T] or None).
        """
        ...

    @abc.abstractmethod
    def forward(self, batch: dict) -> dict:
        """Training forward pass.

        Args:
            batch: {"audio": Tensor [B,1,T], "text": list[str], "labels": Tensor [B,S]}

        Returns:
            {"loss": Tensor, "logits": Tensor}
        """
        ...

    def load_checkpoint(self, path: str) -> None:
        """Load model state from safetensors directory."""
        from safetensors.torch import load_file
        import os
        for fname in os.listdir(path):
            if fname.endswith(".safetensors"):
                state = load_file(os.path.join(path, fname))
                missing, unexpected = self.load_state_dict(state, strict=False)
                if missing:
                    print(f"Missing keys from {fname}: {missing[:5]}...")
                if unexpected:
                    print(f"Unexpected keys from {fname}: {unexpected[:5]}...")
