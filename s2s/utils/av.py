"""
Audio I/O helpers — wraps ffmpeg for PCM/Opus conversion.
"""
import io
import subprocess
import typing as tp

import torch
import torchaudio


def load_audio(path: str, sr: int = 16000) -> torch.Tensor:
    """Load audio file and resample to target sample rate.

    Args:
        path: Path to audio file.
        sr: Target sample rate.

    Returns:
        [1, T] float32 tensor, normalized to [-1, 1].
    """
    waveform, orig_sr = torchaudio.load(path)
    if orig_sr != sr:
        waveform = torchaudio.functional.resample(waveform, orig_sr, sr)
    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def save_audio(tensor: torch.Tensor, path: str, sr: int = 24000) -> None:
    """Save audio tensor to file.

    Args:
        tensor: [1, T] or [T] float32 tensor.
        path: Output path (wav, mp3, etc.).
        sr: Sample rate.
    """
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    torchaudio.save(path, tensor.cpu().float(), sr)


def bytes_to_tensor(raw: bytes, sr: int = 24000, channels: int = 1) -> torch.Tensor:
    """Convert raw PCM bytes (16-bit signed LE) to float32 tensor.

    Args:
        raw: Raw PCM bytes.
        sr: Sample rate (unused, just for interface consistency).
        channels: Number of channels.

    Returns:
        [channels, T] float32 tensor, normalized to [-1, 1].
    """
    import numpy as np
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        arr = arr.reshape(-1, channels).T
        return torch.from_numpy(arr.copy())
    return torch.from_numpy(arr).unsqueeze(0)


def tensor_to_bytes(t: torch.Tensor, sr: int = 24000) -> bytes:
    """Convert float32 tensor to raw PCM bytes (16-bit signed LE).

    Args:
        t: [1, T] or [T] float32 tensor, values in [-1, 1].
        sr: Sample rate (unused).

    Returns:
        Raw PCM bytes (int16 LE).
    """
    import numpy as np
    if t.dim() > 1:
        t = t.squeeze(0)
    arr = (t.cpu().float().clamp(-1.0, 1.0).numpy() * 32767).astype(np.int16)
    return arr.tobytes()


def pcm_to_opus(pcm_bytes: bytes, sr: int = 24000, channels: int = 1) -> bytes:
    """Convert raw PCM to Opus via ffmpeg pipe.

    Args:
        pcm_bytes: Raw 16-bit LE PCM bytes.
        sr: Sample rate.
        channels: Number of channels.

    Returns:
        Opus-encoded bytes.
    """
    cmd = [
        "ffmpeg", "-f", "s16le", "-ar", str(sr), "-ac", str(channels),
        "-i", "pipe:0", "-c:a", "libopus", "-f", "ogg", "pipe:1",
        "-loglevel", "quiet",
    ]
    proc = subprocess.run(cmd, input=pcm_bytes, capture_output=True)
    return proc.stdout


def opus_to_pcm(opus_bytes: bytes, sr: int = 24000, channels: int = 1) -> bytes:
    """Convert Opus bytes to raw PCM via ffmpeg pipe.

    Args:
        opus_bytes: Opus-encoded bytes (ogg container).
        sr: Target sample rate.
        channels: Number of channels.

    Returns:
        Raw 16-bit LE PCM bytes.
    """
    cmd = [
        "ffmpeg", "-f", "ogg", "-i", "pipe:0",
        "-f", "s16le", "-ar", str(sr), "-ac", str(channels),
        "pipe:1", "-loglevel", "quiet",
    ]
    proc = subprocess.run(cmd, input=opus_bytes, capture_output=True)
    return proc.stdout
