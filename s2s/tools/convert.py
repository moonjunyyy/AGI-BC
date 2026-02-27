"""
One-shot weight converter: mimi / omni2 / moshi_lm → canonical safetensors layout.

Usage:
    python -m s2s.tools.convert omni2 --src /path/to/omni2 --dst ./weights/omni2
    python -m s2s.tools.convert moshi --src /path/to/moshi --dst ./weights/moshi
    python -m s2s.tools.convert mimi --src /path/to/moshi --dst ./weights/moshi
"""
import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict

import torch
from safetensors.torch import load_file, save_file


# ---------------------------------------------------------------------------
# Mimi converter
# ---------------------------------------------------------------------------

def convert_mimi(src: str, dst: str) -> None:
    """Convert moshi mimi codec checkpoint to canonical mimi.safetensors.

    The moshi checkpoint contains keys like `encoder.*`, `decoder.*`, `quantizer.*`
    which map directly to our seanet.py/vq.py key paths.
    """
    os.makedirs(dst, exist_ok=True)
    src_path = Path(src)

    # Load from safetensors or pytorch checkpoint
    if (src_path / "model.safetensors").exists():
        state = load_file(str(src_path / "model.safetensors"))
    elif (src_path / "pytorch_model.bin").exists():
        state = torch.load(str(src_path / "pytorch_model.bin"), map_location="cpu", weights_only=True)
    else:
        # Try to find any .safetensors file
        sf_files = list(src_path.glob("*.safetensors"))
        if sf_files:
            state = load_file(str(sf_files[0]))
        else:
            raise FileNotFoundError(f"No checkpoint found in {src}")

    # Extract mimi-related keys
    mimi_keys = {k: v for k, v in state.items()
                 if any(k.startswith(p) for p in ["encoder.", "decoder.", "quantizer.",
                                                    "encoder_transformer.", "decoder_transformer.",
                                                    "downsample.", "upsample."])}

    if not mimi_keys:
        # Maybe the whole checkpoint is the mimi model
        mimi_keys = state

    save_file(mimi_keys, str(Path(dst) / "mimi.safetensors"))
    print(f"Saved {len(mimi_keys)} tensors to {dst}/mimi.safetensors")


# ---------------------------------------------------------------------------
# Omni2 converter
# ---------------------------------------------------------------------------

def convert_omni2(src: str, dst: str) -> None:
    """Convert llama-omni2 HF checkpoint to canonical safetensors layout.

    Splits into:
      - whisper_encoder.safetensors  (speech_encoder.* keys, strip prefix)
      - speech_projector.safetensors (speech_projector.* keys, strip prefix)
      - qwen2_lm.safetensors         (model.* keys, strip "model." prefix)
      - speech_generator.safetensors (speech_generator.* keys, strip prefix)
    """
    os.makedirs(dst, exist_ok=True)
    src_path = Path(src)

    # Load sharded or single safetensors
    state: Dict[str, torch.Tensor] = {}
    index_file = src_path / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        loaded_shards = set()
        for key, shard_name in weight_map.items():
            if shard_name not in loaded_shards:
                shard_path = src_path / shard_name
                shard_state = load_file(str(shard_path))
                state.update(shard_state)
                loaded_shards.add(shard_name)
    elif (src_path / "model.safetensors").exists():
        state = load_file(str(src_path / "model.safetensors"))
    else:
        raise FileNotFoundError(f"No safetensors checkpoint found in {src}")

    # Split by prefix
    whisper_enc: Dict[str, torch.Tensor] = {}
    projector: Dict[str, torch.Tensor] = {}
    qwen2_lm: Dict[str, torch.Tensor] = {}
    speech_gen: Dict[str, torch.Tensor] = {}

    for key, tensor in state.items():
        if key.startswith("speech_encoder."):
            new_key = key[len("speech_encoder."):]
            whisper_enc[new_key] = tensor
        elif key.startswith("speech_projector."):
            new_key = key[len("speech_projector."):]
            projector[new_key] = tensor
        elif key.startswith("speech_generator."):
            new_key = key[len("speech_generator."):]
            speech_gen[new_key] = tensor
        elif key.startswith("model."):
            new_key = key[len("model."):]
            qwen2_lm[new_key] = tensor
        elif key.startswith("lm_head."):
            qwen2_lm[key] = tensor

    dst_path = Path(dst)
    if whisper_enc:
        save_file(whisper_enc, str(dst_path / "whisper_encoder.safetensors"))
        print(f"Saved {len(whisper_enc)} tensors → whisper_encoder.safetensors")
    if projector:
        save_file(projector, str(dst_path / "speech_projector.safetensors"))
        print(f"Saved {len(projector)} tensors → speech_projector.safetensors")
    if qwen2_lm:
        save_file(qwen2_lm, str(dst_path / "qwen2_lm.safetensors"))
        print(f"Saved {len(qwen2_lm)} tensors → qwen2_lm.safetensors")
    if speech_gen:
        save_file(speech_gen, str(dst_path / "speech_generator.safetensors"))
        print(f"Saved {len(speech_gen)} tensors → speech_generator.safetensors")

    # Copy tokenizer files so AutoTokenizer.from_pretrained(dst) works
    import shutil
    _TOKENIZER_FILES = [
        "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "vocab.json", "merges.txt",
    ]
    for fname in _TOKENIZER_FILES:
        src_file = src_path / fname
        if src_file.exists():
            shutil.copy2(str(src_file), str(dst_path / fname))
            print(f"Copied {fname}")


# ---------------------------------------------------------------------------
# Moshi LM converter
# ---------------------------------------------------------------------------

def convert_moshi_lm(src: str, dst: str) -> None:
    """Convert kyutai/moshiko-pytorch-bf16 to canonical moshi_lm.safetensors.

    The moshi LM checkpoint contains keys like `lm.*`. We strip the `lm.` prefix
    to match our s2s/lm/moshi.py key paths.
    """
    os.makedirs(dst, exist_ok=True)
    src_path = Path(src)

    # Load checkpoint
    if (src_path / "model.safetensors").exists():
        state = load_file(str(src_path / "model.safetensors"))
    else:
        sf_files = list(src_path.glob("*.safetensors"))
        if sf_files:
            state = load_file(str(sf_files[0]))
        else:
            raise FileNotFoundError(f"No checkpoint found in {src}")

    # Strip `lm.` prefix if present
    new_state = {}
    for key, tensor in state.items():
        if key.startswith("lm."):
            new_state[key[len("lm."):]] = tensor
        else:
            new_state[key] = tensor

    save_file(new_state, str(Path(dst) / "moshi_lm.safetensors"))
    print(f"Saved {len(new_state)} tensors → moshi_lm.safetensors")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="S2S weight converter")
    parser.add_argument("model", choices=["mimi", "omni2", "moshi"], help="Which model to convert")
    parser.add_argument("--src", required=True, help="Source checkpoint directory")
    parser.add_argument("--dst", required=True, help="Destination directory for converted weights")
    args = parser.parse_args()

    if args.model == "mimi":
        convert_mimi(args.src, args.dst)
    elif args.model == "omni2":
        convert_omni2(args.src, args.dst)
    elif args.model == "moshi":
        convert_moshi_lm(args.src, args.dst)
        # Also convert mimi from the same source
        convert_mimi(args.src, args.dst)


if __name__ == "__main__":
    main()
