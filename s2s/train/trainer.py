"""
S2STrainer: Training loop extending m00nny_utils _MetaTrainer.
"""
import os
import sys
import time
from typing import Optional

import torch
import torch.distributed as dist

_m00nny_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "m00nny_utils")
if _m00nny_path not in sys.path:
    sys.path.insert(0, os.path.dirname(_m00nny_path))

try:
    from m00nny_utils.util._trainer import _MetaTrainer
    from m00nny_utils.lr_scheduler.warmup_cosine_anneling import WarmUpCosineAnnelingScheduler
    _m00nny_available = True
except ImportError:
    _m00nny_available = False
    _MetaTrainer = object
    WarmUpCosineAnnelingScheduler = None

from ..lm.base import S2SModel
from ..utils.loader import S2SDataLoader


class S2STrainer(_MetaTrainer if _m00nny_available else object):
    """S2S training loop.

    Args:
        args: Namespace with fields:
            model_type: "omni2" or "moshi"
            model_config: dict for model __init__
            weights_dir: path to canonical safetensors weights (optional)
            data_path: path to dataset
            batch_size: int
            lr: float
            epochs: int
            warmup_steps: int
            save_dir: str
            device: str
            dtype: str
            world_size: int
            global_rank: int
            dist_backend: str
            dist_url: str
            dist_master_addr: str
            dist_master_port: str
            random_seed: int
            tp_degree: int (default 1)
    """

    def __init__(self, args):
        if _m00nny_available:
            super().__init__(args)
        else:
            self.args = args
            self.device = torch.device(getattr(args, "device", "cuda"))
            self.dtype = getattr(torch, getattr(args, "dtype", "float32"))
            self.world_size = getattr(args, "world_size", 1)
            self.global_rank = getattr(args, "global_rank", 0)
            os.makedirs(getattr(args, "save_dir", "checkpoints"), exist_ok=True)

        self.model: Optional[S2SModel] = None
        self.optimizer = None
        self.scheduler = None
        self.dataloader = None

    def build_model(self) -> S2SModel:
        """Build and optionally shard the model."""
        model_type = getattr(self.args, "model_type", "omni2")
        model_config = getattr(self.args, "model_config", {})
        weights_dir = getattr(self.args, "weights_dir", None)
        tp_degree = getattr(self.args, "tp_degree", 1)

        if model_type == "omni2":
            from ..lm.omni2 import Omni2Model
            if weights_dir:
                model = Omni2Model.from_safetensors(weights_dir, model_config, device=str(self.device))
            else:
                model = Omni2Model(model_config).to(self.device)
        elif model_type == "moshi":
            from ..lm.moshi import MoshiModel
            if weights_dir:
                model = MoshiModel.from_safetensors(weights_dir, model_config, device=str(self.device))
            else:
                model = MoshiModel(model_config).to(self.device)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Apply tensor parallelism if requested
        if tp_degree > 1:
            from ..utils.tp import shard_model
            model = shard_model(model, tp_degree)

        # Apply DDP if distributed
        if dist.is_initialized() and self.world_size > 1:
            if _m00nny_available:
                try:
                    from m00nny_utils.parallel.parameter_hook import ParameterHook
                    self._param_hook = ParameterHook(model)
                except Exception:
                    model = torch.nn.parallel.DistributedDataParallel(
                        model, device_ids=[self.local_rank]
                    )
            else:
                model = torch.nn.parallel.DistributedDataParallel(model)

        return model

    def build_dataloader(self) -> S2SDataLoader:
        """Build the S2S dataloader."""
        from ..utils.av import load_audio

        data_path = getattr(self.args, "data_path", "data")
        batch_size = getattr(self.args, "batch_size", 8)
        num_workers = getattr(self.args, "num_workers", 4)

        # Simple dataset: list of audio files
        audio_files = []
        if os.path.isdir(data_path):
            for fname in sorted(os.listdir(data_path)):
                if fname.endswith((".wav", ".mp3", ".flac")):
                    audio_files.append(os.path.join(data_path, fname))

        def transform(path: str) -> dict:
            audio = load_audio(path, sr=16000)
            return {"audio": audio, "text": "", "labels": torch.zeros(1, dtype=torch.long)}

        return S2SDataLoader(
            dataset=audio_files,
            transform=transform,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )

    def train_step(self, batch: dict) -> torch.Tensor:
        """Single training step."""
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = self.model(batch)
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad()
        return loss

    def val_step(self, batch: dict) -> dict:
        """Single validation step."""
        with torch.no_grad():
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = self.model(batch)
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in out.items()}

    def run(self) -> None:
        """Full training loop."""
        self.model = self.build_model()
        self.dataloader = self.build_dataloader()

        lr = getattr(self.args, "lr", 1e-4)
        epochs = getattr(self.args, "epochs", 10)
        warmup_steps = getattr(self.args, "warmup_steps", 100)
        save_dir = getattr(self.args, "save_dir", "checkpoints")

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)

        total_steps = epochs * len(self.dataloader)
        if WarmUpCosineAnnelingScheduler is not None:
            self.scheduler = WarmUpCosineAnnelingScheduler(
                self.optimizer, warmup_steps=warmup_steps, total_steps=total_steps
            )

        self.model.train()
        global_step = 0
        for epoch in range(epochs):
            for batch in self.dataloader:
                loss = self.train_step(batch)
                global_step += 1
                if global_step % 100 == 0:
                    rank = self.global_rank if hasattr(self, "global_rank") else 0
                    if rank == 0:
                        print(f"Epoch {epoch} step {global_step} loss {loss.item():.4f}")

            # Save checkpoint
            if hasattr(self, "global_rank") and self.global_rank == 0:
                ckpt_path = os.path.join(save_dir, f"epoch_{epoch:03d}.pt")
                torch.save({"model": self.model.state_dict(), "epoch": epoch}, ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")
