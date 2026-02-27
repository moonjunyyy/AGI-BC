"""
Tensor-parallel worker pool for single-machine multi-GPU inference.

Architecture
------------
  Main process
    ├─ rank_queues[0..N-1]  (mp.Queue, one per worker)
    └─ output_queue         (mp.Queue, rank-0 worker → main)

  Worker-0 (cuda:0)  ──┐
  Worker-1 (cuda:1)  ──┤  NCCL all-reduce / all-gather (in-process NCCL)
  Worker-2 (cuda:2)  ──┤
  Worker-3 (cuda:3)  ──┘

Each inference call:
  1. Main puts (audio_cpu, prompt, …) on ALL rank_queues simultaneously.
  2. All workers wake up and run generate_stream() in lock-step —
     the NCCL collectives inside ShardedLinear keep them synchronised.
  3. Rank-0 puts the result on output_queue; main blocks on get().

The threading.Lock in TensorParallelPool.infer() ensures only one
request is in-flight at a time, avoiding NCCL ordering issues.

No torchrun, no accelerate, no external launcher required.
"""
import os
import sys
import threading
from typing import Iterator, Optional

import torch
import torch.multiprocessing as mp


# ---------------------------------------------------------------------------
# Worker entry-point (runs in each child process)
# ---------------------------------------------------------------------------

def _worker_entry(
    rank: int,
    world_size: int,
    weights_dir: str,
    config: dict,
    dtype_str: str,
    rank_queue: mp.Queue,
    output_queue: mp.Queue,
    master_addr: str,
    master_port: str,
) -> None:
    """Initialise dist, load the TP-sharded model, serve inference requests."""
    # Make the s2s package importable inside the spawned process
    root = os.path.dirname(  # AGI-BC/
        os.path.dirname(      # s2s/
            os.path.dirname(  # utils/
                os.path.abspath(__file__)
            )
        )
    )
    if root not in sys.path:
        sys.path.insert(0, root)

    import torch
    import torch.distributed as dist
    from s2s.utils.log import get_logger

    log = get_logger(f"s2s.tp_worker[{rank}]")

    # ── dist init ──────────────────────────────────────────────────────────
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )
    log.info(f"dist initialised  rank={rank}/{world_size}  device={device}")

    # ── model load ─────────────────────────────────────────────────────────
    from s2s.lm.omni2 import Omni2Model
    dtype = getattr(torch, dtype_str, torch.float32)

    model = Omni2Model.from_safetensors(
        weights_dir, config, device=device, tp_degree=world_size
    )
    model.to(dtype).eval()
    log.info("model ready")

    # ── inference loop ─────────────────────────────────────────────────────
    while True:
        item = rank_queue.get()
        if item is None:          # shutdown sentinel
            break

        audio_cpu, text_prompt, max_new_tokens, temperature = item
        audio = audio_cpu.to(device)

        result: dict = {"text": "", "audio": None}
        try:
            with torch.no_grad():
                for r in model.generate_stream(
                    iter([audio]),
                    text_prompt=text_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                ):
                    result = r
                    break
        except Exception as exc:
            log.warning(f"generate_stream error: {exc}")
            result = {"text": "", "audio": None, "error": str(exc)}

        # Only rank-0 sends the result back to the main process
        if rank == 0:
            if result.get("audio") is not None:
                result["audio"] = result["audio"].cpu()
            output_queue.put(result)

    dist.destroy_process_group()
    log.info("worker shut down")


# ---------------------------------------------------------------------------
# Pool — public API used by the server
# ---------------------------------------------------------------------------

class TensorParallelPool:
    """Manage N worker processes for tensor-parallel inference.

    Exposes generate_stream() so it is a drop-in replacement for any S2SModel
    inside the existing server / eval pipeline.

    Args:
        weights_dir:  Directory with safetensors shards.
        config:       Model config dict (same as Omni2Model.from_safetensors).
        tp_degree:    Number of GPUs / worker processes.
        dtype:        Weight dtype string, e.g. "bfloat16".
        master_addr:  NCCL rendezvous address (default localhost).
        master_port:  NCCL rendezvous port (default 29500).
    """

    def __init__(
        self,
        weights_dir: str,
        config: dict,
        tp_degree: int,
        dtype: str = "bfloat16",
        master_addr: str = "127.0.0.1",
        master_port: str = "29500",
    ) -> None:
        ctx = mp.get_context("spawn")
        self.tp_degree   = tp_degree
        self.rank_queues = [ctx.Queue() for _ in range(tp_degree)]
        self.output_queue = ctx.Queue()
        self._lock = threading.Lock()  # one in-flight request at a time

        self.processes = []
        for rank in range(tp_degree):
            p = ctx.Process(
                target=_worker_entry,
                args=(
                    rank, tp_degree,
                    weights_dir, config, dtype,
                    self.rank_queues[rank], self.output_queue,
                    master_addr, master_port,
                ),
                daemon=True,
            )
            p.start()
            self.processes.append(p)

    # ------------------------------------------------------------------
    # Public interface (mirrors S2SModel)
    # ------------------------------------------------------------------

    def infer(
        self,
        audio: torch.Tensor,
        text_prompt: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
    ) -> dict:
        """Send one request to all workers and return the result from rank-0."""
        item = (audio.cpu(), text_prompt, max_new_tokens, temperature)
        with self._lock:
            for q in self.rank_queues:
                q.put(item)
            return self.output_queue.get()

    def generate_stream(
        self,
        audio_frames: Iterator[torch.Tensor],
        text_prompt: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
    ) -> Iterator[dict]:
        """Drop-in replacement for S2SModel.generate_stream()."""
        frames = list(audio_frames)
        if not frames:
            return
        audio = torch.cat(frames, dim=-1)
        yield self.infer(audio, text_prompt=text_prompt,
                         max_new_tokens=max_new_tokens,
                         temperature=temperature)

    def forward(self, batch: dict) -> dict:
        raise NotImplementedError("TensorParallelPool is inference-only.")

    def shutdown(self) -> None:
        """Send shutdown sentinel to all workers and wait for them to exit."""
        for q in self.rank_queues:
            q.put(None)
        for p in self.processes:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
