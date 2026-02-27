"""
Tensor-parallel helpers wrapping m00nny_utils/parallel/sharded_modules.py.

Worker processes are spawned by spawn_tp_workers().  Each worker initialises
dist, loads the sharded model, then signals "ready" via startup_queue before
entering the inference loop.  spawn_tp_workers() blocks until ALL workers
have signalled ready (or raises on the first error), so the returned
TPWorkers handle is safe to use immediately.
"""
import os
import sys
import threading

import torch
import torch.nn as nn
import torch.multiprocessing as mp

_COL_PARALLEL = [r".*\.q_proj$", r".*\.k_proj$", r".*\.v_proj$",
                 r".*\.gate_proj$", r".*\.up_proj$"]
_ROW_PARALLEL = [r".*\.o_proj$", r".*\.down_proj$"]
_EMBED        = [r".*embed_tokens$"]


def shard_model(model: nn.Module, tp_degree: int = 1) -> nn.Module:
    """Shard model weights across TP ranks.  Requires dist.init_process_group()."""
    if tp_degree <= 1:
        return model
    if not torch.distributed.is_initialized():
        raise RuntimeError("dist not initialised — call shard_model inside a TP worker")
    try:
        from m00nny_utils.parallel.sharded_modules import _convert_to_sharded_module_recursive
    except ImportError as e:
        raise ImportError(
            "m00nny_utils is required for tensor parallelism. "
            "Install it or set --tp-degree 1."
        ) from e
    return _convert_to_sharded_module_recursive(
        model,
        embed_parallel_ids=_EMBED,
        col_parallel_ids=_COL_PARALLEL,
        row_parallel_ids=_ROW_PARALLEL,
    )


def _tp_worker(rank, world_size, model_name, weights_dir, config, dtype_str,
               rank_queues, output_queue, startup_queue, master_addr, master_port):
    """Worker entry — one per GPU rank.  Signals startup_queue when ready (or on error)."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    try:
        torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)

        if model_name == "moshi":
            from s2s.lm.moshi import MoshiModel
            ModelCls = MoshiModel
        else:
            from s2s.lm.omni2 import Omni2Model
            ModelCls = Omni2Model

        dtype = getattr(torch, dtype_str, torch.float32)
        model = ModelCls.from_safetensors(weights_dir, config, device=device, tp_degree=world_size)
        model.to(dtype).eval()

    except Exception as exc:
        import traceback
        startup_queue.put(("error", rank, f"{exc}\n{traceback.format_exc()}"))
        return

    # Signal ready — main process waits for this before sending any requests
    startup_queue.put(("ready", rank))

    while True:
        item = rank_queues[rank].get()
        if item is None:
            break
        audio_cpu, text_prompt, max_new_tokens, temperature = item
        result = {"text": "", "audio": None}
        try:
            with torch.no_grad():
                for r in model.generate_stream(
                    iter([audio_cpu.to(device)]),
                    text_prompt=text_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                ):
                    result = r
                    break
        except Exception as exc:
            import traceback
            print(f"[tp_worker rank={rank}] generate_stream error:\n{traceback.format_exc()}", flush=True)
            result = {"text": "", "audio": None, "error": str(exc)}
        if rank == 0:
            if result.get("audio") is not None:
                result["audio"] = result["audio"].cpu()
            output_queue.put(result)

    torch.distributed.destroy_process_group()


class TPWorkers:
    """Thin wrapper around TP worker queues.  Exposes generate_stream() like a model."""

    def __init__(self, rank_queues, output_queue):
        self._rank_queues  = rank_queues
        self._output_queue = output_queue
        self._lock         = threading.Lock()

    def generate_stream(self, audio_frames, text_prompt=None,
                        max_new_tokens=256, temperature=1.0):
        frames = list(audio_frames)
        if not frames:
            return
        audio = torch.cat(frames, dim=-1).cpu()
        with self._lock:
            for q in self._rank_queues:
                q.put((audio, text_prompt, max_new_tokens, temperature))
            yield self._output_queue.get()

    def shutdown(self):
        for q in self._rank_queues:
            q.put(None)


def spawn_tp_workers(model_name: str, weights_dir: str, config: dict, tp_degree: int,
                     dtype: str = "bfloat16",
                     master_addr: str = "127.0.0.1",
                     master_port: str = "29500") -> TPWorkers:
    """Spawn N worker processes and wait until all are ready.  Returns a TPWorkers handle."""
    ctx           = mp.get_context("spawn")
    rank_queues   = [ctx.Queue() for _ in range(tp_degree)]
    output_queue  = ctx.Queue()
    startup_queue = ctx.Queue()

    for rank in range(tp_degree):
        p = ctx.Process(
            target=_tp_worker,
            args=(rank, tp_degree, model_name, weights_dir, config, dtype,
                  rank_queues, output_queue, startup_queue, master_addr, master_port),
            daemon=True,
        )
        p.start()

    # Block until every worker signals ready or one signals error
    errors = []
    for _ in range(tp_degree):
        msg = startup_queue.get()   # blocks
        if msg[0] == "error":
            errors.append(f"rank={msg[1]}: {msg[2]}")

    if errors:
        raise RuntimeError("TP worker(s) failed to initialise:\n" + "\n".join(errors))

    return TPWorkers(rank_queues, output_queue)
