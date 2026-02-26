"""
s2s_run.py — S2S Framework CLI

Subcommands:
  train    Train an S2S model (omni2 or moshi)
  convert  One-shot weight conversion from upstream checkpoints
  serve    Start the FastAPI + WebSocket server
  webui    Launch the Gradio web UI
  eval     Run dual-agent dialogue evaluation
  infer    Run single-file streaming inference

Usage:
  uv run python3 s2s_run.py train   --model omni2 --weights ./weights/omni2 --data ./data/train
  uv run python3 s2s_run.py convert omni2 --src /path/to/omni2 --dst ./weights/omni2
  uv run python3 s2s_run.py serve   --model omni2 --weights ./weights/omni2 --port 8998
  uv run python3 s2s_run.py webui   --server-url ws://localhost:8998
  uv run python3 s2s_run.py eval    --model-a omni2 --weights-a ./weights/omni2 \
                                    --model-b moshi  --weights-b ./weights/moshi \
                                    --goal "reach agreement" --max-turns 10
  uv run python3 s2s_run.py infer   --model omni2 --weights ./weights/omni2 --audio input.wav
"""

import argparse
import os
import sys
import types


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _s2s_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _add_common_model_args(parser: argparse.ArgumentParser) -> None:
    """Args shared by commands that load a single S2S model."""
    parser.add_argument(
        "--model", type=str, default="omni2", choices=["omni2", "moshi"],
        help="Model architecture (default: omni2)",
    )
    parser.add_argument(
        "--weights", type=str, required=True,
        help="Path to weights directory produced by 'convert'",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Torch device, e.g. cpu / cuda / cuda:0 (default: cpu)",
    )
    parser.add_argument(
        "--dtype", type=str, default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype (default: float32)",
    )


def _load_model(args):
    """Instantiate and return an S2SModel from CLI args."""
    import torch
    from s2s.lm.omni2 import Omni2Model
    from s2s.lm.moshi import MoshiModel

    device = args.device
    dtype = getattr(torch, args.dtype)
    config: dict = {}  # populated via weights_dir/config.json if present

    # Try loading config from weights dir
    cfg_path = os.path.join(args.weights, "config.json")
    if os.path.isfile(cfg_path):
        import json
        with open(cfg_path) as f:
            config = json.load(f)

    if args.model == "omni2":
        model = Omni2Model.from_safetensors(args.weights, config, device)
    else:
        model = MoshiModel.from_safetensors(args.weights, config, device)

    model.to(dtype)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Sub-command: train
# ---------------------------------------------------------------------------

def _add_train_args(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("train", help="Train an S2S model")
    _add_common_model_args(p)

    # Data
    p.add_argument("--data", type=str, required=True,
                   help="Path to training data directory")
    p.add_argument("--val-data", type=str, default=None,
                   help="Path to validation data (optional)")

    # Training hyper-parameters
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--optimizer", type=str, default="adamw",
                   choices=["adamw", "adam", "sgd", "rmsprop"])
    p.add_argument("--lr-scheduler", type=str, default="cosine",
                   choices=["cosine", "constant", "step", "exponential"])
    p.add_argument("--grad-clip", type=float, default=1.0)

    # Distributed
    p.add_argument("--world-size", type=int, default=1,
                   help="Number of GPUs / processes")
    p.add_argument("--dist-backend", type=str, default="nccl")
    p.add_argument("--dist-master-addr", type=str, default="127.0.0.1")
    p.add_argument("--dist-master-port", type=str, default="29500")

    # Tensor-parallel
    p.add_argument("--tp-degree", type=int, default=1,
                   help="Tensor-parallel degree (1 = disabled)")

    # Misc
    p.add_argument("--num-workers", type=int, default=4,
                   help="Prefetcher worker threads")
    p.add_argument("--save-dir", type=str, default="./checkpoints/s2s",
                   help="Directory for checkpoints and logs")
    p.add_argument("--save-every", type=int, default=1,
                   help="Save checkpoint every N epochs")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=50,
                   help="Log loss every N steps")
    p.set_defaults(func=cmd_train)


def cmd_train(args) -> None:
    from s2s.train.trainer import S2STrainer
    import torch.multiprocessing as mp

    # Build a namespace compatible with _MetaTrainer
    train_args = types.SimpleNamespace(
        # _MetaTrainer required fields
        save_dir=args.save_dir,
        world_size=args.world_size,
        global_rank=0,
        dist_backend=args.dist_backend,
        dist_url="tcp://",
        dist_master_addr=args.dist_master_addr,
        dist_master_port=args.dist_master_port,
        device=args.device,
        dtype=args.dtype,
        random_seed=args.random_seed,

        # S2STrainer-specific
        model=args.model,
        weights=args.weights,
        data=args.data,
        val_data=args.val_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        optimizer=args.optimizer,
        lr_scheduler=args.lr_scheduler,
        grad_clip=args.grad_clip,
        tp_degree=args.tp_degree,
        num_workers=args.num_workers,
        save_every=args.save_every,
        resume=args.resume,
        log_every=args.log_every,
    )

    trainer = S2STrainer(train_args)

    if args.world_size > 1:
        mp.spawn(trainer.worker, nprocs=args.world_size, join=True)
    else:
        trainer.worker(0)


# ---------------------------------------------------------------------------
# Sub-command: convert
# ---------------------------------------------------------------------------

def _add_convert_args(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("convert", help="Convert upstream checkpoints to canonical safetensors")
    cs = p.add_subparsers(dest="checkpoint", required=True)

    for name, help_text in [
        ("mimi",  "Convert Moshi Mimi codec weights"),
        ("omni2", "Convert LLaMA-Omni2 weights (Whisper + projector + Qwen2 + generator)"),
        ("moshi", "Convert Moshi joint LM weights"),
    ]:
        cp = cs.add_parser(name, help=help_text)
        cp.add_argument("--src", type=str, required=True,
                        help="Source checkpoint directory or file")
        cp.add_argument("--dst", type=str, required=True,
                        help="Destination directory for canonical safetensors")

    p.set_defaults(func=cmd_convert)


def cmd_convert(args) -> None:
    from s2s.tools.convert import convert_mimi, convert_omni2, convert_moshi_lm

    os.makedirs(args.dst, exist_ok=True)
    dispatch = {"mimi": convert_mimi, "omni2": convert_omni2, "moshi": convert_moshi_lm}
    fn = dispatch[args.checkpoint]
    print(f"[convert] {args.checkpoint}  {args.src}  →  {args.dst}")
    fn(args.src, args.dst)
    print(f"[convert] Done. Weights saved to {args.dst}")


# ---------------------------------------------------------------------------
# Sub-command: serve
# ---------------------------------------------------------------------------

def _add_serve_args(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("serve", help="Start the FastAPI + WebSocket server")
    _add_common_model_args(p)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8998)
    p.add_argument("--reload", action="store_true",
                   help="Enable uvicorn auto-reload (dev mode)")
    p.add_argument("--workers", type=int, default=1,
                   help="Number of uvicorn worker processes")
    p.set_defaults(func=cmd_serve)


def cmd_serve(args) -> None:
    # Pre-load the model so the server can use it
    model = _load_model(args)

    from s2s.serve.server import set_model, app
    set_model(model)

    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
    )


# ---------------------------------------------------------------------------
# Sub-command: webui
# ---------------------------------------------------------------------------

def _add_webui_args(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "webui",
        help="Load model, start server, and open the browser UI (all-in-one)",
    )
    _add_common_model_args(p)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8998)
    p.add_argument("--no-browser", action="store_true",
                   help="Do not automatically open a browser tab")
    p.set_defaults(func=cmd_webui)


def cmd_webui(args) -> None:
    import threading
    import webbrowser
    import uvicorn

    model = _load_model(args)
    from s2s.serve.server import set_model, app
    set_model(model)

    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    print(f"[webui] Serving at {url}  (Ctrl+C to stop)")
    uvicorn.run(app, host=args.host, port=args.port)


# ---------------------------------------------------------------------------
# Sub-command: eval (keyword Q&A)
# ---------------------------------------------------------------------------

def _add_eval_args(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "eval",
        help="Keyword Q&A eval: one agent describes a word, the other guesses it",
    )
    # Both agents use the same model
    _add_common_model_args(p)
    p.add_argument("--tp-degree", type=int, default=1,
                   help="Tensor-parallel degree for loading (default: 1)")

    # Game config
    p.add_argument("--keyword", type=str, required=True,
                   help="The word the describer must describe (without saying it)")
    p.add_argument("--max-turns", type=int, default=20,
                   help="Max turns combined (default: 20)")
    p.add_argument("--describer-prompt", type=str, default="",
                   help="Override default describer system prompt")
    p.add_argument("--guesser-prompt", type=str, default="",
                   help="Override default guesser system prompt")

    # Output
    p.add_argument("--output", type=str, default="eval_result.json",
                   help="Path to write KeywordQAResult JSON (default: eval_result.json)")
    p.add_argument("--audio-dir", type=str, default="./eval_audio",
                   help="Directory to save per-turn audio files")
    p.set_defaults(func=cmd_eval)


def cmd_eval(args) -> None:
    import json
    import dataclasses

    from s2s.pipeline.eval_dialogue import KeywordQAEvaluator, KeywordQAGoal

    print(f"[eval] Loading model {args.model} from {args.weights}")
    model = _load_model(args)

    goal = KeywordQAGoal(
        keyword=args.keyword,
        max_turns=args.max_turns,
        describer_prompt=args.describer_prompt,
        guesser_prompt=args.guesser_prompt,
    )

    os.makedirs(args.audio_dir, exist_ok=True)
    evaluator = KeywordQAEvaluator(model, goal, audio_dir=args.audio_dir, device=args.device)
    result = evaluator.run()

    status = f"guessed at turn {result.guessed_at_turn}" if result.guessed else "not guessed"
    print(f"\n[eval] keyword='{result.keyword}'  {result.turns} turns  {status}")

    with open(args.output, "w") as f:
        json.dump(dataclasses.asdict(result), f, indent=2, default=str)
    print(f"[eval] Result saved to {args.output}")


# ---------------------------------------------------------------------------
# Sub-command: infer
# ---------------------------------------------------------------------------

def _add_infer_args(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("infer", help="Run single-file streaming inference")
    _add_common_model_args(p)
    p.add_argument("--audio", type=str, required=True,
                   help="Input audio file (any format supported by ffmpeg)")
    p.add_argument("--output", type=str, default="output.wav",
                   help="Output audio file (default: output.wav)")
    p.add_argument("--chunk-ms", type=int, default=80,
                   help="Chunk size in ms fed to the streaming pipeline (default: 80)")
    p.add_argument("--text-only", action="store_true",
                   help="Print transcription/response text only, no audio output")
    p.set_defaults(func=cmd_infer)


def cmd_infer(args) -> None:
    import queue
    import torch
    from s2s.utils.av import load_audio, save_audio
    from s2s.pipeline.inference import StreamingInferencePipeline

    print(f"[infer] Loading model {args.model} from {args.weights}")
    model = _load_model(args)

    print(f"[infer] Loading audio: {args.audio}")
    waveform = load_audio(args.audio, sr=24000)   # [1, T]

    audio_q: queue.Queue = queue.Queue()
    output_q: queue.Queue = queue.Queue()

    pipeline = StreamingInferencePipeline(model, device=args.device, chunk_ms=args.chunk_ms)

    # Feed waveform in chunks
    chunk_samples = int(24000 * args.chunk_ms / 1000)
    for start in range(0, waveform.shape[-1], chunk_samples):
        chunk = waveform[..., start:start + chunk_samples]
        audio_q.put(chunk)
    audio_q.put(None)  # sentinel

    print("[infer] Running inference …")
    pipeline.run(audio_q, output_q)

    # Collect output
    text_parts = []
    audio_parts = []
    while not output_q.empty():
        item = output_q.get_nowait()
        if item is None:
            break
        if "text" in item and item["text"]:
            text_parts.append(item["text"])
        if "audio" in item and item["audio"] is not None:
            audio_parts.append(item["audio"])

    full_text = "".join(text_parts)
    print(f"\n[infer] Response text:\n{full_text}\n")

    if not args.text_only and audio_parts:
        audio_out = torch.cat(audio_parts, dim=-1)
        save_audio(audio_out, args.output, sr=24000)
        print(f"[infer] Audio saved to {args.output}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="s2s_run",
        description="S2S Framework — speech-to-speech inference, training, and evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    _add_train_args(sub)
    _add_convert_args(sub)
    _add_serve_args(sub)
    _add_webui_args(sub)
    _add_eval_args(sub)
    _add_infer_args(sub)

    return parser


def main() -> None:
    # Ensure AGI-BC root is on sys.path so `s2s` package resolves
    root = _s2s_root()
    if root not in sys.path:
        sys.path.insert(0, root)

    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
