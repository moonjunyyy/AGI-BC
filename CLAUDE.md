# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AGI-BC is a multimodal deep learning system for **Korean conversational backchannel prediction**. It classifies listener responses into 4 types: `NoBC`, `Continuer`, `Understanding`, `Empathic Response`. Inputs combine audio (HuBert/LSTM/STFT-CNN), Korean text (koBERT), and video (VideoMAE).

## Environment Setup

- **Python**: 3.12.2 (see `.python-version`)
- **Package manager**: UV (see `uv.lock` and `pyproject.toml`)
- **Conda alternative**: `environment.yaml` (Linux-64, CUDA 12.4, PyTorch 2.5.1)
- **Dataset path**: `/local_datasets` (default `--data_path`)
- **Sentiment dictionary**: `data/SentiWord_info.json`

## Running

**Training:**
```bash
python main.py \
  --model BPM_MT \
  --mode cross_entropy \
  --dataset ETRI \
  --language koBert \
  --audio HuBert \
  --video VideoMAE \
  --batch_size 128 --epochs 100 --lr 0.0005 --dropout 0.3
```

**Multi-GPU (SLURM):** See `run.sh` — sets `MASTER_PORT`, `WORLD_SIZE`, `MASTER_ADDR` and passes `--world_size`.

**Feature extraction:** `run_features.sh`

**LLM evaluation:**
- `python gpt4.py` — GPT-4 few-shot (requires OpenAI API key)
- `python llama.py` — Llama 3.1 8B via local llama-server on port 24763

## Key CLI Arguments

| Argument | Default | Options |
|---|---|---|
| `--model` | `BPM_MT` | `BPM_ST`, `BPM_ST_Target`, `Ours`, `Ours_Video`, `LoRA_BC`, `Finetune`, `Proxy_Prototype`, `Diffused_Backchannel`, `Praveen_etal` |
| `--language` | `koBert` | `ELECTRA` |
| `--audio` | `HuBert` | `LSTM`, `STFT_CNN` |
| `--video` | `VideoMAE` | — |
| `--dataset` | `ETRI` | `SWBD` (English Switchboard) |
| `--mode` | `cross_entropy` | focal loss variants |
| `--world_size` | `1` | number of GPUs |

## Architecture

```
Audio (WAV) → HuBert/LSTM/STFT-CNN → [B, T, 768]
Text  (KO)  → koBERT              → [B, T, 768]   →  Fusion → 4-class BC
Video (MP4) → VideoMAE            → [B, T, D]
```

**Entry point:** `main.py` instantiates `Trainer(args)` and calls `trainer.run()`.

**Trainer** (`utils/trainer.py`): Orchestrates distributed training (PyTorch DDP + NCCL). Spawns processes for multi-GPU; handles model init, dataset loading (80/20 split), and optimization loop.

**Model factory / Dataset factory:** `utils/utils.py` — maps string names to classes.

**Loss functions:** `utils/criterions.py` (cross-entropy, focal variants); contrastive losses in `utils/contrastive_loss.py`.

**Attention layers** (`layer/`):
- `self_attention_layer.py` — multi-head self-attention + FFN
- `cross_attention_layer.py` — cross-modal attention (query dim ≠ KV dim)
- `lora.py` — LoRA parameter adaptation layer

**Key model files** (`model/`):
- `bpm_mt.py` — `BPM_MT` (multitask: backchannel + sentiment), `BPM_ST`, `BPM_ST_Target`
- `ours.py` — Advanced model with LoRA injection on all projection layers + 12-layer cross-attention + sentiment dict
- `ours_video.py` / `ours_video_allign.py` — video-integrated variants
- `hubert.py` — HuBert audio encoder (768-dim output)
- `audio_lstm.py` — MFCC + 4-layer biLSTM (26-dim)

**Dataset** (`dataset/`):
- `ETRI_Dataset.py` — main loader; supports full multimodal, dialog windows (1.5s context / 0.5s prediction), threshold-based windowing, balanced/unbalanced sampling, 16-frame video extraction
- `SWBD_Dataset.py` — English Switchboard baseline

**Custom utility library** (`m00nny_utils/`): DDP sharding helpers (`parallel/`), base trainer class (`util/_trainer.py`), WarmUpCosineAnnealing scheduler (`lr_scheduler/`), media I/O for audio/video.

## Submodules

`cosyvoice/`, `matcha/`, `llama_omni2/`, `moshi/`, `kobert/`, `kobert_tokenizer/` are git submodules — do not modify unless working specifically on those components.
