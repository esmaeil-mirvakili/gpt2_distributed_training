# Gpt2 Distributed Training
This repo contains pytorch code for implementing and training GPT2 model from Scratch, Distributed.

This repository implements a **from‑scratch GPT‑2** in PyTorch and wires it up for **distributed training** via **DDP**, **FSDP**, or **DeepSpeed**.

---

## Highlights

- **Model:** Minimal GPT‑2 reimplementation with `GPT2Config` (layers/heads/embedding size/seq length, optional flash‑attention toggle).
- **Distributed:** Pluggable trainers
  - **DDP** (`trainer/ddp_trainer.py`)
  - **FSDP** (`trainer/fsdp_trainer.py`) with mixed‑precision and (de)prefetch knobs
  - **DeepSpeed** (`trainer/deepspeed_trainer.py`) with an external JSON config (e.g. `deepspeed_configs/zero1.json`)
- **Hydra configs** in `configs/` with groups: `trainer/`, `model/`, `dataset/`; default composition is in `configs/config.yaml`.
- **Checkpoints & logging:** W&B integrated, checkpoint strategies per trainer, loguru for human‑readable logs.

---

## Quickstart

### 1) Environment

Install PyTorch first (choose the right CUDA build yourself):  
https://pytorch.org/get-started/locally/

Then the Python deps (note: `requirements.txt` does **not** include `torch` and repeats `wandb`; fix as needed):
```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121   # example; pick your CUDA
```

### 2) Prepare data

This project expects **token shards** on disk. Use the helper to build them:

```bash
python data/download.py --dataset wikitext --dir data/wikitext --shard_size 100000000 --num_shards 10
```

### 3) Run training

Hydra composes configs from `configs/config.yaml`. Choose a trainer by selecting the `trainer` group.

**Single machine, multi‑GPU, DDP**:
```bash
torchrun --nproc_per_node=8 main.py trainer=ddp   dataset.train_dataset.data_root=./data/wikitext   model.config.n_layer=12 model.config.n_head=12 model.config.n_embd=768   trainer.config.max_steps=10000 trainer.config.grad_accumulation_steps=8
```

**FSDP** (recommended for larger models):
```bash
torchrun --nproc_per_node=8 main.py trainer=fsdp   dataset.train_dataset.data_root=./data/wikitext   trainer.config.use_mixed_precision=true   trainer.config.checkpoint_strategy.config.checkpoint_dir=./checkpoints
```

**DeepSpeed (ZeRO‑1 by default)**:
```bash
deepspeed --num_gpus=8 main.py trainer=deepspeed   dataset.train_dataset.data_root=./data/wikitext   trainer.config.deepspeed_config=deepspeed_configs/zero1.json
```