import inspect
import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from time import time
from typing import Optional

import torch
import torch.nn.functional as F
from hydra.utils import get_class, instantiate
from loguru import logger
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from typing_extensions import Literal

from data.data import DatasetConfig
from models.gpt2 import GPT2Config


@dataclass
class LRSchedulerConfig:
    max_lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 1000
    max_steps: int = 100000


@dataclass
class TrainerConfig:
    seed: int = 42
    max_steps: int = 10
    optimizer: dict = field(default_factory=dict)
    lr_scheduler: LRSchedulerConfig = field(default_factory=lambda: LRSchedulerConfig())
    val_steps: int = 1000
    checkpoint_dir: str = "checkpoints"
    matmul_precision: Literal["highest", "high", "medium"] = "high"
    batch_size: int = 64
    seq_length: int = 1024
    grad_accumulation_steps: int = 1
    train_dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig())
    val_dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig())
    model: dict = field(
        default_factory=lambda: {
            "_target_": "models.gpt2.GPT2",
            "config": GPT2Config(),
        }
    )
    use_model_compile: bool = True
    weight_decay: float = 0.1


class DDPTrainer:
    def __init__(
        self,
        config=None,
        device: Optional[Literal["cpu", "mps", "cuda"]] = None,
        resume: Optional[str] = None,
    ):
        self.config = config or TrainerConfig()
        if device is None:
            self.device = self.device_type = "cpu"
            if torch.cuda.is_available():
                self.device = self.device_type = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = self.device_type = "mps"
        else:
            self.device = self.device_type = device
        assert (
            "cuda" not in self.device_type or torch.cuda.is_available()
        ), "CUDA is not available, but the device is set to CUDA. Please check your setup."
        assert self.device_type != "mps" or (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ), "MPS is not available, but the device is set to MPS. Please check your setup."
        assert self.device_type in [
            "cpu",
            "mps",
            "cuda",
        ], f"Invalid device: {self.device_type}. Supported devices are 'cpu', 'mps', and 'cuda'."
        if resume and os.path.exists(resume):
            self._load_checkpoint(resume)
        else:
            self._initialize()

    def _initialize(self, start_step=0):
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_master = self.rank == 0
        self.distributed = self.world_size > 1 and self.device_type == "cuda"
        if self.device_type == "cuda":
            self.device = f"cuda:{self.local_rank}"
            torch.cuda.set_device(torch.device(self.device))

        self.ctx = (
            torch.autocast(device_type=self.device_type, dtype=torch.bfloat16)
            if self.device_type == "cuda"
            else nullcontext()
        )

        torch.manual_seed(self.config.seed)
        if self.device_type == "cuda":
            torch.cuda.manual_seed(self.config.seed)

        # Set matmul precision
        torch.set_float32_matmul_precision(self.config.matmul_precision)

        self.model: torch.nn.Module = self._prepare_model()
        self.optimizer = self._prepare_optimizer()
        self.train_dataset, self.val_dataset = self._prepare_datasets()
        self.start_step = start_step
        if self.is_master:
            logger.info("Training configuration:")
            logger.info(f"\tMax steps: {self.config.max_steps}")
            logger.info(f"\tBatch size: {self.config.batch_size}")
            logger.info(f"\tSequence length: {self.config.seq_length}")
            logger.info(
                f"\tGrad accumulation steps: {self.config.grad_accumulation_steps}"
            )
            logger.info(f"\tDevice: {self.device}")
            logger.info(f"\tDistributed: {self.distributed}")
            logger.info(f"\tWorld size: {self.world_size}")

    @logger.catch
    def train(self):
        if self.distributed:
            init_process_group()
        logger.info("Starting training...")
        for step in range(self.start_step, self.config.max_steps):
            is_last_step = step == self.config.max_steps - 1
            self._train_step(self.train_dataset, step)
            if step % self.config.val_steps == 0 or is_last_step:
                self._val_step(self.val_dataset, step)
        logger.info("Training completed.")
        if self.distributed:
            destroy_process_group()

    def _train_step(self, train_dataset, step):
        t0 = time()
        self.model.train()
        self.optimizer.zero_grad()
        accumulated_loss = 0.0  # for logging
        # gradient accumulation
        for _ in range(self.config.grad_accumulation_steps):
            x, y = next(train_dataset)
            metrics = self._calculate_loss(x, y)
            loss = metrics["loss"] / self.config.grad_accumulation_steps
            accumulated_loss += loss.detach()
            if self.distributed:
                # Use no_sync to avoid synchronizing gradients during accumulation
                with self.raw_model.no_sync():
                    loss.backward()
            else:
                loss.backward()
        if self.distributed:
            torch.distributed.all_reduce(
                accumulated_loss, op=torch.distributed.ReduceOp.AVG
            )
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        lr = self._schedule_lr(step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        self.optimizer.step()

        if self.device_type == "cuda":
            torch.cuda.synchronize()
        t1 = time()
        elapsed = (t1 - t0) * 1000  # convert to milliseconds
        tokens_per_sec = (
            self.config.batch_size
            * self.config.grad_accumulation_steps
            * self.world_size
        ) / (t1 - t0)
        if self.is_master:
            logger.info(
                f"Step {step}: loss={accumulated_loss:.4f} | norm: {norm:.2f} | dt={elapsed:.2f}ms | tok/sec={tokens_per_sec:.2f}"
            )

    def _val_step(self, val_dataset, step):
        self.model.eval()
        val_dataset.reset()
        loss = 0
        step_count = 0
        all_neg_log_probs = 0.0
        tokens_count = 0
        with torch.no_grad():
            for x, y in val_dataset:
                metrics = self._calculate_loss(x, y)
                loss += metrics["loss"]
                step_count += 1
                all_neg_log_probs += self._calculate_neg_sum_log_probs(
                    metrics["logits"], y
                )
                tokens_count += y.numel()
            avg_loss = (loss / step_count) if step_count > 0 else 0.0
            perplexity = (
                torch.exp(all_neg_log_probs / tokens_count)
                if tokens_count > 0
                else float("inf")
            )
            if self.distributed:
                torch.distributed.all_reduce(
                    avg_loss, op=torch.distributed.ReduceOp.AVG
                )
        if self.is_master:
            logger.info(
                f"Validation step {step}: avg_loss={avg_loss:.4f}, perplexity={perplexity:.4f}"
            )
            if step != 0:
                self._checkpoint(step, avg_loss)

    def _checkpoint(self, step, avg_loss):
        assert self.is_master, "Checkpointing can only be done by the master process."
        checkpoint_dir = self.config.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step:05d}.pt")
        state = {
            "step": step,
            "avg_loss": avg_loss,
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
        }
        torch.save(state, checkpoint_path)

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = checkpoint["config"]
        self._initialize(checkpoint["step"])
        self.raw_model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def _calculate_loss(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        with self.ctx:
            loss, logits = self.model(x, y)
        return {
            "loss": loss,
            "logits": logits,
        }

    def _calculate_neg_sum_log_probs(self, logits, y):
        probs = F.softmax(logits.view(-1, logits.size(-1)), dim=1)
        probs = probs[torch.arange(y.numel()), y.view(-1)]
        neg_log_probs = -probs.log().sum()
        return neg_log_probs

    def _prepare_model(self):
        self.raw_model: torch.nn.Module = instantiate(self.config.model)
        self.raw_model.to(self.device)
        if self.config.use_model_compile and self.device_type != "mps":
            self.raw_model = torch.compile(self.raw_model)
        if self.distributed:
            return DDP(self.raw_model, output_device=self.local_rank)
        return self.raw_model

    def _prepare_datasets(self):
        train_dataset = instantiate(
            self.config.train_dataset,
            self.config.batch_size,
            self.config.seq_length,
            world_size=self.world_size,
            rank=self.rank,
            split="train",
        )
        val_dataset = instantiate(
            self.config.val_dataset,
            self.config.batch_size,
            self.config.seq_length,
            world_size=self.world_size,
            rank=self.rank,
            split="val",
        )
        return train_dataset, val_dataset

    def _prepare_optimizer(self):
        assert (
            "_target_" in self.config.optimizer
        ), "Optimizer config must have '_target_' key"
        learnable_params = [
            param
            for _, param in self.raw_model.named_parameters()
            if param.requires_grad
        ]
        decay_params = list(filter(lambda p: p.dim() >= 2, learnable_params))
        no_decay_params = list(filter(lambda p: p.dim() < 2, learnable_params))
        model_params = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        opt_class = get_class(self.config.optimizer["_target_"])
        fused_available = "fused" in inspect.signature(opt_class).parameters
        if fused_available and self.device_type == "cuda":
            optimizer = instantiate(self.config.optimizer, model_params, fused=True)
        else:
            optimizer = instantiate(self.config.optimizer, model_params)
        return optimizer

    def _schedule_lr(self, step):
        # Linear warmup
        if step < self.config.lr_scheduler.warmup_steps:
            # start from max_lr/warmup_steps and goes to max_lr
            return (
                self.config.lr_scheduler.max_lr
                * (step + 1)
                / self.config.lr_scheduler.warmup_steps
            )
        # Constant learning rate after max_steps
        if step > self.config.lr_scheduler.max_steps:
            return self.config.lr_scheduler.min_lr
        # Cosine decay
        decay_ratio = (step - self.config.lr_scheduler.warmup_steps) / (
            self.config.lr_scheduler.max_steps - self.config.lr_scheduler.warmup_steps
        )
        assert 0 <= decay_ratio <= 1, "Decay ratio must be in [0, 1]"
        # coeff starts from 1 and goes to 0
        coeff = 0.5 * (1 + torch.cos(torch.pi * decay_ratio))
        return self.config.lr_scheduler.min_lr + coeff * (
            self.config.lr_scheduler.max_lr - self.config.lr_scheduler.min_lr
        )

    def _prepare_batch(self, batch):
        return batch
