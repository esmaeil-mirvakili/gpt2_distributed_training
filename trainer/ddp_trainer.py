import inspect
import os
import re
from contextlib import nullcontext
from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from hydra.utils import get_class, instantiate
from loguru import logger
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from typing_extensions import Literal

from trainer.base_trainer import BaseTrainerConfig, BaseTrainer
from trainer.utils import CheckpointStrategy


@dataclass
class LRSchedulerConfig:
    max_lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 1000
    max_steps: int = 100000


@dataclass
class CheckpointStrategyConfig:
    _target_: str = "trainer.ddp_trainer.DDPCheckpointStrategy"
    config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DDPTrainerConfig(BaseTrainerConfig):
    seed: int = 42
    max_steps: int = 10
    optimizer: Dict[str, Any] = field(default_factory=dict)
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    val_steps: int = 1000
    checkpoint_strategy: CheckpointStrategyConfig = field(
        default_factory=CheckpointStrategyConfig
    )
    matmul_precision: str = "high"
    grad_accumulation_steps: int = 1
    use_model_compile: bool = True
    weight_decay: float = 0.1

    def __post_init__(self):
        allowed_precisions = {"highest", "high", "medium"}
        if self.matmul_precision not in allowed_precisions:
            raise ValueError(
                f"Invalid matmul_precision '{self.matmul_precision}'. Allowed values are {allowed_precisions}."
            )


class DDPCheckpointStrategy(CheckpointStrategy):
    def save_checkpoint(self, model, optimizer, loss, step, is_master):
        if not is_master:
            return

        checkpoint_dir = self.config.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step:05d}.pt")
        state = {
            "step": step,
            "avg_loss": loss,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(state, checkpoint_path)

    def load_checkpoint(self, model, optimizer, device, is_master):
        checkpoint_pattern = r"checkpoint_(\d+)\.pt"
        latest_checkpoint = None
        latest_step = -1
        checkpoint_dir = self.config.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        for filename in os.listdir(checkpoint_dir):
            match = re.match(checkpoint_pattern, filename)
            if match:
                step = int(match.group(1))
                if step > latest_step:
                    latest_step = step
                    latest_checkpoint = os.path.join(checkpoint_dir, filename)
        if latest_checkpoint is None:
            if is_master:
                logger.warning("No checkpoint found.")
            return 0
        if not is_master:
            logger.info(f"Loading checkpoint from {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if is_master:
            logger.info(
                f"Loaded checkpoint step {checkpoint['step']} with avg_loss {checkpoint['avg_loss']}"
            )
        return checkpoint["step"] + 1


class DDPTrainer(BaseTrainer):
    def __init__(
        self,
        device: Optional[Literal["cpu", "mps", "cuda"]] = None,
        resume: Optional[bool] = False,
        config: Optional[DDPTrainerConfig] = None,
    ):
        super().__init__(resume=resume)
        self.config = config or DDPTrainerConfig()
        self.raw_model: torch.nn.Module = None
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

    def _initialize(self, model: torch.nn.Module, train_dataset, val_dataset):
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
        
        if self.distributed:
            init_process_group(backend="nccl")

        self.checkpoint_strategy = instantiate(self.config.checkpoint_strategy)
        self.model = self._prepare_model(model)
        self.optimizer = self._prepare_optimizer()
        self.train_dataset, self.val_dataset = self._prepare_datasets(train_dataset, val_dataset)
        if self.is_master:
            logger.info("Training configuration:")
            logger.info(f"\tMax steps: {self.config.max_steps}")
            logger.info(
                f"\tGrad accumulation steps: {self.config.grad_accumulation_steps}"
            )
            logger.info(f"\tDevice: {self.device}")
            logger.info(f"\tDistributed: {self.distributed}")
            logger.info(f"\tWorld size: {self.world_size}")

    @logger.catch
    def train(self, model: torch.nn.Module, train_dataset, val_dataset):
        self._initialize(model, train_dataset, val_dataset)
        if self.is_master:
            logger.info("Training Initialization complete.")
        self.start_step = 0
        if self.resume:
            if self.is_master:
                logger.info("Resuming training...")
            self.start_step = self.checkpoint_strategy.load_checkpoint(
                self.raw_model, self.optimizer, self.device, self.is_master
            )
        if self.is_master:
            logger.info("Starting training...")
        for step in range(self.start_step, self.config.max_steps):
            is_last_step = step == self.config.max_steps - 1
            self._train_step(self.train_dataset, step)
            if step % self.config.val_steps == 0 or is_last_step:
                self._val_step(self.val_dataset, step)
        if self.is_master:
            logger.info("Training completed.")
        if self.distributed:
            destroy_process_group()
    
    @staticmethod
    def _to_scalar(x):
        try:
            from torch.distributed._tensor import DTensor  # PyTorch 2.7+
            if isinstance(x, DTensor):
                return x.to_local().detach().float().item()
        except Exception:
            pass
        if isinstance(x, torch.Tensor):
            return x.detach().float().item()
        return float(x)

    def _train_step(self, train_dataset, step):
        t0 = time()
        self.model.train()
        self.optimizer.zero_grad()
        accumulated_loss = 0.0  # for logging
        # gradient accumulation
        for micro_step in range(self.config.grad_accumulation_steps):
            x, y = next(train_dataset)
            metrics = self._calculate_loss(x, y)
            loss = metrics["loss"] / self.config.grad_accumulation_steps
            accumulated_loss += loss.detach()
            need_sync = (micro_step == self.config.grad_accumulation_steps - 1)
            if hasattr(self.model, "require_backward_grad_sync"):
                self.model.require_backward_grad_sync = need_sync
            loss.backward()
        if hasattr(self.model, "require_backward_grad_sync"):
            self.model.require_backward_grad_sync = True
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
            self.train_dataset.batch_size
            * self.train_dataset.seq_length
            * self.config.grad_accumulation_steps
            * self.world_size
        ) / (t1 - t0)
        if self.is_master:
            logger.info(
                f"Step {step}: loss={self._to_scalar(accumulated_loss):.4f} | norm: {self._to_scalar(norm):.2f} | dt={elapsed:.2f}ms | tok/sec={tokens_per_sec:.2f}"
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
                torch.distributed.all_reduce(
                    perplexity, op=torch.distributed.ReduceOp.AVG
                )
        if self.is_master:
            logger.info(
                f"Validation step {step}: avg_loss={avg_loss:.4f}, perplexity={perplexity:.4f}"
            )
        if step != 0:
            self.checkpoint_strategy.save_checkpoint(
                self.raw_model, self.optimizer, avg_loss, step, self.is_master
            )

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

    def _prepare_model(self, model: torch.nn.Module):
        self.raw_model = model
        self.raw_model.to(self.device)
        if self.config.use_model_compile and self.device_type != "mps":
            self.raw_model = torch.compile(self.raw_model)
        if self.distributed:
            return DDP(self.raw_model, output_device=self.local_rank)
        return self.raw_model

    def _prepare_datasets(self, train_dataset, val_dataset):
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
