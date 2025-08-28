from trainer.base_trainer import BaseTrainer, BaseTrainerConfig
import os
import torch
import deepspeed
from time import time
from loguru import logger
from hydra.utils import get_class, instantiate
import inspect
from trainer.utils import CheckpointStrategy
import math
from torch.nn import functional as F
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class LRSchedulerConfig:
    max_lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 1000
    max_steps: int = 100000


@dataclass
class CheckpointStrategyConfig:
    _target_: str = "trainer.deepspeed_trainer.DeepSpeedCheckpointStrategy"
    config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DeepSpeedTrainerConfig(BaseTrainerConfig):
    seed: int = 42
    max_steps: int = 10
    optimizer: Dict[str, Any] = field(default_factory=dict)
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    val_steps: int = 1000
    checkpoint_strategy: CheckpointStrategyConfig = field(
        default_factory=CheckpointStrategyConfig
    )
    matmul_precision: str = "high"
    use_model_compile: bool = True
    weight_decay: float = 0.1
    deepspeed_config: str = "deepspeed_configs/zero1.json"

    def __post_init__(self):
        allowed_precisions = {"highest", "high", "medium"}
        if self.matmul_precision not in allowed_precisions:
            raise ValueError(
                f"Invalid matmul_precision '{self.matmul_precision}'. Allowed values are {allowed_precisions}."
            )

class DeepSpeedCheckpointStrategy(CheckpointStrategy):
    def save_checkpoint(self, model, optimizer, loss, step, is_master):
        checkpoint_dir = self.config.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        state = {
            "step": step,
            "avg_loss": loss
        }
        deepspeed.comm.barrier()
        model.save_checkpoint(checkpoint_dir, tag=f"checkpoint_{step:07d}", client_state=state)
        deepspeed.comm.barrier()


    def load_checkpoint(self, model, optimizer, device, is_master):
        checkpoint_dir = self.config.checkpoint_dir
        checkpoint = model.load_checkpoint(
            checkpoint_dir,
            tag=None,  # automatically loads the latest checkpoint
            load_optimizer_states=True,
            load_lr_scheduler_states=True
        )
        if checkpoint is None:
            if is_master:
                logger.warning("No checkpoint found.")
            return 0
        load_path, client_state = checkpoint
        if is_master:
            logger.info(
                f"Loaded checkpoint step {client_state.get('step', 0)} with avg_loss {client_state.get('avg_loss', 0)} from {load_path}."
            )
        return int(client_state.get('step', 0)) + 1


class DeepSpeedTrainer(BaseTrainer):
    def __init__(self, device=None, resume=False, config=None):
        super().__init__(device, resume, config)
        self.config = config or DeepSpeedTrainerConfig()
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

        torch.manual_seed(self.config.seed)
        if self.device_type == "cuda":
            torch.cuda.manual_seed(self.config.seed)

        # Set matmul precision
        torch.set_float32_matmul_precision(self.config.matmul_precision)

        self.checkpoint_strategy = instantiate(self.config.checkpoint_strategy)
        model = self._prepare_model(model)
        optimizer = self._prepare_optimizer()
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self._schedule_lr)
        model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            model_parameters=(p for p in model.parameters() if p.requires_grad),
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=self.config.deepspeed_config
        )
        self.model = model_engine
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.train_dataset, self.val_dataset = self._prepare_datasets(train_dataset, val_dataset)
        if self.is_master:
            logger.info("Training configuration:")
            logger.info(f"Max steps: {self.config.max_steps}")
            logger.info(f"DeepSpeed Config: {self.config.deepspeed_config}")
            logger.info(f"Device: {self.device}")
            logger.info(f"Distributed: {self.distributed}")
            logger.info(f"World size: {self.world_size}")
    
    def _destroy(self):
        pass

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
                self.model, self.optimizer, self.device, self.is_master
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

    def _train_step(self, train_dataset, step):
        self.model.train()
        if not hasattr(self, "accumulated_loss"):
            self.accumulated_loss = 0.0
            self.micro_updates = 0
            self.t0 = time()
        x, y = next(train_dataset)
        metrics = self._calculate_loss(x, y)
        loss = metrics["loss"]
        self.accumulated_loss += float(loss.detach())
        self.micro_updates += 1
        self.model.backward(loss)
        is_boundary = self.model.is_gradient_accumulation_boundary()
        if is_boundary and self.device_type == "cuda":
            torch.cuda.synchronize()
        self.model.step()
        if is_boundary:
            # average the accumulated losses over the micro-batches
            avg_loss = self.accumulated_loss / max(1, self.micro_updates)
            t = torch.tensor([avg_loss],
                            device=self.device)
            deepspeed.comm.all_reduce(t, op=deepspeed.comm.ReduceOp.AVG)
            avg_loss = t.item()
            
            try:
                grad_norm = self.model.get_global_grad_norm()
                grad_norm = float(grad_norm if not isinstance(grad_norm, torch.Tensor) else grad_norm.item())
            except Exception:
                # Fallback: approximate by reducing sum of squares
                sq = torch.zeros(1, device=self.device)
                for p in self.model.module.parameters():
                    if p.grad is not None:
                        sq += p.grad.detach().float().pow(2).sum()
                deepspeed.comm.all_reduce(sq, op=deepspeed.comm.ReduceOp.SUM)
                grad_norm = float(sq.sqrt().item())

            t1 = time()
            if self.is_master:
                elapsed = (t1 - self.t0) * 1000  # convert to milliseconds
                tokens_per_sec = (
                    self.train_dataset.batch_size
                    * self.train_dataset.seq_length
                    * self.model.gradient_accumulation_steps()
                    * self.world_size
                ) / (t1 - self.t0)
                logger.info(
                    f"Step {step}: loss={avg_loss:.4f} | norm: {grad_norm:.2f} | dt={elapsed:.2f}ms | tok/sec={tokens_per_sec:.2f}"
                )
            self.accumulated_loss = 0
            self.micro_updates = 0
            self.t0 = time()

    def _val_step(self, val_dataset, step):
        self.model.eval()
        val_dataset.reset()
        loss = torch.zeros(1, device=self.model.device)
        step_count = torch.zeros(1, device=self.model.device)
        all_neg_log_probs = torch.zeros(1, device=self.model.device)
        tokens_count = torch.zeros(1, device=self.model.device)
        with torch.no_grad():
            for x, y in val_dataset:
                metrics = self._calculate_loss(x, y)
                loss += metrics["loss"]
                step_count += 1
                all_neg_log_probs += self._calculate_neg_sum_log_probs(
                    metrics["logits"], y
                )
                tokens_count += y.numel()
            if self.distributed:
                deepspeed.comm.all_reduce(
                    loss, op=deepspeed.comm.ReduceOp.SUM
                )
                deepspeed.comm.all_reduce(
                    step_count, op=deepspeed.comm.ReduceOp.SUM
                )
                deepspeed.comm.all_reduce(
                    all_neg_log_probs, op=deepspeed.comm.ReduceOp.SUM
                )
                deepspeed.comm.all_reduce(
                    tokens_count, op=deepspeed.comm.ReduceOp.SUM
                )
            avg_loss = (loss / torch.clamp_min(step_count, 1)).item()
            perplexity = (
                torch.exp(all_neg_log_probs / tokens_count).item()
                if tokens_count.item() > 0
                else float("inf")
            )
        if self.is_master:
            logger.info(
                f"Validation step {step}: avg_loss={avg_loss:.4f}, perplexity={perplexity:.4f}"
            )
        if step != 0:
            self.checkpoint_strategy.save_checkpoint(
                self.model, self.optimizer, avg_loss, step, self.is_master
            )
    
    def _calculate_loss(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        loss, logits = self.model(x, y)
        return {
            "loss": loss,
            "logits": logits,
        }
    
    def _calculate_neg_sum_log_probs(self, logits, y):
        probs = F.softmax(logits.view(-1, logits.size(-1)), dim=1)
        probs = probs[torch.arange(y.numel(), device=logits.device), y.view(-1)]
        neg_log_probs = -probs.log().sum()
        return neg_log_probs

    def _prepare_model(self, model: torch.nn.Module):
        self.raw_model = model
        self.raw_model.to(self.device)
        if self.config.use_model_compile and self.device_type != "mps":
            self.raw_model = torch.compile(self.raw_model)
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
            optimizer = instantiate(self.config.optimizer, model_params, lr=self.config.lr_scheduler.max_lr, fused=True)
        else:
            optimizer = instantiate(self.config.optimizer, model_params, lr=self.config.lr_scheduler.max_lr)
        return optimizer

    def _schedule_lr(self, step):
        # Linear warmup
        if step < self.config.lr_scheduler.warmup_steps:
            # start from max_lr/warmup_steps and goes to max_lr
            return (
                (step + 1) / self.config.lr_scheduler.warmup_steps
            )
        # Constant learning rate after max_steps
        if step > self.config.lr_scheduler.max_steps:
            return self.config.lr_scheduler.min_lr / self.config.lr_scheduler.max_lr
        # Cosine decay
        decay_ratio = (step - self.config.lr_scheduler.warmup_steps) / (
            self.config.lr_scheduler.max_steps - self.config.lr_scheduler.warmup_steps
        )
        assert 0 <= decay_ratio <= 1, "Decay ratio must be in [0, 1]"
        # coeff starts from 1 and goes to 0
        coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return (self.config.lr_scheduler.min_lr + coeff * (
            self.config.lr_scheduler.max_lr - self.config.lr_scheduler.min_lr
        )) / self.config.lr_scheduler.max_lr