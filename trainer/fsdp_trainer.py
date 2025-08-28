import os
import re
from dataclasses import dataclass
from typing import List, Literal, Optional, Union

import torch
import torch.nn as nn
from loguru import logger
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    _init_optim_state,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor, distribute_tensor

from trainer.ddp_trainer import DDPTrainer, DDPTrainerConfig
from trainer.utils import _to_scalar, CheckpointStrategy


@dataclass
class FSDPTrainerConfig(DDPTrainerConfig):
    forward_prefetch_modules: Optional[int] = None
    backward_prefetch_modules: Optional[int] = None
    use_mixed_precision: bool = True


class FSDPCheckpointStrategy(CheckpointStrategy):
    def save_checkpoint(self, model, optimizer, loss, step, is_master):
        checkpoint_dir = self.config.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step:07d}.pt")
        state = {
            "step": step,
            "avg_loss": _to_scalar(loss),
            "model": self._get_model_state_dict(model, is_master),
            "optimizer": self._get_optimizer_state_dict(optimizer, model, is_master),
        }
        if is_master:
            torch.save(state, checkpoint_path)

    def load_checkpoint(self, model, optimizer, device, is_master):
        checkpoint_pattern = r"checkpoint_(\d+)\.pt"
        latest_checkpoint = None
        latest_step = -1
        checkpoint_dir = self.config.checkpoint_dir
        if is_master:
            os.makedirs(checkpoint_dir, exist_ok=True)
            for filename in os.listdir(checkpoint_dir):
                match = re.match(checkpoint_pattern, filename)
                if match:
                    step = int(match.group(1))
                    if step > latest_step:
                        latest_step = step
                        latest_checkpoint = os.path.join(checkpoint_dir, filename)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            obj = [latest_checkpoint, latest_step]
            torch.distributed.broadcast_object_list(obj, src=0)
            latest_checkpoint, latest_step = obj[0], obj[1]
        if latest_checkpoint is None:
            if is_master:
                logger.warning("No checkpoint found.")
            return 0
        if is_master:
            logger.info(f"Loading checkpoint from {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint, map_location=device)
        else:
            checkpoint = None
        self._load_model_state_dict(checkpoint["model"] if is_master else {}, model)
        self._load_optimizer_state_dict(checkpoint["optimizer"] if is_master else {}, optimizer, model)
        if is_master:
            logger.info(
                f"Loaded checkpoint step {checkpoint['step']} with avg_loss {checkpoint['avg_loss']}"
            )
        return latest_step + 1

    def _get_model_state_dict(self, model, is_master: bool):
        return get_model_state_dict(
            model=model,
            options=StateDictOptions(
                full_state_dict=True, cpu_offload=self.config.use_cpu_offload
            ),
        )

    def _load_model_state_dict(self, state_dict, model):
        set_model_state_dict(
            model=model,
            model_state_dict=state_dict,
            options=StateDictOptions(
                full_state_dict=True, broadcast_from_rank0=True
            ),
        )

    def _get_optimizer_state_dict(self, optimizer, model, is_master: bool):
        return get_optimizer_state_dict(
            model=model,
            optimizers=optimizer,
            options=StateDictOptions(
                full_state_dict=True, cpu_offload=self.config.use_cpu_offload
            ),
        )

    def _load_optimizer_state_dict(self, state_dict, optimizer, model):
        set_optimizer_state_dict(
            model=model,
            optimizers=optimizer,
            optim_state_dict=state_dict,
            options=StateDictOptions(
                full_state_dict=True, broadcast_from_rank0=True
            ),
        )


class FSDPTrainer(DDPTrainer):
    def __init__(
        self,
        device: Optional[Literal["cpu", "mps", "cuda"]] = None,
        resume: Optional[bool] = False,
        config: Optional[FSDPTrainerConfig] = None,
    ):
        super().__init__(device=device, resume=resume, config=config)

    def _prepare_model(self, model: torch.nn.Module):
        self.raw_model = model
        self.raw_model.to(self.device)
        if self.config.use_model_compile and self.device_type != "mps":
            self.raw_model = torch.compile(self.raw_model)
        if self.distributed:
            # apply fully_shard on layers and root model
            fsdp_kwargs = self._get_fsdp_kwargs()
            for layer in self.raw_model.layers:
                fully_shard(layer, **fsdp_kwargs)
            fully_shard(self.raw_model, **fsdp_kwargs)
            assert isinstance(
                self.raw_model, FSDPModule
            ), "Model must be wrapped in FSDPModule"
            self._prepare_prefetching()
        return self.raw_model

    def _prepare_prefetching(self):
        if self.config.forward_prefetch_modules:
            for i in range(len(self.raw_model.layers)):
                layers_to_prefetch = self.get_next_n_layers(
                    self.raw_model.layers, i, self.config.forward_prefetch_modules
                )
                self.raw_model.layers[i].set_modules_to_forward_prefetch(
                    layers_to_prefetch
                )
        if self.config.backward_prefetch_modules:
            for i in range(len(self.raw_model.layers)):
                layers_to_prefetch = self.get_next_n_layers(
                    self.raw_model.layers, i, self.config.backward_prefetch_modules
                )
                self.raw_model.layers[i].set_modules_to_backward_prefetch(
                    layers_to_prefetch
                )

    def _get_fsdp_kwargs(self):
        fsdp_kwargs = {}
        if self.config.use_mixed_precision:
            # Use bfloat16 for parameters and float32 for reductions
            fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32
            )
        return fsdp_kwargs

    @staticmethod
    def get_next_n_layers(layers, i, n):
        return [layers[i + j] for j in range(1, min(n, len(layers) - i))]
    
    def _initialize(self, model, train_dataset, val_dataset):
        ret = super()._initialize(model, train_dataset, val_dataset)
        assert self.distributed, "FSDPTrainer requires distributed training"
        return ret
    
