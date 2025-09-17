from abc import ABC
import os
from dataclasses import dataclass
from typing import Optional, Literal
import torch
from loguru import logger
import wandb

@dataclass
class BaseTrainerConfig:
    wandb_project: Optional[str] = "gpt2-distributed-training"
    wandb_run_name: Optional[str] = "training-run"

class BaseTrainer(ABC):

    _NOT_IMPLEMENTED_MSG = "This method should be implemented by subclasses."

    def __init__(
        self,
        device: Optional[Literal["cpu", "mps", "cuda"]] = None,
        resume: Optional[bool] = False,
        config: Optional[BaseTrainerConfig] = None,
    ):
        self.device = device
        self.resume = resume
        self.config = config or BaseTrainerConfig()
        os.environ.setdefault("WANDB_PROJECT", config.wandb_project)
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
        )
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")
        
    def _initialize(self, model: torch.nn.Module, train_dataset, val_dataset):
        raise NotImplementedError(self._NOT_IMPLEMENTED_MSG)
    
    def _destroy(self):
        raise NotImplementedError(self._NOT_IMPLEMENTED_MSG)
    
    @logger.catch
    def train(self, model: torch.nn.Module, train_dataset, val_dataset):
        raise NotImplementedError(self._NOT_IMPLEMENTED_MSG)

    def _train_step(self, train_dataset, step):
        raise NotImplementedError(self._NOT_IMPLEMENTED_MSG)

    def _val_step(self, val_dataset, step):
        raise NotImplementedError(self._NOT_IMPLEMENTED_MSG)

    def _prepare_model(self, model: torch.nn.Module):
        raise NotImplementedError(self._NOT_IMPLEMENTED_MSG)

    def _prepare_datasets(self, train_dataset, val_dataset):
        raise NotImplementedError(self._NOT_IMPLEMENTED_MSG)

    def _prepare_optimizer(self):
        raise NotImplementedError(self._NOT_IMPLEMENTED_MSG)
    
    def _log_metrics(self, metrics: dict, step: int, prefix: str = "train"):
        metrics_str = []
        for key, value in metrics.items():
            if isinstance(value, float):
                metrics_str.append(f"{key}={value:.4f}")
            else:
                metrics_str.append(f"{key}={value}")
        metrics_str = " | ".join(metrics_str)
        logger.info(f"{prefix} metrics at step {step}: {metrics_str}")
        if wandb.run is not None:
            wandb_metrics = {f"{prefix}/{key}": value for key, value in metrics.items()}
            wandb_metrics["global_step"] = step
            wandb.log(wandb_metrics)