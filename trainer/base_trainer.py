from abc import ABC
from dataclasses import dataclass
from typing import Optional, Literal
import torch

@dataclass
class BaseTrainerConfig:
    pass

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
        
    def _initialize(self, model: torch.nn.Module, train_dataset, val_dataset):
        raise NotImplementedError(self._NOT_IMPLEMENTED_MSG)

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