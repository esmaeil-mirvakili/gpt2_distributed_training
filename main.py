import sys
from dataclasses import dataclass, field
from typing import Optional, Any
import os

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf, MISSING

from trainer.ddp_trainer import DDPTrainer, DDPTrainerConfig
from trainer.fsdp_trainer import FSDPTrainer, FSDPTrainerConfig
from trainer.base_trainer import BaseTrainer, BaseTrainerConfig
from models.gpt2 import GPT2Config


def init_logger(path: str):
    logger.add(os.path.join(path, "training_{time}.log"))


@dataclass
class ModelConfig:
    _target_: str = "models.gpt2.GPT2"
    config: GPT2Config = field(default_factory=GPT2Config)


@dataclass
class TrainerConfig:
    device: Optional[str] = None
    resume: Optional[bool] = None
@dataclass
class DDPConfig(TrainerConfig):
    _target_: str = "trainer.ddp_trainer.DDPTrainer"
    config: DDPTrainerConfig = field(default_factory=DDPTrainerConfig)
    
@dataclass
class FSDPConfig(TrainerConfig):
    _target_: str = "trainer.fsdp_trainer.FSDPTrainer"
    config: FSDPTrainerConfig = field(default_factory=FSDPTrainerConfig)
    

@dataclass
class Config:
    trainer: TrainerConfig = MISSING
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: dict = field(default_factory=dict)
    log_dir: str = "logs/"


cs = ConfigStore.instance()
cs.store(name="schema", node=Config)
cs.store(group="trainer", name="ddp", node=DDPConfig)
cs.store(group="trainer", name="fsdp", node=FSDPConfig)

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(config: Config):
    init_logger(config.log_dir)
    logger.info(f"Configuration loaded: \n{OmegaConf.to_yaml(config, resolve=True)}")
    trainer = instantiate(config.trainer, _convert_="partial", _recursive_=False)
    model = instantiate(config.model, _recursive_=True)
    train_dataset = instantiate(config.dataset.train_dataset)
    val_dataset = instantiate(config.dataset.val_dataset)
    trainer.train(model, train_dataset, val_dataset)


def hydra_arg_fix():
    hydra_formatted_args = []
    for arg in sys.argv:
        if arg.startswith("--"):
            hydra_formatted_args.append(arg.replace("--", ""))
        else:
            hydra_formatted_args.append(arg)
    sys.argv = hydra_formatted_args


if __name__ == "__main__":
    # hydra_arg_fix()
    main()
