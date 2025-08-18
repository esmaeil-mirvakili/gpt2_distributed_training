import sys
from dataclasses import dataclass, field
from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore
from loguru import logger
from omegaconf import OmegaConf

from trainer.ddp_trainer import DDPTrainer, TrainerConfig


def init_logger():
    logger.add("logs/training_{time}.log")


@dataclass
class Config:
    device: Optional[str] = None
    training_type: str = "ddp"
    resume: Optional[str] = None
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def __post_init__(self):
        allowed_devices = ["cpu", "mps", "cuda"]
        if self.device not in allowed_devices:
            raise ValueError(
                f"Invalid device '{self.device}'. Allowed values are {allowed_devices}."
            )
        allowed_training_types = ["ddp", "fsdp", "deepspeed"]
        if self.training_type not in allowed_training_types:
            raise ValueError(
                f"Invalid training_type '{self.training_type}'. Allowed values are {allowed_training_types}."
            )


cs = ConfigStore.instance()
cs.store(name="gpt2_training_config", node=Config)


@hydra.main(version_base=None, config_path="configs", config_name="ddp_training")
def main(cfg: Optional[Config] = None):
    init_logger()
    config = OmegaConf.merge(OmegaConf.structured(Config), cfg)
    logger.info(f"Configuration loaded: {type(config)}")
    trainer = None
    if getattr(config, "training_type", "ddp") == "ddp":
        trainer = DDPTrainer(
            config.trainer,
            getattr(config, "device", None),
            getattr(config, "resume", None),
        )
    else:
        raise ValueError(
            f"Unsupported training type '{config.training_type}'. Only 'ddp' is currently supported."
        )
    trainer.train()


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
