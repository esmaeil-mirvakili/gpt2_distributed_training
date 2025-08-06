from dataclasses import dataclass, field
from typing import Literal, Optional

import hydra
from loguru import logger

from trainer.ddp_trainer import DDPTrainer, TrainerConfig


def init_logger():
    logger.add("logs/training_{time}.log")


@dataclass
class Config:
    trainer: TrainerConfig = field(default_factory=lambda: TrainerConfig())
    device: Literal["cpu", "mps", "cuda"] = "cpu"
    training_type: Literal["ddp", "fsdp", "deepspeed"] = "ddp"
    resume: Optional[str] = None


@hydra.main(version_base=None, config_path="configs", config_name="ddp_training")
def main(config: Config):
    init_logger()
    logger.info("Configuration loaded: ", config)
    if getattr(config, "training_type", "ddp") == "ddp":
        trainer = DDPTrainer(
            config.trainer,
            getattr(config, "device", None),
            getattr(config, "resume", None),
        )
    trainer.train()


if __name__ == "__main__":
    main()
