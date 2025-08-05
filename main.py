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
    training_type: Literal["single_gpu", "ddp", "fsdp", "deepspeed"] = "single_gpu"
    resume: Optional[str] = None


@hydra.main(version_base=None, config_path="configs", config_name="single_gpu")
def main(config: Config):
    init_logger()
    logger.info("Configuration loaded: ", config)
    trainer = DDPTrainer(
        config.trainer,
        config.device,
        getattr(config, "resume", None),
    )
    trainer.train()


if __name__ == "__main__":
    main()
