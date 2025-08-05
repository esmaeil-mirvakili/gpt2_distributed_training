import os

import numpy as np
import tiktoken
import torch
from attr import dataclass
from loguru import logger


@dataclass
class DatasetConfig:
    _target_: str = "data.DataLoaderLite"
    data_root: str = "data/wikitext"
    seq_length: int = 1024


def load_tokens(filename):
    data = np.load(filename)
    return torch.tensor(data, dtype=torch.long)


class DataLoaderLite:
    def __init__(self, B, T, data_root=None, world_size=1, rank=0, split="train"):
        self.B = B
        self.T = T
        self.world_size = world_size
        self.rank = rank
        self.current_shard = 0
        self.tokens = None
        self.current_pos = 0
        assert split in {"train", "val"}
        assert data_root is not None, "data_root must be specified"

        # get the shard filenames
        shards = os.listdir(data_root)
        shards = filter(lambda s: split in s, shards)
        shards = sorted(shards)
        self.shards = [os.path.join(data_root, s) for s in shards]
        assert len(shards) > 0, f"no shards found for split {split}"
        if rank == 0:
            logger.info(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_pos = self.B * self.T * self.rank

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        buf = self.tokens[self.current_pos : self.current_pos + self.B * self.T + 1]
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)
        self.current_pos += self.B * self.T * self.world_size
        if self.current_pos + (self.B * self.T * self.world_size + 1) > len(
            self.tokens
        ):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_pos = self.B * self.T * self.rank
        return x, y
