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


def load_tokens(filename):
    data = np.load(filename)
    return torch.tensor(data, dtype=torch.long)


class DataLoaderLite:
    def __init__(self, batch_size, seq_length, data_root=None, split="train"):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.rank = int(os.environ.get("RANK", 0))
        self.current_shard = 0
        self.tokens = None
        self.current_pos = 0
        assert split in {"train", "val"}
        assert data_root is not None, "data_root must be specified"

        # get the shard filenames
        shards = os.listdir(data_root)
        shards = filter(lambda s: split in s, shards)
        shards = sorted(shards)
        self.split = split
        self.shards = [os.path.join(data_root, s) for s in shards]
        assert len(shards) > 0, f"no shards found for split {split}"
        if self.rank == 0:
            logger.info(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_pos = self.batch_size * self.seq_length * self.rank

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        if self.split != "train" and self.current_pos + (self.batch_size * self.seq_length * self.world_size + 1) > len(
            self.tokens
        ):
            raise StopIteration
        buf = self.tokens[self.current_pos : self.current_pos + self.batch_size * self.seq_length + 1]
        x = buf[:-1].view(self.batch_size, self.seq_length)
        y = buf[1:].view(self.batch_size, self.seq_length)
        self.current_pos += self.batch_size * self.seq_length * self.world_size
        if self.split == "train" and self.current_pos + (self.batch_size * self.seq_length * self.world_size + 1) > len(
            self.tokens
        ):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_pos = self.batch_size * self.seq_length * self.rank
        return x, y
