import tiktoken
import torch


class DataLoaderLite:
    def __init__(self, B, T, world_size=1, rank=0):
        self.B = B
        self.T = T
        self.world_size = world_size
        self.rank = rank

        with open('data/input.txt', 'r') as f:
            data = f.read()

        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(data)
        self.tokens = torch.tensor(tokens)
        self.current_pos = B * T * rank

    def __next__(self):
        buf = self.tokens[self.current_pos:self.current_pos + self.B * self.T + 1]
        x = buf[:-1]
        y = buf[1:]
        x = x.view(self.B, self.T)
        y = y.view(self.B, self.T)
        self.current_pos += self.B * self.T * self.world_size
        if self.current_pos + self.B * self.T * self.world_size + 1 > len(self.tokens):
            self.current_pos = self.B * self.T * self.rank
        return x, y
