import tiktoken
import torch


class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open('data/input.txt', 'r') as f:
            data = f.read()

        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(data)
        self.tokens = torch.tensor(tokens)
        self.current_pos = 0

    def __next__(self):
        buf = self.tokens[self.current_pos:self.current_pos + self.B * self.T + 1]
        x = buf[:-1]
        y = buf[1:]
        x = x.view(self.B, self.T)
        y = y.view(self.B, self.T)
        self.current_pos += self.B * self.T
        if self.current_pos + self.B * self.T + 1 > len(self.tokens):
            self.current_pos = 0
        return x, y
