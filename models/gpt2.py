import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPT2Config:
    block_size: int = 1024  # max seq length
    vocab_size: int = 50257  # number of tokens = 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of attention heads
    n_embd: int = 768  # embedding dimension
    use_flash_attn: bool = True  # use flash attention implementation


class GPT2(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            # token encoding: vocab_size => n_embd
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            # position encoding: block_size -> n_embd
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            # hidden decode blocks
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # final LayerNorm
            'ln_f': nn.LayerNorm(config.n_embd)
        })
        # Language Model Head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # scale activations (in mlp) to count for residual connection
                # std *= 1 / sqrt(number of residual connections)
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # batch size, seq length
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # positions
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # (T)
        pos_embd = self.transformer.wpe(pos)  # (T, n_embd)
        tok_embd = self.transformer.wte(idx)  # (B, T, n_emb)
        x = pos_embd + tok_embd  # broadcasting: (B, T, n_embd)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints

        config = GPT2Config(**config_args)
        model = GPT2(config)
        sd = model.state_dict()
        # ignoring the bias => they are just buffers
        sd_keys = [key for key in sd.keys() if not key.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [key for key in sd_hf.keys() if not key.endswith('.attn.masked_bias')]
        sd_keys_hf = [key for key in sd_keys_hf if not key.endswith('.attn.bias')]

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # OpenAI checkpoint uses Conv1D for these layers => we need to transpose them inorder to use linear layers
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for key in sd_keys_hf:
            if any(key.endswith(w) for w in transposed):
                # make sure that the weights should be transposed
                assert sd_hf[key].shape[::-1] == sd[key].shape
                with torch.no_grad():
                    sd[key].copy_(sd_hf[key].t())
            else:
                # make sure that the weights matches ours
                assert sd_hf[key].shape == sd[key].shape
                sd[key].copy_(sd_hf[key])
        return model


class Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # query, key, and value projection
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size
        self.use_flash_attn = config.use_flash_attn
        # mask (OpenAI calls it bias)
        self.register_buffer('bias',
                             torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size,
                                                                                               config.block_size))

    def forward(self, x):
        # batch_size, seq_len, embedding dim
        B, T, C = x.size()
        assert C == self.n_embd

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.n_embd // self.n_head)   # (B, T, n_head, head_dim)
        q = q.transpose(1, 2)   # (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.n_embd // self.n_head)   # (B, T, n_head, head_dim)
        k = k.transpose(1, 2)   # (B, n_head, T, head_dim)
        v = v.view(B, T, self.n_head, self.n_embd // self.n_head)   # (B, T, n_head, head_dim)
        v = v.transpose(1, 2)   # (B, n_head, T, head_dim)

        if self.use_flash_attn:
            # y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=self.bias[:, :, :T, :T] == 0)
        else:
            attn = (q @ k.transpose(-2, -1))    # (B, n_head, T, T)
            # normalization
            attn = attn * (1.0 / math.sqrt(self.n_embd // self.n_head))
            attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            y = attn @ v    # (B, n_head, T, head_dim)

        y = y.transpose(1, 2)   # (B, T, n_head, head_dim)
        y = y.contiguous().view(B, T, self.n_embd)  # (B, T, n_embd)
        y = self.c_proj(y)
        return y
