import argparse
import math
import time
import torch
import torch.nn.functional as F
from data.data import DataLoaderLite
from models.gpt2 import GPT2, GPT2Config
from trainer.utils import configure_optimizers


def eval():
    num_return_sequences = 5
    max_length = 30
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    print(f'using device: {device}')

    model = GPT2.from_pretrained('gpt2')
    # model = GPT2(GPT2Config())
    print('weights are loaded!')
    model.eval()
    model.to(device)

    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode("Hello, I am a language model that")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    x = tokens.to(device)

    torch.manual_seed(42)
    if device.startswith('cuda'):
        torch.cuda.manual_seed(42)
    while x.size(1) < max_length:
        print(f'predicting token {x.size(1)}')
        with torch.no_grad():
            logits, _ = model(x)  # (B, T, vocab_size)
            # take logit at the last position
            logits = logits[:, -1, :]  # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select one token from top-k probabilities
            ix = torch.multinomial(topk_probs, 1)  # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
            # append the predicted token to the x
            x = torch.cat((x, xcol), dim=1)

    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print('> ', decoded)


def train(args):
    if args.device is None:
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
    else:
        device = args.device
    print(f'using device: {device}')

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # change the matmul precision:
    #   highest:    fp32 (24 mantissa bits with 23 bits explicitly stored)
    #   high:       TensorFloat32 (10 mantissa bits explicitly stored)
    #   medium:     bfloat16 (8 mantissa bits with 7 bits explicitly stored)
    torch.set_float32_matmul_precision(args.precision)

    B, T = 1, 1024
    total_batch_size = 4 * T
    assert total_batch_size % (B * T) == 0, "total batch size should be divisible by B * T"
    grad_accum_steps = total_batch_size // (B * T)
    print(f'Total batch size is {total_batch_size}:')
    print(f'\tAccumulation steps: {grad_accum_steps}')

    train_loader = DataLoaderLite(B, T)

    model = GPT2(GPT2Config(vocab_size=50304))  # 50304 is a nice number => lots of power of 2: 393 * 128
    model.to(device)

    # As of now, torch.compile() is not supported on MPS (Apple Metal Performance Shaders)
    if device != 'mps':
        # compile model to make the code fast
        # adds compilation time to the training
        model = torch.compile(model)

    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 10
    max_steps = 50

    # cosine learning rate scheduler
    def get_lr(it):
        # 1. linear warmup
        if it < warmup_steps:
            # starts at max_lr/warmup_steps goes to max_lr
            return max_lr * (it + 1) / warmup_steps
        # 2. if beyond lr_decay_iters, return min_lr
        if it > max_steps:
            return min_lr
        # 3. in between => use cosine decay
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        # starts at 1 goes to 0
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-04, betas=(0.9, 0.95), eps=1e-8)
    optimizer = configure_optimizers(model, weight_decay=0.1, learning_rate=64-4, device=device)
    for step in range(50):
        t0 = time.time()
        optimizer.zero_grad()
        # for logging
        accum_loss = 0
        # gradient accumulation => equivalent to sum(loss)
        for micro_step in range(grad_accum_steps):
            x, y = next(train_loader)
            x = x.to(device)
            y = y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                # Some CUDA ops might not be cast: https://docs.pytorch.org/docs/stable/amp.html#cuda-op-specific-behavior
                logits, loss = model(x, y)
            # scaling loss because of gradient accumulation => we need to average the sum
            loss = loss / grad_accum_steps
            accum_loss += loss.detach()
            loss.backward()
        # gradient norm clipping => prevent the model from getting big shocks in terms of gradient magnitude
        # clip the global norm of the gradient at 1.0 (GPT3 hyperparam)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # global norm of the params

        # learning rate scheduling
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


        optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()  # for calculating the step time
        t1 = time.time()
        dt = (t1 - t0) * 1000
        token_per_sec = (B * T) / (t1 - t0)
        print(f'Step {step}: loss={accum_loss} | norm: {norm:.2f} | dt={dt:.2f}ms | tok/sec={token_per_sec:.2f}')


def parse_args():
    parser = argparse.ArgumentParser(description="GPT2 Single GPU Training")
    parser.add_argument('--device', default=None, help="Precision for fp operations on GPU")
    parser.add_argument('--precision', default='highest', help="Precision for fp operations on GPU")
    parser.add_argument('--flash_att', action='store_true', default=False, help="Use flash attention.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
