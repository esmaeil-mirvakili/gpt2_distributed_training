import time
import argparse
import torch
import torch.nn.functional as F
from data.data import DataLoaderLite
from models.gpt2 import GPT2, GPT2Config


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
    train_loader = DataLoaderLite(B, T)

    model = GPT2(GPT2Config(vocab_size=50304))  # 50304 is a nice number => lots of power of 2: 393 * 128
    model.to(device)

    # As of now, torch.compile() is not supported on MPS (Apple Metal Performance Shaders)
    if device != 'mps':
        # compile model to make the code fast
        # adds compilation time to the training
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-04, betas=(0.9, 0.95), eps=1e-8)
    for i in range(50):
        t0 = time.time()
        x, y = next(train_loader)
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            # Some CUDA ops might not be cast: https://docs.pytorch.org/docs/stable/amp.html#cuda-op-specific-behavior
            logits, loss = model(x, y)
        loss.backward()
        # gradient norm clipping => prevent the model from getting big shocks in terms of gradient magnitude
        # clip the global norm of the gradient at 1.0 (GPT3 hyperparam)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # global norm of the params

        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # for calculating the step time
        t1 = time.time()
        dt = (t1 - t0) * 1000
        token_per_sec = (B * T) / (t1 - t0)
        print(f'Step {i}: loss={loss} | norm: {norm:.2f} | dt={dt:.2f}ms | tok/sec={token_per_sec:.2f}')


def parse_args():
    parser = argparse.ArgumentParser(description="GPT2 Single GPU Training")
    parser.add_argument('--device', default=None, help="Precision for fp operations on GPU")
    parser.add_argument('--precision', default='highest', help="Precision for fp operations on GPU")
    parser.add_argument('--flash_att', action='store_true', default=False, help="Use flash attention.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
