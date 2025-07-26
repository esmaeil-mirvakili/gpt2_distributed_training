import argparse
import math
import os
import time
import torch
import torch.nn.functional as F
from data.data import DataLoaderLite
from models.gpt2 import GPT2, GPT2Config
from trainer.utils import configure_optimizers
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP


def train(args):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    with open(log_file, "w") as f:  # open for writing to clear the file
        pass

    # initializing the ddp
    ddp = int(os.environ.get("RANK", -1)) != -1 and torch.cuda.is_available()
    if ddp:
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])  # process rank globally
        ddp_local_rank = int(
            os.environ["LOCAL_RANK"]
        )  # process rank in the node (locally) => gpu id on the node
        ddp_world_size = int(os.environ["WORLD_SIZE"])  # total number of processes
        device = f"cuda:{ddp_local_rank}"
        # master process do some coordination and synchronization
        is_master = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        is_master = True
        if args.device is None:
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
        else:
            device = args.device

    device_type = "cuda" if device.startswith("cuda") else "cpu"

    if is_master:
        print(f"DDP activated: {ddp}")
        print(f"world size: {ddp_world_size}")
        print(f"rank: {ddp_rank}")
        print(f"local rank: {ddp_local_rank}")

    print(f"using device: {device}")
    if "cuda" in device:
        torch.cuda.set_device(torch.device(device))

    # important to set the seed the same for all processes in ddp => start form the same initialization
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # change the matmul precision:
    #   highest:    fp32 (24 mantissa bits with 23 bits explicitly stored)
    #   high:       TensorFloat32 (10 mantissa bits explicitly stored)
    #   medium:     bfloat16 (8 mantissa bits with 7 bits explicitly stored)
    torch.set_float32_matmul_precision(args.precision)

    B, T = 64, 1024
    total_batch_size = 2**19
    assert (
        total_batch_size % (B * T * ddp_world_size) == 0
    ), "total batch size should be divisible by B * T * world_size"
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if is_master:
        # print once
        print(f"Total batch size is {total_batch_size}:")
        print(f"\tAccumulation steps: {grad_accum_steps}")

    train_loader = DataLoaderLite(B, T, world_size=ddp_world_size, rank=ddp_rank)
    val_loader = DataLoaderLite(
        B, T, world_size=ddp_world_size, rank=ddp_rank, split="val"
    )

    raw_model = GPT2(
        GPT2Config(vocab_size=50304)
    )  # 50304 is a nice number => lots of power of 2: 393 * 128
    raw_model.to(device)

    # As of now, torch.compile() is not supported on MPS (Apple Metal Performance Shaders)
    if device != "mps":
        # compile model to make the code fast
        # adds compilation time to the training
        raw_model = torch.compile(raw_model)

    # wrap th emodel in DDP for ddp training
    if ddp:
        model = DDP(raw_model, device_ids=[ddp_local_rank])

    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 715
    max_steps = 19073

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
    optimizer = configure_optimizers(
        raw_model, weight_decay=0.1, learning_rate=64 - 4, device=device
    )
    for step in range(max_steps):
        is_last_step = step == max_steps - 1
        if step % 250 == 0 or is_last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_accum_loss = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = next(val_loader)
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_accum_loss += loss.detach()
                if ddp:
                    torch.distributed.all_reduce(
                        val_accum_loss, op=torch.distributed.ReduceOp.AVG
                    )
                if is_master:
                    print(f"validation loss: {val_accum_loss.item():.4f}")
                    with open(log_file, "a") as f:
                        f.write(f"{step} val {val_accum_loss.item():.4f}\n")
                    if step > 0 and (step % 5000 == 0 or is_last_step):
                        # optionally write model checkpoints
                        checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                        checkpoint = {
                            "model": raw_model.state_dict(),
                            "config": raw_model.config,
                            "step": step,
                            "val_loss": val_accum_loss.item(),
                        }
                        # you might also want to add optimizer.state_dict() and
                        # rng seeds etc., if you wanted to more exactly resume training
                        torch.save(checkpoint, checkpoint_path)
        t0 = time.time()
        model.train()
        optimizer.zero_grad()
        # for logging
        accum_loss = 0
        # gradient accumulation => equivalent to sum(loss)
        for micro_step in range(grad_accum_steps):
            x, y = next(train_loader)
            x = x.to(device)
            y = y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                # Some CUDA ops might not be cast: https://docs.pytorch.org/docs/stable/amp.html#cuda-op-specific-behavior
                logits, loss = model(x, y)
            # scaling loss because of gradient accumulation => we need to average the sum
            loss = loss / grad_accum_steps
            accum_loss += loss.detach()

            # only activate the gradient syncing between ddp processes on the last micro batch
            # this way is a little problematic if the logic of pytorch changes => use ddp.no_sync() instead
            if ddp:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            loss.backward()

        # collect and average the accum_loss from all processes
        if ddp:
            torch.distributed.all_reduce(accum_loss, op=torch.distributed.ReduceOp.AVG)

        # gradient norm clipping => prevent the model from getting big shocks in terms of gradient magnitude
        # clip the global norm of the gradient at 1.0 (GPT3 hyperparam)
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0
        )  # global norm of the params

        # learning rate scheduling
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()  # for calculating the step time
        t1 = time.time()
        dt = (t1 - t0) * 1000
        token_per_sec = (B * T * grad_accum_steps * ddp_world_size) / (t1 - t0)
        if is_master:
            print(
                f"Step {step}: loss={accum_loss} | norm: {norm:.2f} | dt={dt:.2f}ms | tok/sec={token_per_sec:.2f}"
            )

        # destroy the process group
        if ddp:
            destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description="GPT2 Single GPU Training")
    parser.add_argument(
        "--device", default=None, help="Precision for fp operations on GPU"
    )
    parser.add_argument(
        "--precision", default="highest", help="Precision for fp operations on GPU"
    )
    parser.add_argument(
        "--flash_att", action="store_true", default=False, help="Use flash attention."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
