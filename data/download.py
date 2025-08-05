import argparse
import multiprocessing as mp
import os
from functools import partial

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

DATASET_DICT = {
    "fineweb": {
        "dataset_name": "HuggingFaceFW/fineweb",
        "dataset_subset": "sample-10BT",
    },
    "wikitext": {
        "dataset_name": "Salesforce/wikitext",
        "dataset_subset": "wikitext-103-v1",
    },
}


# tokenize a document
def tokenize(doc, enc, eot):
    # <|endoftext|> is a delimiter for text => each doc should start with <|endoftext|>
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    tokens_np = tokens_np.astype(np.uint16)
    return tokens_np


def download(
    local_dir="data",
    dataset="fineweb",
    shard_size=int(1e8),
):
    # create cache local dir
    DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir, dataset)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # download from huggingface[
    # fine_web_dataset = load_dataset("HuggingFaceFW/fineweb", name=dataset_name, split="train")
    dataset_data = load_dataset(
        DATASET_DICT[dataset]["dataset_name"],
        DATASET_DICT[dataset]["dataset_subset"],
        split="train",
    )
    print(
        f"Loaded {len(dataset_data)} documents from {DATASET_DICT[dataset]['dataset_name']}"
    )

    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens["<|endoftext|>"]

    def write_datafile(filename, tokens_np):
        np.save(filename, tokens_np)

    tokenizer_func = partial(tokenize, enc=enc, eot=eot)

    # tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
    nprocs = max(1, os.cpu_count() // 2)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # preallocate buffer to hold current shard
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        for tokens in pool.imap(tokenizer_func, dataset_data, chunksize=16):

            # is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < shard_size:
                # simply append tokens to current shard
                all_tokens_np[token_count : token_count + len(tokens)] = tokens
                token_count += len(tokens)
                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(
                        total=shard_size, unit="tokens", desc=f"Shard {shard_index}"
                    )
                progress_bar.update(len(tokens))
            else:
                # write the current shard and start a new one
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(
                    DATA_CACHE_DIR, f"{dataset}_{split}_{shard_index:06d}"
                )
                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count : token_count + remainder] = tokens[
                    :remainder
                ]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                # populate the next shard with the leftovers of the current doc
                all_tokens_np[0 : len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

        # write any remaining tokens as the last shard
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(
                DATA_CACHE_DIR, f"{dataset}_{split}_{shard_index:06d}"
            )
            write_datafile(filename, all_tokens_np[:token_count])


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset downloader")
    parser.add_argument("--dir", default="data/", help="Directory to save the dataset")
    parser.add_argument(
        "--dataset",
        default="fineweb",
        choices=["fineweb", "wikitext"],
        help="Dataset name to download",
    )
    parser.add_argument(
        "--shard_size", default=int(1e8), type=int, help="Use flash attention."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download(local_dir=args.dir, dataset=args.dataset, shard_size=args.shard_size)
