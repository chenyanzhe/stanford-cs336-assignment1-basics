import argparse
import os

import numpy as np

from cs336_basics.tokenizer import Tokenizer

BATCH_SIZE = 1_000_000


def tokenize_dataset(
    input_path: str | os.PathLike,
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
    output_path: str | os.PathLike,
    special_tokens: list[str],
):
    total_tokens = 0
    tokenizer = Tokenizer.from_files(
        vocab_filepath=vocab_path, merges_filepath=merges_path, special_tokens=special_tokens
    )
    buffer = []

    with open(input_path, encoding="utf-8") as f_in, open(output_path, "wb") as f_out:
        for token_id in tokenizer.encode_iterable(f_in):
            buffer.append(token_id)
            if len(buffer) >= BATCH_SIZE:
                total_tokens += len(buffer)
                np.array(buffer, dtype=np.uint16).tofile(f_out)
                buffer.clear()
                print(f"Tokenized {total_tokens:,} tokens...")
        if buffer:
            total_tokens += len(buffer)
            np.array(buffer, dtype=np.uint16).tofile(f_out)

    print(f"Finished tokenizing {total_tokens:,} tokens.")
    print(f"Saved tokenized data to {output_path}, size: {os.path.getsize(output_path):,} bytes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="tokenize_dataset")
    parser.add_argument(
        "--input-path",
        default="data/TinyStoriesV2-GPT4-train.txt",
        help="Path to a text file with training data.",
    )
    parser.add_argument(
        "--vocab-path",
        default="data/TinyStoriesV2-vocab.pkl",
        help="Path to a vocabulary file for the tokenizer.",
    )
    parser.add_argument(
        "--merges-path",
        default="data/TinyStoriesV2-merges.pkl",
        help="Path to a merges file for the tokenizer.",
    )
    parser.add_argument(
        "--output-path", default="data/tinystories_train.bin", help="Output path to store tokenized data."
    )
    parser.add_argument(
        "--special-tokens", nargs="*", default=["<|endoftext|>"], help="A list of strings to add to the vocabulary."
    )
    args = parser.parse_args()
    tokenize_dataset(
        input_path=args.input_path,
        vocab_path=args.vocab_path,
        merges_path=args.merges_path,
        output_path=args.output_path,
        special_tokens=args.special_tokens,
    )
    print("All done.")
