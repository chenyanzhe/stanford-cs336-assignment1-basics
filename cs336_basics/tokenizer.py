import os
import time
import argparse
import pickle
import pathlib
import heapq
import regex as re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from cs336_basics.pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
COMPILED_PAT = re.compile(PAT)


class MaxHeapItem:
    def __init__(self, count, pair):
        self.count = count
        self.pair = pair

    def __lt__(self, other):
        if self.count != other.count:
            return self.count > other.count

        return self.pair > other.pair


def merge_pair(tokens: list[bytes], pair: tuple[bytes, bytes]) -> list[bytes]:
    A, B = pair
    result = []

    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i] == A and tokens[i + 1] == B:
            result.append(A + B)
            i += 2
        else:
            result.append(tokens[i])
            i += 1

    return result


def count_pairs(tokens: list[bytes]) -> defaultdict[int]:
    pair_counts = defaultdict(int)
    for i, j in zip(tokens[:-1], tokens[1:]):
        pair_counts[(i, j)] += 1
    return pair_counts


def pre_tokenization(args: tuple):
    input_path, special_tokens, start, end = args
    word_counts = defaultdict(int)
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Remove special tokens before pre-tokenization.
        if special_tokens:
            corpora = re.split("|".join(map(re.escape, special_tokens)), chunk)
        else:
            corpora = [chunk]
        # Pre-tokenization and count frequency.
        for corpus in corpora:
            for match in re.finditer(COMPILED_PAT, corpus):
                word_counts[tuple(bytes([b]) for b in match.group().encode("utf-8"))] += 1
    return word_counts


# Performance records for test_train_bpe_speed
# v1. pre-tokenization: 0.046s merges: 2.212s
# v2. pre-tokenization: 0.104s merges: 1.216s
# v3. pre-tokenization: 0.106s merges: 0.222s
#
# Perforamnce records for all 3 tests (v3):
# test_train_bpe_speed
#   pre-tokenization: 0.103s merges: 0.235s
# test_train_bpe
#   pre-tokenization: 0.044s merges: 0.225s
# test_train_bpe_special_tokens
#   pre-tokenization: 1.898s merges: 0.639s
#
# Performance records for all 3 tests (v4):
# test_train_bpe_speed
#   pre-tokenization: 0.099s merges: 0.222s
# test_train_bpe
#   pre-tokenization: 0.092s merges: 0.229s
# test_train_bpe_special_tokens
#   pre-tokenization: 0.549s merges: 0.694s
#
# Performance records for all 3 tests (v5):
# test_train_bpe_speed
#   pre-tokenization: 0.092s merges: 0.182s
# test_train_bpe
#   pre-tokenization: 0.090s merges: 0.185s
# test_train_bpe_special_tokens
#   pre-tokenization: 0.534s merges: 0.616s
def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes=4,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # Initialize vocabulary with the initial byte vocabulary and special tokens.
    vocab = dict()
    for i in range(256):
        vocab[i] = bytes([i])
    next_v = 256
    for token in special_tokens:
        vocab[next_v] = token.encode("utf-8")
        next_v += 1

    merges = list()
    if next_v >= vocab_size:
        return vocab, merges

    start_time = time.time()
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    # Parallelize pre-tokenization work by sending each start/end pair to a set of processes.
    tasks = [(input_path, special_tokens, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
    word_counts = defaultdict(int)  # pre-tokenized words and counts
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        for chunk_counts in executor.map(pre_tokenization, tasks):
            for word, count in chunk_counts.items():
                word_counts[word] += count

    end_time = time.time()

    # Introducing word id (index), to decouple the word and its token representation.
    # WHY: when we merge a pair of tokens in a word, we only need to update word_tokens,
    # we don't need to update word_freqs and pair_to_words.
    word_tokens = list(list(t) for t in word_counts.keys())  # index -> list(bytes)
    word_freqs = list(word_counts.values())  # index -> int

    # Build pair_counts and pair_to_words stats.
    heap = []  # heap(MaxHeapItem(int, tuple(bytes, bytes)))
    pair_counts = defaultdict(int)  # tuple(bytes, bytes) -> int
    pair_to_words = defaultdict(set[int])  # tuple(bytes, bytes) -> set(int)
    for idx in range(len(word_tokens)):
        for pair in zip(word_tokens[idx][:-1], word_tokens[idx][1:]):
            pair_counts[pair] += word_freqs[idx]
            heapq.heappush(heap, MaxHeapItem(pair_counts[pair], pair))
            pair_to_words[pair].add(idx)

    while next_v < vocab_size and pair_counts:
        # Find the pair to merge.
        pair_to_merge = None
        while heap:
            item = heapq.heappop(heap)
            if item.pair not in pair_counts or item.count != pair_counts[item.pair]:
                continue
            pair_to_merge = item.pair
            break
        if pair_to_merge is None:
            break

        merges.append(pair_to_merge)
        vocab[next_v] = pair_to_merge[0] + pair_to_merge[1]
        next_v += 1

        # For each word contains the merge pair, updates pair_counts and pair_to_words.
        for idx in list(pair_to_words[pair_to_merge]):
            tokens = word_tokens[idx]
            count = word_freqs[idx]
            merged_tokens = merge_pair(tokens, pair_to_merge)
            word_tokens[idx] = merged_tokens

            old_pair_counts = count_pairs(tokens)
            new_pair_counts = count_pairs(merged_tokens)

            for p in old_pair_counts.keys():
                pair_counts[p] += (new_pair_counts[p] - old_pair_counts[p]) * count
                if pair_counts[p] == 0:
                    pair_counts.pop(p)
                else:
                    heapq.heappush(heap, MaxHeapItem(pair_counts[p], p))
                if new_pair_counts[p] == 0:
                    pair_to_words[p].remove(idx)
                    if len(pair_to_words[p]) == 0:
                        pair_to_words.pop(p)

            for p in new_pair_counts.keys():
                if old_pair_counts[(p)] == 0:
                    pair_counts[p] += new_pair_counts[p] * count
                    heapq.heappush(heap, MaxHeapItem(pair_counts[p], p))
                    pair_to_words[p].add(idx)

    end_time2 = time.time()
    print(f"Perf stats - pre-tokenization: {end_time - start_time:.3f}s merges: {end_time2 - end_time:.3f}s")

    return vocab, merges


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="tokenizer", description="A BPE tokenizer")
    parser.add_argument(
        "--input-path",
        default="data/TinyStoriesV2-GPT4-train.txt",
        help="Path to a text file with BPE tokenizer training data.",
    )
    parser.add_argument("--output-path", default="output/", help="Output path to store vocabulary and merges.")
    parser.add_argument(
        "--vocab-size",
        default=10000,
        type=int,
        help="A positive integer that defines the maximum final vocabulary size.",
    )
    parser.add_argument(
        "--special-tokens", nargs="*", default=["<|endoftext|>"], help="A list of strings to add to the vocabulary."
    )
    parser.add_argument("--num-processes", default=4, type=int, help="Number of processes used in pre-tokenization.")
    args = parser.parse_args()
    dataset_name = re.split("[-_]", pathlib.Path(args.input_path).stem)[0]
    print(f"Train BPE tokenizer with {dataset_name}: {args}")
    vocab, merges = train_bpe(args.input_path, args.vocab_size, args.special_tokens, args.num_processes)
    print(f"Train complete - vocabulary size: {len(vocab)}, longest token: {max(vocab.values(), key=len)}")
    print(f"Saving vocab and merges to {args.output_path} ({dataset_name}-vocab.pkl and {dataset_name}-merges.pkl)")
    with open(args.output_path + dataset_name + "-vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open(args.output_path + dataset_name + "-merges.pkl", "wb") as f:
        pickle.dump(merges, f)
    print("All done.")
