import os
import time
import regex as re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from cs336_basics.pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
COMPILED_PAT = re.compile(PAT)


def merge_pair(tokens: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
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

    return tuple(result)


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
def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
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
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    # Parallelize pre-tokenization work by sending each start/end pair to a set of processes.
    tasks = [(input_path, special_tokens, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
    word_counts = defaultdict(int)  # pre-tokenized words and counts
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        for chunk_counts in executor.map(pre_tokenization, tasks):
            for word, count in chunk_counts.items():
                word_counts[word] += count

    end_time = time.time()

    # Build pair_counts and pair_to_words stats.
    pair_counts = defaultdict(int)  # byte pairs and counts.
    pair_to_words = defaultdict(set[tuple[bytes, ...]])  ## byte pair to word
    for word, count in word_counts.items():
        for i, j in zip(word[:-1], word[1:]):
            pair_counts[(i, j)] += count
            pair_to_words[(i, j)].add(word)

    while next_v < vocab_size and pair_counts:
        # Merge the pair (a, b) that occurs the most.
        a, b = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        merges.append((a, b))
        vocab[next_v] = a + b
        next_v += 1

        # For each word contains the merge pair (a, b), updates word_counts, pair_counts and pair_to_words.
        for word in list(pair_to_words[(a, b)]):
            count = word_counts[word]
            merged_word = merge_pair(word, (a, b))
            word_counts[merged_word] = word_counts.pop(word)

            for i, j in zip(merged_word[:-1], merged_word[1:]):
                pair_counts[(i, j)] += count
                pair_to_words[(i, j)].add(merged_word)

            for i, j in zip(word[:-1], word[1:]):
                pair_counts[(i, j)] -= count
                if pair_counts[(i, j)] == 0:
                    del pair_counts[(i, j)]
                if word in pair_to_words[(i, j)]:
                    pair_to_words[(i, j)].remove(word)
                if len(pair_to_words[(i, j)]) == 0:
                    del pair_to_words[(i, j)]

    end_time2 = time.time()
    print(f"pre-tokenization: {end_time - start_time:.3f}s merges: {end_time2 - end_time:.3f}s")

    return vocab, merges
