import os
# import time
import regex as re
from collections import defaultdict
from cs336_basics.pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


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


# Performance Records
# 1. pre-tokenization: 0.046, merges: 2.212
# 2. pre-tokenization: 0.104, merges: 1.216
# 3. pre-tokenization: 0.106, merges: 0.222
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

    # start_time = time.time()
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        word_counts = defaultdict(int)  # pre-tokenized words and counts
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Remove special tokens before pre-tokenization.
            corpora = re.split("|".join(map(re.escape, special_tokens)), chunk)
            # Pre-tokenization and count frequency.
            for corpus in corpora:
                for match in re.finditer(PAT, corpus):
                    word_counts[tuple(bytes([b]) for b in match.group().encode("utf-8"))] += 1
    # end_time = time.time()
    # print("pre-tokenization took ", end_time - start_time)

    # Build pair_counts and pair_to_words stats.
    pair_counts = defaultdict(int)  # byte pairs and counts.
    pair_to_words = defaultdict(set[tuple[bytes, ...]])  ## byte pair to word
    for word, count in word_counts.items():
        for i, j in zip(word[:-1], word[1:]):
            pair_counts[(i, j)] += count
            pair_to_words[(i, j)].add(word)

    while next_v < vocab_size:
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

    # end_time2 = time.time()
    # print("merges took ", end_time2 - end_time)

    return vocab, merges
