import os
import time
import regex as re
from collections import defaultdict
from cs336_basics.pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# Performance Records
# 1. pre-tokenization: 0.046, merges: 2.212
# 2. pre-tokenization: 0.104, merges: 1.216
def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
  # start_time = time.time()
  with open(input_path, "rb") as f:
    num_processes = 4
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>") 

    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    f_table = defaultdict(int)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Remove special tokens before pre-tokenization.
        corpora = re.split("|".join(map(re.escape, special_tokens)), chunk)
        # Pre-tokenization and count frequency.
        for corpus in corpora:
           for match in re.finditer(PAT, corpus):
              f_table[tuple(bytes([b]) for b in match.group().encode("utf-8"))] += 1
  # end_time = time.time()
  # print("pre-tokenization took ", end_time - start_time)

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

  # Build pair_counts stats.
  pair_counts = defaultdict(int)
  for tokens, count in f_table.items():
     for a, b in zip(tokens[:-1], tokens[1:]):
        pair_counts[(a, b)] += count
  
  while next_v < vocab_size:
     # Find the pair to merge: (a, b).
     a, b = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
     merges.append((a, b))
     vocab[next_v] = a + b
     next_v += 1
     # Merging a and b invalidates the (a, b) pair.
     del pair_counts[(a, b)]
     for tokens, count in list(f_table.items()):
        merged_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                # Merging a and b in [prev_a, a, b, next_b] will:
                # - invalidate (prev_a, a) and (b, next_b) pairs.
                # - create (prev_a, a + b) and (a + b, next_b) pairs.
                merged_tokens.append(a + b)
                if i > 0:
                   pair_counts[(tokens[i - 1], a)] -= count
                   pair_counts[(tokens[i - 1], a + b)] += count
                if i + 2 < len(tokens):
                   pair_counts[(b, tokens[i + 2])] -= count
                   pair_counts[(a + b, tokens[i + 2])] += count
                i += 2
            else:
                merged_tokens.append(tokens[i])
                i += 1
        f_table[tuple(merged_tokens)] = f_table.pop(tokens)

  # end_time2 = time.time()
  # print("merges took ", end_time2 - end_time)

  return vocab, merges
