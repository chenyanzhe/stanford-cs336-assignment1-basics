import os
import time
import regex as re
from collections import defaultdict
from cs336_basics.pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# Performance Records
# 1. pre-tokenization: 0.046, merges: 2.212
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
  while next_v < vocab_size:
     # Do one merge.
     pair_counts = defaultdict(int)
     for item, count in f_table.items():
        for a, b in zip(item[:-1], item[1:]):
           pair_counts[(a, b)] += count
     merged_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
     merges.append(merged_pair)
     vocab[next_v] = merged_pair[0] + merged_pair[1]
     next_v += 1
     # Merge the pair in f_table.
     updated_f_table = defaultdict(int)
     for item, count in f_table.items():
        updated_f_table[merge_pair(item, merged_pair)] = count
     f_table = updated_f_table

  # end_time2 = time.time()
  # print("merges took ", end_time2 - end_time)

  return vocab, merges
