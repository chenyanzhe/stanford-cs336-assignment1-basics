from collections import defaultdict
from heapq import merge
import pickle
import regex as re
from collections.abc import Iterable, Iterator

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
COMPILED_PAT = re.compile(PAT)


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


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.reverse_vocab = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens = set(special_tokens) if special_tokens is not None else set()

        # Add special tokens to vocab and reverse_vocab if they are not already present
        next_v = len(vocab)
        for token in (token.encode("utf-8") for token in self.special_tokens):
            if token not in self.reverse_vocab:
                self.reverse_vocab[token] = next_v
                self.vocab[next_v] = token
                next_v += 1

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def split_by_special_tokens(self, text: str) -> list[str]:
        if not self.special_tokens:
            return [text]

        # Filter out empty strings from the split result to avoid adding them to the token list
        return [
            token
            for token in re.split(
                "(" + "|".join(map(re.escape, sorted(self.special_tokens, key=len, reverse=True))) + ")", text
            )
            if token
        ]

    def encode(self, text: str) -> list[int]:
        if not text:
            return []

        pre_tokens = self.split_by_special_tokens(text)
        words = []
        word_to_id = {}  # dict[bytes, int]
        word_tokens = []  # list(list[bytes])
        pair_to_words = defaultdict(set)  # dict[tuple[bytes, bytes], set[int]]
        next_word_id = 0
        for token in pre_tokens:
            if token in self.special_tokens:
                words.append(token.encode("utf-8"))
            else:
                split_workds = [match.group().encode("utf-8") for match in re.finditer(COMPILED_PAT, token)]
                words.extend(split_workds)
                for word in split_workds:
                    if word in word_to_id:
                        continue
                    word_to_id[word] = next_word_id
                    word_tokens.append(list(bytes([b]) for b in word))
                    for pair in zip(word_tokens[-1][:-1], word_tokens[-1][1:]):
                        pair_to_words[pair].add(next_word_id)
                    next_word_id += 1

        for pair_to_merge in self.merges:
            for idx in list(pair_to_words[pair_to_merge]):
                tokens = word_tokens[idx]
                merged_tokens = merge_pair(tokens, pair_to_merge)
                word_tokens[idx] = merged_tokens

                old_pairs = set(zip(tokens[:-1], tokens[1:]))
                new_pairs = set(zip(merged_tokens[:-1], merged_tokens[1:]))

                for p in old_pairs:
                    if p not in new_pairs:
                        pair_to_words[p].discard(idx)

                for p in new_pairs:
                    if p not in old_pairs:
                        pair_to_words[p].add(idx)

        result = []
        for word in words:
            if word not in word_to_id:
                result.append(self.reverse_vocab[word])
                continue
            for token in word_tokens[word_to_id[word]]:
                result.append(self.reverse_vocab[token])

        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        return b"".join([self.vocab[id] for id in ids]).decode("utf-8", errors="replace")
