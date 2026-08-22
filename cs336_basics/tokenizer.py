import heapq
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
        self.merge_to_index = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = set(special_tokens) if special_tokens is not None else set()
        self.word_encode_cache = {}

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

    def encode_word(self, word: bytes) -> list[int]:
        if word in self.word_encode_cache:
            return self.word_encode_cache[word]

        heap = []
        tokens = [bytes([b]) for b in word]
        for pair in zip(tokens[:-1], tokens[1:]):
            if pair in self.merge_to_index:
                heapq.heappush(heap, (self.merge_to_index[pair], pair))

        while heap:
            _, pair = heapq.heappop(heap)
            tokens = merge_pair(tokens, pair)

            # Update the heap with new pairs formed after the merge
            new_heap = []
            for new_pair in zip(tokens[:-1], tokens[1:]):
                if new_pair in self.merge_to_index:
                    heapq.heappush(new_heap, (self.merge_to_index[new_pair], new_pair))
            heap = new_heap

        result = [self.reverse_vocab[t] for t in tokens]
        self.word_encode_cache[word] = result

        return result

    def encode(self, text: str) -> list[int]:
        if not text:
            return []

        pre_tokens = self.split_by_special_tokens(text)
        result = []
        for token in pre_tokens:
            if token in self.special_tokens:
                result.append(self.reverse_vocab[token.encode("utf-8")])
            else:
                for match in re.finditer(COMPILED_PAT, token):
                    result.extend(self.encode_word(match.group().encode("utf-8")))

        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        return b"".join([self.vocab[id] for id in ids]).decode("utf-8", errors="replace")
