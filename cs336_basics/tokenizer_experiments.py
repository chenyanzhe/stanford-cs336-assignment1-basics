import random
import timeit

from cs336_basics.tokenizer import Tokenizer


def sample_documents(num_docs: int, file_path: str) -> list[bytes]:
    with open(file_path, "rb") as f:
        documents = f.read().split(b"<|endoftext|>")
    if num_docs > len(documents):
        raise ValueError(f"Requested {num_docs} documents, but only {len(documents)} available.")
    sampled_docs = random.sample(documents, num_docs)
    return sampled_docs


def calculate_compression_ratio(docs: list[bytes], tokenizer: Tokenizer) -> float:
    original_size = sum(len(doc) for doc in docs)
    compressed_size = sum(len(tokenizer.encode(doc.decode("utf-8"))) for doc in docs)
    return compressed_size / original_size if original_size > 0 else 0.0


def evaluate_compression_ratio(
    docs_path: str,
    vocab_path: str,
    merges_path: str,
    num_docs: int = 10,
) -> float:
    sampled_docs = sample_documents(num_docs, docs_path)
    tokenizer = Tokenizer.from_files(vocab_path, merges_path)
    return calculate_compression_ratio(sampled_docs, tokenizer)


def tokenizer_experiment_a():
    tiny_stories_compression_ratio = (
        sum(
            evaluate_compression_ratio(
                "data/TinyStoriesV2-GPT4-valid.txt",
                "output/TinyStoriesV2-vocab.pkl",
                "output/TinyStoriesV2-merges.pkl",
            )
            for _ in range(10)
        )
        / 10
    )

    owt_compression_ratio = (
        sum(
            evaluate_compression_ratio(
                "data/owt_valid.txt",
                "output/owt-vocab.pkl",
                "output/owt-merges.pkl",
            )
            for _ in range(10)
        )
        / 10
    )

    print("tokenizer_experiment_a results:")
    print(f"\tTinyStories compression ratio: {tiny_stories_compression_ratio:.4f}")
    print(f"\tOpenWebText compression ratio: {owt_compression_ratio:.4f}")


def tokenizer_experiment_b():
    compression_ratio = (
        sum(
            # Tokenize OpenWebText sample with the TinyStories tokenizer.
            evaluate_compression_ratio(
                "data/owt_valid.txt",
                "output/TinyStoriesV2-vocab.pkl",
                "output/TinyStoriesV2-merges.pkl",
            )
            for _ in range(10)
        )
        / 10
    )

    print("tokenizer_experiment_b results:")
    print(f"\tCompression ratio: {compression_ratio:.4f}")


def evaluate_throughput(
    docs_path: str,
    vocab_path: str,
    merges_path: str,
) -> float:
    with open(docs_path, "rb") as f:
        docs = f.read()
    total_bytes = len(docs)
    docs_str = docs.decode("utf-8")

    tokenizer = Tokenizer.from_files(vocab_path, merges_path)
    average_time = timeit.timeit(lambda: tokenizer.encode(docs_str), number=10) / 10

    return total_bytes / average_time if average_time > 0 else 0.0


def tokenizer_experiment_c():
    print("tokenizer_experiment_c results:")
    tiny_stories_throughput = evaluate_throughput(
        "data/TinyStoriesV2-GPT4-valid.txt",
        "output/TinyStoriesV2-vocab.pkl",
        "output/TinyStoriesV2-merges.pkl",
    )
    print(f"\tTinyStories throughput: {tiny_stories_throughput:.2f} bytes/second")
    owt_throughput = evaluate_throughput(
        "data/owt_valid.txt",
        "output/owt-vocab.pkl",
        "output/owt-merges.pkl",
    )
    print(f"\tOpenWebText throughput: {owt_throughput:.2f} bytes/second")


if __name__ == "__main__":
    tokenizer_experiment_a()
    tokenizer_experiment_b()
    tokenizer_experiment_c()
