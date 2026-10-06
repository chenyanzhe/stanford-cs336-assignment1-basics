import argparse

import torch

from cs336_basics.decoding import generate
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer_lm import TransformerLM

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate", description="A script to generate tokens from a trained model checkpoint."
    )
    # File & Resource Paths
    parser.add_argument(
        "--checkpoint-path",
        default="ckpt/tinystories_5k.4999.pt",
        type=str,
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--vocab-path",
        default="data/TinyStoriesV2-vocab.pkl",
        type=str,
        help="Path to the vocabulary file.",
    )
    parser.add_argument(
        "--merges-path",
        default="data/TinyStoriesV2-merges.pkl",
        type=str,
        help="Path to the merges file.",
    )
    parser.add_argument(
        "--special-tokens", nargs="*", default=["<|endoftext|>"], help="A list of strings to add to the vocabulary."
    )
    parser.add_argument(
        "--eos-token",
        default="<|endoftext|>",
        type=str,
        help="End-of-sequence token.",
    )
    # Model Architecture Config
    parser.add_argument(
        "--vocab-size",
        default=10000,
        type=int,
        help="Size of the vocabulary.",
    )
    parser.add_argument(
        "--context-length",
        default=256,
        type=int,
        help="Maximum context length for the model.",
    )
    parser.add_argument(
        "--d-model",
        default=512,
        type=int,
        help="Dimension of the model.",
    )
    parser.add_argument(
        "--num-layers",
        default=4,
        type=int,
        help="Number of layers in the model.",
    )
    parser.add_argument(
        "--num-heads",
        default=16,
        type=int,
        help="Number of attention heads in the model.",
    )
    parser.add_argument(
        "--d-ff",
        default=1344,
        type=int,
        help="Dimension of the feed-forward network in the model.",
    )
    parser.add_argument(
        "--rope-theta",
        default=10000.0,
        type=float,
        help="Theta value for the RoPE (Rotary Positional Embedding).",
    )
    parser.add_argument(
        "--device-type",
        default="cuda" if torch.cuda.is_available() else "cpu",
        type=str,
        help="Device to run the model on (e.g., 'cuda' or 'cpu').",
    )
    # Sampling Hyperparameters
    parser.add_argument(
        "--prompt",
        default="",
        type=str,
        help="Initial prompt to start generating tokens.",
    )
    parser.add_argument(
        "--max-new-tokens",
        default=256,
        type=int,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        default=1.0,
        type=float,
        help="Sampling temperature for token generation.",
    )
    parser.add_argument(
        "--top-p",
        default=1.0,
        type=float,
        help="Top-p sampling probability for token generation.",
    )
    args = parser.parse_args()
    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab_path,
        merges_filepath=args.merges_path,
        special_tokens=args.special_tokens,
    )
    eos_token_id = tokenizer.encode(args.eos_token)[0]

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(args.device_type)
    ckpt = torch.load(args.checkpoint_path, map_location=args.device_type)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    prompt = args.eos_token if args.prompt == "" else args.prompt
    input_tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=args.device_type)
    output_tokens = generate(
        model=model,
        prompt_tokens=input_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=eos_token_id,
    )

    generated_text = tokenizer.decode(output_tokens.squeeze(0).tolist())
    print(generated_text)
