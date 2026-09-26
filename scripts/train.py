import argparse
import json
import math
import os
import time

import numpy as np
import torch

from cs336_basics.adamw import AdamW
from cs336_basics.checkpointing import save_checkpoint
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.data_loading import get_batch
from cs336_basics.get_lr_cosine_schedule import get_lr_cosine_schedule
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.transformer_lm import TransformerLM


def run_validation(model, dataset, args):
    model.eval()
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        while num_batches < args.validation_iters:
            inputs, targets = get_batch(
                dataset=dataset,
                batch_size=args.validation_batch_size,
                context_length=args.context_length,
                device=args.device_type,
            )
            loss = cross_entropy(model(inputs), targets)
            total_loss += loss.item()
            num_batches += 1
    model.train()
    return total_loss / num_batches


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="train", description="A transformer LM training loop")
    # Training loop arguments.
    parser.add_argument("--training-iters", default=10000, type=int, help="The maximum number of iterations to train.")
    # TransformerLM arguments.
    parser.add_argument("--vocab-size", default=10000, type=int, help="The size of the vocabulary.")
    parser.add_argument("--context-length", default=256, type=int, help="The maximum context length.")
    parser.add_argument("--d-model", default=512, type=int, help="Dimensionality of the Transformer block inputs.")
    parser.add_argument("--num-layers", default=4, type=int, help="The number of Transformer blocks to use.")
    parser.add_argument(
        "--num-heads", default=16, type=int, help="Number of heads to use in multi-head self-attention."
    )
    parser.add_argument(
        "--d-ff", default=1344, type=int, help="Dimensionality of the position-wise feed-forward inner layer."
    )
    parser.add_argument("--rope-theta", default=10000.0, type=float, help="Theta value for the RoPE.")
    # Optimizer arguments.
    parser.add_argument(
        "--adamw-beta1", default=0.9, type=float, help="Beta value to update the first moment esimates."
    )
    parser.add_argument(
        "--adamw-beta2", default=0.999, type=float, help="Beta value to update the second moment estimates."
    )
    parser.add_argument("--adamw-weight-decay", default=1e-2, type=float, help="The weight decay rate.")
    parser.add_argument(
        "--adamw-eps", default=1e-8, type=float, help="The eps value used to improve numerical stability."
    )
    # LR cosine schedule arguments.
    parser.add_argument(
        "--max-learning-rate",
        default=1e-3,
        type=float,
        help="The maximum learning rate for cosine learning rate schedule (with warmup).",
    )
    parser.add_argument(
        "--min-learning-rate",
        default=1e-5,
        type=float,
        help="The minimum / final learning rate for the cosine learning rate schedule (with warmup).",
    )
    parser.add_argument(
        "--warmup-iters", default=1000, type=int, help="The number of iterations to linearly warm-up the learning rate."
    )
    parser.add_argument(
        "--cosine-cycle-iters", default=None, type=int, help="The number of cosine annealing iterations."
    )
    # Gradient clipping arguments.
    parser.add_argument("--max-l2-norm", default=1.0, type=float, help="The maximum l2-norm value.")
    # Data loading arguments.
    parser.add_argument("--batch-size", default=32, type=int, help="The batch size.")
    parser.add_argument(
        "--device-type", default="cuda" if torch.cuda.is_available() else "cpu", type=str, help="PyTorch device string."
    )
    parser.add_argument(
        "--training-dataset-path", default="data/tinystories_train.bin", type=str, help="Path to training dataset."
    )
    # Checkpointing arguments.
    parser.add_argument(
        "--checkpoint-path",
        default="ckpt/training-ckpt",
        type=str,
        help="Path to serialize the model, optimizer, and iteration to.",
    )
    parser.add_argument(
        "--checkpoint-interval", default=1000, type=int, help="The number of iterations between checkpoints."
    )
    parser.add_argument(
        "--log-interval", default=20, type=int, help="The number of iterations between logging the loss."
    )
    parser.add_argument("--log-file", default="logs/tinystories.jsonl", type=str, help="Path to save training logs.")
    # Validation arguments.
    parser.add_argument(
        "--validation-dataset-path", default="data/tinystories_valid.bin", type=str, help="Path to validation dataset."
    )
    parser.add_argument(
        "--validation-interval", default=500, type=int, help="The number of iterations between validation runs."
    )
    parser.add_argument("--validation-batch-size", default=20, type=int, help="The batch size for validation.")
    parser.add_argument(
        "--validation-iters", default=10, type=int, help="The number of iterations to run validation for."
    )

    args = parser.parse_args()
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(args.device_type)
    opt = AdamW(
        params=model.parameters(),
        lr=get_lr_cosine_schedule(
            it=0,
            max_learning_rate=args.max_learning_rate,
            min_learning_rate=args.min_learning_rate,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.training_iters if args.cosine_cycle_iters is None else args.cosine_cycle_iters,
        ),
        betas=(args.adamw_beta1, args.adamw_beta2),
        weight_decay=args.adamw_weight_decay,
        eps=args.adamw_eps,
    )
    training_dataset = np.memmap(args.training_dataset_path, dtype=np.uint16, mode="r")
    validation_dataset = np.memmap(args.validation_dataset_path, dtype=np.uint16, mode="r")

    if args.log_file:
        dir_name = os.path.dirname(args.log_file)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
    if args.checkpoint_path:
        dir_name = os.path.dirname(args.checkpoint_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
    start_time = time.time()

    for it in range(args.training_iters):
        opt.zero_grad()
        # Update the learning rate according to the cosine schedule.
        lr = get_lr_cosine_schedule(
            it=it,
            max_learning_rate=args.max_learning_rate,
            min_learning_rate=args.min_learning_rate,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.training_iters if args.cosine_cycle_iters is None else args.cosine_cycle_iters,
        )
        for group in opt.param_groups:
            group["lr"] = lr
        # inputs: (batch_size, context_length), targets: (batch_size, context_length)
        inputs, targets = get_batch(
            dataset=training_dataset,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=args.device_type,
        )
        # model(inputs): (batch_size, context_length, vocab_size)
        # targets: (batch_size, context_length)
        # loss: (1, )
        loss = cross_entropy(model(inputs), targets)
        if (it + 1) % args.log_interval == 0:
            wallclock_time = time.time() - start_time
            tokens = (it + 1) * args.batch_size * args.context_length
            log_entry = {
                "step": it,
                "wallclock_time": wallclock_time,
                "tokens": tokens,
                "train_loss": loss.item(),
                "lr": lr,
            }
            print(
                f"[TRAIN] Step: {it:>6} | Time: {wallclock_time:<10.2f} | Tokens: {tokens:<12,d} | Loss: {loss.item():<10.4f} | LR: {lr:<10.4e}"
            )
            if args.log_file:
                with open(args.log_file, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

        loss.backward()
        gradient_clipping(model.parameters(), args.max_l2_norm)
        opt.step()
        # Save a checkpoint every `checkpoint_interval` iterations.
        if (it + 1) % args.checkpoint_interval == 0:
            save_checkpoint(model=model, optimizer=opt, iteration=it, out=args.checkpoint_path + f".{it}.pt")
        # Run validation.
        if (it + 1) % args.validation_interval == 0:
            val_loss = run_validation(model, validation_dataset, args)
            wallclock_time = time.time() - start_time
            tokens = (it + 1) * args.batch_size * args.context_length
            val_perplexity = math.exp(val_loss)
            log_entry = {
                "step": it,
                "wallclock_time": wallclock_time,
                "tokens": tokens,
                "val_loss": val_loss,
                "val_perplexity": round(val_perplexity, 4),
            }
            print(
                f"[VALID] Step: {it:>6} | Time: {wallclock_time:<10.2f} | Tokens: {tokens:<12,d} | Loss: {val_loss:<10.4f} | PPL: {val_perplexity:<10.4f}"
            )
            if args.log_file:
                with open(args.log_file, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

    # Save the final checkpoint after training is complete.
    save_checkpoint(
        model=model,
        optimizer=opt,
        iteration=args.training_iters - 1,
        out=args.checkpoint_path + f".{args.training_iters - 1}.pt",
    )
