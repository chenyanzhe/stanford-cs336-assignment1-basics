import argparse

import numpy as np

from cs336_basics.adamw import AdamW
from cs336_basics.checkpointing import save_checkpoint
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.data_loading import get_batch
from cs336_basics.get_lr_cosine_schedule import get_lr_cosine_schedule
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.transformer_lm import TransformerLM

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="training_loop", description="A transformer LM training loop")
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
        "--cosine-cycle-iters", default=10000, type=int, help="The number of cosine annealing iterations."
    )
    # Gradient clipping arguments.
    parser.add_argument("--max-l2-norm", default=1.0, type=float, help="The maximum l2-norm value.")
    # Data loading arguments.
    parser.add_argument("--batch-size", default=32, type=int, help="The batch size.")
    parser.add_argument("--device-type", default="cuda", type=str, help="PyTorch device string.")
    parser.add_argument("--training-dataset-path", type=str, help="Path to training dataset.")
    # Checkpointing arguments.
    parser.add_argument(
        "--checkpoint-path",
        default="training-ckpt",
        type=str,
        help="Path to serialize the model, optimizer, and iteration to.",
    )
    parser.add_argument(
        "--checkpoint-interval", default=1000, type=int, help="The number of iterations between checkpoints."
    )
    parser.add_argument(
        "--log-interval", default=20, type=int, help="The number of iterations between logging the loss."
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
            cosine_cycle_iters=args.cosine_cycle_iters,
        ),
        betas=(args.adamw_beta1, args.adamw_beta2),
        weight_decay=args.adamw_weight_decay,
        eps=args.adamw_eps,
    )
    dataset = np.memmap(args.training_dataset_path, dtype=np.uint16, mode="r")
    for it in range(args.training_iters):
        opt.zero_grad()
        # Update the learning rate according to the cosine schedule.
        lr = get_lr_cosine_schedule(
            it=it,
            max_learning_rate=args.max_learning_rate,
            min_learning_rate=args.min_learning_rate,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.cosine_cycle_iters,
        )
        for group in opt.param_groups:
            group["lr"] = lr
        # inputs: (batch_size, context_length), targets: (batch_size, context_length)
        inputs, targets = get_batch(
            dataset=dataset, batch_size=args.batch_size, context_length=args.context_length, device=args.device_type
        )
        # model(inputs): (batch_size, context_length, vocab_size)
        # targets: (batch_size, context_length)
        # loss: (1, )
        loss = cross_entropy(model(inputs), targets)
        if (it + 1) % args.log_interval == 0:
            print(f"Iteration {it}, Loss: {loss.cpu().item()}")
        loss.backward()
        gradient_clipping(model.parameters(), args.max_l2_norm)
        opt.step()
        # Save a checkpoint every `checkpoint_interval` iterations.
        if (it + 1) % args.checkpoint_interval == 0:
            save_checkpoint(model=model, optimizer=opt, iteration=it, out=args.checkpoint_path + f".{it}.pt")

    # Save the final checkpoint after training is complete.
    save_checkpoint(
        model=model,
        optimizer=opt,
        iteration=args.training_iters - 1,
        out=args.checkpoint_path + f".{args.training_iters - 1}.pt",
    )
