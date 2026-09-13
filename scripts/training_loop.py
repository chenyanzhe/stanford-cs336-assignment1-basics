import argparse

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
    parser.add_argument("--traning-iters", type=int, help="The maximum number of iterations to train.")
    # TransformerLM arguments.
    parser.add_argument("--vocab-size", default=10000, type=int, help="The size of the vocabulary.")
    parser.add_argument("--context_length", default=1024, type=int, help="The maximum context length.")
    parser.add_argument("--d-model", default=1600, type=int, help="Dimensionality of the Transformer block inputs.")
    parser.add_argument("--num-layers", default=48, type=int, help="The number of Transformer blocks to use.")
    parser.add_argument(
        "--num-heads", default=25, type=int, help="Number of heads to use in multi-head self-attention."
    )
    parser.add_argument(
        "--d-ff", default=4288, type=int, help="Dimensionality of the position-wise feed-forward inner layer."
    )
    parser.add_argument("--rope-theta", type=float, help="Theta value for the RoPE.")
    # Optimizer arguments.
    parser.add_argument("--adamw-beta1", type=float, help="Beta value to update the first moment esimates.")
    parser.add_argument("--adamw-beta2", type=float, help="Beta value to update the second moment estimates.")
    parser.add_argument("--adamw-weight-decay", type=float, help="The weight decay rate.")
    parser.add_argument("--adamw-eps", type=float, help="The eps value used to improve numerical stability.")
    # LR cosine schedule arguments.
    parser.add_argument(
        "--max-learning-rate",
        type=float,
        help="The maximum learning rate for cosine learning rate schedule (with warmup).",
    )
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        help="The minimum / final learning rate for the cosine learning rate schedule (with warmup).",
    )
    parser.add_argument(
        "--warmup-iters", type=int, help="The number of iterations to linearly warm-up the learning rate."
    )
    parser.add_argument("--cosine-cycle-iters", type=int, help="The number of cosine annealing iterations.")
    # Gradient clipping arguments.
    parser.add_argument("--max-l2-norm", type=float, help="The maximum l2-norm value.")
    # Data loading arguments.
    parser.add_argument("--batch-size", type=int, help="The batch size.")
    parser.add_argument("--device-type", type=str, help="PyTorch device string.")
    parser.add_argument("--training-dataset-path", type=str, help="Path to training dataset.")
    # Checkpointing arguments.
    parser.add_argument("--checkpoint-path", type=str, help="Path to serialize the model, optimizer, and iteration to.")

    args = parser.parse_args()
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )
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
    # TODO: load training dataset.
    dataset = None
    for it in range(args.traning_iters):
        opt.zero_grad()
        inputs, targets = get_batch(
            dataset=dataset, batch_size=args.batch_size, context_length=args.context_length, device=args.device_type
        )
        loss = cross_entropy(model(inputs), targets)
        print(loss.cpu().item())
        loss.backward()
        gradient_clipping(opt.parameters(), args.max_l2_norm)
        opt.step()
        save_checkpoint(model=model, optimizer=opt, iteration=it, out=args.checkpoint_path)
