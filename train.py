import argparse
import math
import torch
import os
import datetime
import time

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter

from model import SanGuoGPTModel
from sanguo_data import SanGuoData

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='sanguo-utf8.txt', help='Input text for training.', type=str)
parser.add_argument('-o', '--output', default='sanguogpt.pth', help='File name of the saved model.', type=str)
parser.add_argument('--no_save_model', action='store_true', help='If specified, do not save the model after training.')
parser.add_argument('-b', '--batch_size', default=32, type=int)
parser.add_argument('-l', '--block_size', default=256, help='Sequence length (block size) in training.', type=int)
parser.add_argument('-d', '--d_model', default=384, help='Dimension of each token\'s representation.', type=int)
parser.add_argument('--num_heads', default=8, help='Number of attention heads.', type=int)
parser.add_argument('--num_layers', default=6, help='Number of Multi-Head Attention + FFN layers (blocks).', type=int)
parser.add_argument('--dropout', default=0.01, type=float)
parser.add_argument('--lr_rate', default=6e-4, help='Max learning rate for optimizer.', type=float)
parser.add_argument('--min_lr', default=6e-5, help='Min learning rate for optimizer.', type=float)
parser.add_argument('--lr_decay_iters', default=60000, help='Number of steps before learning rate is flat at min_lr.', type=int)
parser.add_argument('--warmup_iters', default=1000, help='Number of steps to warm up.', type=int)
parser.add_argument('--decay_lr', action='store_true', help='If specified, enable decaying learning rate.')
parser.add_argument('--num_iters', default=40000, help='Number of training iterations.', type=int)
parser.add_argument('--eval_interval', default=1000, help='Evaluate every certain number of training iterations', type=int)
parser.add_argument('--eval_iters', default=10, help='Number of iterations in evaluation cycle.', type=int)
parser.add_argument('--training_set_ratio', default=1.0, help='Fraction of dataset that is used for training.', type=float)
parser.add_argument('--tensorboard', action='store_true', help='If specified, enable visualization with TensorBoard.')
parser.add_argument('--ckpt_interval', default=10000, help='Save checkpoint every certain number of training iterations', type=int)

args = parser.parse_args()

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return args.lr_rate * it / args.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > args.lr_decay_iters:
        return args.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return args.min_lr + coeff * (args.lr_rate - args.min_lr)

@torch.no_grad()
def estimate_loss(model, data, device):
    out = {}
    eval_iters = args.eval_iters
    batch_size = args.batch_size
    # set the model in evaluation mode
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            # For validation, we randomly pick a few items as a batch.
            X, Y = data.get_batch(split, batch_size=batch_size, device=device, random=True)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    # switch back to training mode
    model.train()
    return out

def train(session_name:str = None):
    # Some hyperparameters
    block_size = args.block_size
    batch_size = args.batch_size
    d_model = args.d_model
    n_head = args.num_heads
    n_layer = args.num_layers
    dropout = args.dropout
    lr_rate = get_lr(0) if args.decay_lr else args.lr_rate
    max_iters = args.num_iters
    eval_interval = args.eval_interval
    training_set_ratio = args.training_set_ratio
    ckpt_interval = args.ckpt_interval
    ckpt_basename = os.path.basename(args.output)
    ckpt_path = os.path.dirname(args.output)
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    torch.manual_seed(1337)

    if session_name is None:
        writer = None
    else:
        writer = SummaryWriter(os.path.join('runs', session_name))

    # Prepare the dataset
    sanguo_data = SanGuoData(source = args.input, block_size = block_size, training_set_ratio = training_set_ratio)
    sanguo_data.ingest()
    print(f"Number of tokens in each batch: {block_size*batch_size}")

    # Create the model
    model = SanGuoGPTModel(vocab_size=sanguo_data.vocab_size,
                        d_model=d_model,
                        n_layer=n_layer,
                        dropout=dropout,
                        block_size=block_size,
                        n_head=n_head,
                        device=device
                        )
    m = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    # Visualize the model
    xb, yb = sanguo_data.get_batch('train', batch_size=batch_size, device=device, random=True)
    if writer is not None:
        writer.add_graph(model, (xb, yb))
        writer.flush()

    # For visualizing the embeddings:
    # encoder returns a list for each string (which is a single character in vocabulary).
    # Therefore, the shape of all_token will be like (vocab_size, 1).
    # We want a 1-D list, so we squeeze the last dimension and change the shape to (vocab_size, ).
    all_tokens = torch.tensor([sanguo_data.encoder(ch) for ch in sanguo_data.chars[20:-7]],
                            dtype=torch.long, device=device, requires_grad=False).squeeze(1)
    print('tokens for visualization', all_tokens.shape)

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_rate)
    optimizer.zero_grad(set_to_none=True)

    training_start = time.time()
    acc_loss = 0.0
    num_steps = 0
    # Training loop
    for iter in range(max_iters):
        # Set learning rate for this iteration
        lr = get_lr(iter) if args.decay_lr else lr_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # every once in a while evaluate the loss on train and val sets
        if (iter > 0 and (iter % eval_interval) == 0) or iter == max_iters - 1:
            avg_loss = acc_loss / num_steps
            ppl_train = math.exp(avg_loss)
            # print(f"num_steps={num_steps}, acc_loss={acc_loss:>.3f}, avg_loss={avg_loss:>.3f}")
            print(f"step {iter}: lr {lr:>.5f} avg_loss {avg_loss:>.3f}, avg_perplexity {ppl_train:>.3f}")
            acc_loss = 0.0
            num_steps = 0
            # Log the estimated training loss and validation loss
            if writer is not None:
                writer.add_scalars(
                    # main_tag
                    'Average Training Loss',
                    # tag_scalar
                    {
                        'Training': avg_loss,
                    },
                    # global_step
                    iter)
                writer.add_scalars(
                    # main_tag
                    'Average Training Perplexity',
                    # tag_scalar
                    {
                        'Training': ppl_train,
                    },
                    # global_step
                    iter)
                writer.add_scalars(
                    'Learning rate',
                    {
                        'learning_rate': lr,
                    }
                )

        # save the checkpoint periodically
        if ckpt_interval > 0 and (iter > 0 and (iter % ckpt_interval) == 0):
            ckpt = os.path.join(ckpt_path, ckpt_basename + f"-{iter}")
            print(f"---> save checkpoint: {ckpt}")
            torch.save(model, ckpt)

            # Save the embeddings with checkpoints.
            if writer is not None:
                # Visualize the embeddings.
                embedding_table = model.get_embeddings(all_tokens)  # (vocab_size, d_model)
                # print(embedding_table.shape)
                writer.add_embedding(embedding_table, metadata=sanguo_data.chars[20:-7], tag=f"embeddings-step{iter}")
                writer.flush()


        # sample a batch of data
        xb, yb = sanguo_data.get_batch('train', batch_size=batch_size, device=device)

        # evaluate the loss
        logits, loss = model(xb, yb)
        acc_loss += loss.item()
        num_steps += 1
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # if (iter % eval_interval) == 0 or iter == max_iters - 1:
        #     loss_val = loss.item()
        #     print(f"iteration: {iter:>6d}, loss: {loss_val:>7f}")

    training_end = time.time()
    print("Finished training")
    print(f"Total number of tokens trained: {block_size*batch_size*max_iters}")
    print(f"Time elapsed: {training_end-training_start:.3f} seconds")
    print(f"Training throughput: {(block_size*batch_size*max_iters)/(training_end-training_start):>.3f} tokens/sec")

    return model


def main():
    # Print the arguments
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    
    print("============================================")
    print("Start training")
    print("============================================")
    if args.tensorboard:
        m = train(session_name='experiment-' + datetime.datetime.now().strftime('%Y%m%d%H%M'))
    else:
        m = train()
    
    # Save the model checkpoint
    if not args.no_save_model:
        print(f"Saving model checkpoint to {args.output}")
        torch.save(m, args.output)

if __name__ == "__main__":
    main()
