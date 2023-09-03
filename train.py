import argparse
import torch
import os
import datetime

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
parser.add_argument('--lr_rate', default=1e-3, help='Learning rate for optimizer.', type=float)
parser.add_argument('--num_iters', default=40000, help='Number of training iterations.', type=int)
parser.add_argument('--eval_interval', default=100, help='Evaluate every certain number of training iterations', type=int)
parser.add_argument('--eval_iters', default=10, help='Number of iterations in evaluation cycle.', type=int)
parser.add_argument('--training_set_ratio', default=0.9, help='Fraction of dataset that is used for training.', type=float)

args = parser.parse_args()

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
    lr_rate = args.lr_rate
    max_iters = args.num_iters
    eval_interval = args.eval_interval
    training_set_ratio = args.training_set_ratio
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    torch.manual_seed(1337)

    if session_name is None:
        session_name = 'experiment-' + datetime.datetime.now().strftime('%Y%m%d%H%M')
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
    # print(model)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    # Visualize the model
    xb, yb = sanguo_data.get_batch('train', batch_size=batch_size, device=device, random=True)
    writer.add_graph(model, (xb, yb))
    writer.flush()

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_rate)
    optimizer.zero_grad(set_to_none=True)

    # Training loop
    for iter in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if (iter % eval_interval) == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, sanguo_data, device)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            # Log the estimated training loss and validation loss
            writer.add_scalars(
                # main_tag
                'Estimated Training vs. Validation Loss',
                # tag_scalar
                {
                    'Training': losses['train'],
                    'Validation': losses['val'],
                },
                # global_step
                iter)

        # sample a batch of data
        xb, yb = sanguo_data.get_batch('train', batch_size=batch_size, device=device)

        # evaluate the loss
        logits, loss = model(xb, yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # if (iter % eval_interval) == 0 or iter == max_iters - 1:
        #     loss_val = loss.item()
        #     print(f"iteration: {iter:>6d}, loss: {loss_val:>7f}")
        
    print(f"Total number of tokens trained: {block_size*batch_size*max_iters}")
    print("Finished training")
    writer.flush()

    # Save the model checkpoint
    if not args.no_save_model:
        print(f"Saving model checkpoint to {args.output}")
        torch.save(model, args.output)


def main():
    # Print the arguments
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    
    print("============================================")
    print("Start training")
    print("============================================")
    train()

if __name__ == "__main__":
    main()