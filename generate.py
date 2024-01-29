import argparse
import torch
import time
import streamlit as st

from model import SanGuoGPTModel
from sanguo_data import encoder, decoder, load_token_map

parser = argparse.ArgumentParser()
# DEPRECATED in v0.3: old checkpoint format no longer works.
# parser.add_argument('-m', '--model', default='checkpoints/sanguogpt-v0.2.pth', help='Model checkpoint.', type=str)
parser.add_argument('--resume_from', default='checkpoints/sanguogpt-v0.3.pt', help='[v0.3+] Model checkpoint to resume from.', type=str)
parser.add_argument('-l', '--gen_length', default=100, help='Maximum length to generate.', type=int)
parser.add_argument('--c2i', default='c2i.json', help='Token map file (character to index).', type=str)
parser.add_argument('--i2c', default='i2c.json', help='Token map file (index to character).', type=str)
parser.add_argument('--prompt', default=' ', help='Prompt for text generation.', type=str)
parser.add_argument('--webui', action='store_true', help='If specified, use streamlit-based Web UI instead of command line.')
parser.add_argument('--compile', action='store_true', help='Compile the model (requires PyTorch 2.x and may not work on Mac).')

args = parser.parse_args()

print(f"Loading model from {args.resume_from}")
checkpoint = torch.load(args.resume_from)
model_args = checkpoint['model_args']
device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

model = SanGuoGPTModel(vocab_size=model_args['vocab_size'],
                        d_model=model_args['d_model'],
                        n_layer=model_args['n_layer'],
                        dropout=model_args['dropout'],
                        block_size=model_args['block_size'],
                        n_head=model_args['n_head'],
                        device=device
                        )
model = model.to(device)
if args.compile:
    model = torch.compile(model) # requires PyTorch 2.0
model.load_state_dict(checkpoint['model'])

c2i, i2c = load_token_map(c2i_file=args.c2i, i2c_file=args.i2c)
print(f"Loading token map file from {args.c2i} and {args.i2c}")
print(f"Using {device} device")

# Test with zero input (according to i2c.json, 0 means '\n')
# context = torch.zeros((1, 1), dtype=torch.long, device=device)
# print(decoder(model.generate(context, max_new_tokens=args.gen_length)[0].tolist(), i2c))

def gen_response(prompt:str) -> str:
    start = time.time()
    # encode the prompt into a tensor, and reshape it into (1, T)
    # first dimension is the batch, which is expected by the forward method.
    context = torch.tensor(encoder(prompt, c2i), device=device).unsqueeze(0)
    # model.generate() will truncate the prompt if it's too long, no need to worry about this.
    resp_idx, ppl = model.generate(context, max_new_tokens=args.gen_length) 
    resp = decoder(resp_idx[0].tolist(), i2c)
    end = time.time()
    tokens_generated = min(args.gen_length, len(resp) - len(prompt))
    print(f"{tokens_generated} tokens generated in {end-start:>.3f} seconds, avg {tokens_generated/(end-start):>.3f} tokens/sec.")
    print(f"Perplexity of generation: {ppl[0].item():>.4f}")
    return resp

def webui():
    st.set_page_config(page_title='三国GPT SanGuo GPT')
    st.title('三国GPT SanGuo GPT v0.3')
    with st.form('Generate SanGuo'):
        text = st.text_area('Enter prompt:', args.prompt)
        submitted = st.form_submit_button('Submit')
        if submitted:
            st.info(gen_response(text))

if __name__ == "__main__":
    if args.webui:
        webui()
    else:
        print(gen_response(args.prompt))
