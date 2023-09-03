import torch
import json

class SanGuoData:
    def __init__(self, source = 'sanguo-utf8.txt', block_size = 192, training_set_ratio = 0.9):
        self.source = source
        self.block_size = block_size
        self.training_set_ratio = training_set_ratio
        self.text = None
        self.chars = None
        self.vocab_size = 0
        self.c2i = None
        self.i2c = None
        self.encoder = None
        self.decoder = None
        self.data = None
    
    def ingest(self, gen_dataset=True, gen_token_map=True):
        with open(self.source, 'r', encoding='utf-8') as f:
            self.text = f.read()
        print(f"Length of text: {len(self.text)}")   # 606051 Chinese characters
        # print(self.text[:100])
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars) 

        # I don't plan to use a tokenizer for this.
        # IMHO, a Chinese character is not a "letter". Instead it's more like
        # a word or subword. So we should treat each Chinese character as a token.

        # Turn each character into a number (index into the chars array)
        # Map character to index.
        self.c2i = {ch:i for i, ch in enumerate(self.chars)}
        # Map index to character.
        self.i2c = {i:ch for i, ch in enumerate(self.chars)}

        # Given a string (sequence of characters), encode it into a sequence of indices.
        self.encoder = lambda s: [self.c2i[c] for c in s]
        # Given a sequence of indices, decode it back to the string
        self.decoder = lambda l: ''.join([self.i2c[i] for i in l])

        self.data = torch.tensor(self.encoder(self.text), dtype=torch.long)
        # print(self.data.shape, self.data.dtype)

        if gen_token_map:
            self.save_token_map()

        if gen_dataset:
            self.gen_dataset()
    
    def save_token_map(self, c2i_file:str = 'c2i.json', i2c_file:str='i2c.json'):
        with open(c2i_file, 'w', encoding='utf-8') as f:
            json.dump(self.c2i, f)
        with open(i2c_file, 'w', encoding='utf-8') as f:
            json.dump(self.i2c, f)

    def test_enc_dec(self):
        print("Original text:")
        print(self.text[:50])

        print("\nEncoded:")
        print(self.encoder(self.text[:50]))

        print("\nDecoded:")
        print(self.decoder(self.encoder(self.text[:50])))

    def gen_dataset(self):
        # Split up into training and validation sets.
        # Generate a random permutation of the entire dataset.
        # Sequence of length <block_size> is used to predict the next token
        # The last seq will be [len(data) - block_size - 1, len(data) - 2] (inclusive).
        # The last next token to be predicted will be <len(data) - 1>.
        # So the index won't be out of bound.
        self.perm = torch.randperm(len(self.data) - self.block_size)
        # Then first 90% are training data, and rest are for validation.
        n = int(self.training_set_ratio * len(self.perm))
        # We only save the start position of each example instead of the entire
        # sequence. The sequence will be generated when creating the batches.
        self.train_indices = self.perm[:n]
        self.val_indices = self.perm[n:]
        self.train_batchptr = 0
        self.val_batchptr = 0
    
    # If `random` is True, we randomly pick batch_size items from the set.
    # But since training_indices/val_indices are already shuffled, this is not really
    # needed.
    def get_batch(self, split:str, batch_size, device, random=False):
        # select training or validation set
        indices = self.train_indices if split == 'train' else self.val_indices
        ptr = self.train_batchptr if split == 'train' else self.val_batchptr

        # train_indices/val_indices stores the start locations
        if random:
            ix = torch.randint(len(indices), (batch_size,))
        else:
            # The train/val set is already shuffled, so we just need to sequentially
            # go through the items batch by batch.
            next = ptr + batch_size
            if next < len(indices):
                ix = indices[ptr:next]
            else:
                # Handle the case when we wrap around the list.
                next = next % len(indices)
                ix = torch.cat((indices[ptr:len(indices)], indices[0:next]))
            # Move the batch pointer
            if split == 'train':
                self.train_batchptr = next
            else:
                self.val_batchptr = next
        # Generate the actual examples & labels for the batch.        
        x = torch.stack([self.data[i:i+self.block_size] for i in ix])
        y = torch.stack([self.data[i+1:i+self.block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y


def encoder(s:str, c2i:dict):
    return [c2i[c] for c in s] 

def decoder(l, i2c:dict):
    return ''.join([i2c[i] for i in l]) 

def load_token_map(c2i_file:str = 'c2i.json', i2c_file:str='i2c.json'):
    # Load token map from the file.
    with open('c2i.json', 'r', encoding='utf-8') as f:
        c2i = json.load(f)

    # When loaded from JSON, the keys will become strings (e.g. '3913': '麒' instead of 3913: '麒')
    with open('i2c.json', 'r', encoding='utf-8') as f:
        i2c_raw = json.load(f)
        # Convert the keys to integers.
        i2c = {int(i):i2c_raw[i] for i in i2c_raw.keys()} 
    
    return c2i, i2c

# Test load/save token map.
def test_token_map():
    data = SanGuoData()
    data.ingest()
    # Save the token map to json files.
    data.save_token_map()

    # Load the token map and test encoding/decoding with it.
    c2i, i2c = load_token_map()

    print("Original text:")
    print(data.text[50:100])

    print("\nEncoded:")
    print(encoder(data.text[50:100], c2i))

    print("\nDecoded:")
    print(decoder(encoder(data.text[50:100], c2i), i2c))

if __name__ == "__main__":
    test_token_map()