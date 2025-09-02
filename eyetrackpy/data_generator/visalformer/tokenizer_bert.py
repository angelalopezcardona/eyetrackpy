import torch
import os
from transformers import AutoTokenizer

# Set custom cache directory in user's home
os.environ['TRANSFORMERS_CACHE'] = os.path.expanduser('~/.cache/huggingface')

# tokenizer = AutoTokenizer.from_pretrained("roberta-base")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print('bert-base-uncased tokenizer loaded')

def padding_fn(data):
    img, q, fix, hm, name = zip(*data)

    input_ids = tokenizer(q, return_tensors="pt", padding=True)

    return torch.stack(img), input_ids, torch.stack(fix), torch.stack(hm), name


