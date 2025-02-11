import torch
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

import time, os
from tqdm import tqdm
import numpy as np

from model import mTransformer
from wiki_feeder import WikiFeeder
from BPETokenizer import BPETokenizer
from lr_schedulers import Cosine_Scheduler

base_dir = '.'
tokenizer_dir = '{}/tokenizer'.format(base_dir)

tokenizer = BPETokenizer()
tokenizer.load(tokenizer_dir)

device = torch.device('cpu')
if torch.cuda.is_available():
   device = torch.device('cuda:0')
elif torch.mps.is_available():
  device = torch.device('mps:0')

model = mTransformer(tokenizer.vocabulary_size()).to(device)

def get_loss(xs, labels):
    _, logits = model(xs) # bz, 1024, 30256
    shift_logits = logits[:, :-1, :].contiguous()
    shift_label = labels[:, 1:].contiguous()
    loss = cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), shift_label.view(-1))   # [bz*1024, 30256], [bz*1024]
    return loss

class meanVar():
  def __init__(self):
    self.l = []
  def append(self, x):
    self.l.append(x)
  def getMean(self):
    return np.array(self.l).sum()/len(self.l)

def load_checkpoint(load_path):
  checkpoint = torch.load(load_path, weights_only=False, map_location=device)
  global model
  model = checkpoint['model']
  print('loading from {}'.format(load_path))

load_path = './output/train/2025-02-11_10-41-04/checkpoint.pth.tar'
load_checkpoint(load_path)

def tokenize_input(input_texts):
    bpe_tokens = []
    input_ids = []
    for text in input_texts:
        tokens, ids = tokenizer.encode(text)
        bpe_tokens.append(tokens)
        input_ids.append(ids)
    return bpe_tokens, input_ids

def generate(input_ids):
    xs = torch.tensor(input_ids).to(device) # seq_count, seq_len
    _, logits = model(xs)   # seq_count, seq_len, vocab_size
    output_ids = logits.max(dim=-1)[1] # seq_count, seq_len
    output_ids = output_ids.cpu().numpy().tolist()
    return output_ids

def decode_output(output_ids):
    output_texts = []
    for ids in output_ids:
        output = tokenizer.decode(ids)
        output_texts.append(output)
    return output_texts

def inferece():
  model.eval()
  with torch.no_grad():
    input_texts = [' what is the whether like today']
    bpe_tokens, input_ids = tokenize_input(input_texts)
    output_ids = generate(input_ids)
    output_texts = decode_output(output_ids)

    print('input:')
    print(input_texts)
    print(input_ids)
    print('output:')
    print(output_texts)
    print(output_ids)

inferece()