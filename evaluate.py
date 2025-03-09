import torch
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy, softmax
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
tokenizer_dir = '{}/tokenizer/wikitext'.format(base_dir)

tokenizer = BPETokenizer()
tokenizer.load(tokenizer_dir)

device = torch.device('cpu')
if torch.cuda.is_available():
   device = torch.device('cuda:0')
elif torch.backends.mps.is_available():
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
  model.load_state_dict(checkpoint['model'])
  print('loading from {}'.format(load_path))

load_path = '/Users/jj/Desktop/workspace/mTransformer/output/train/2025-02-28_15-28-57/checkpoint.pth.tar'
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
    xs = torch.tensor(input_ids).to(device) # [seq_count, seq_len]
    _, logits, _ = model(xs)   # logits [seq_count, seq_len, vocab_size]
    output_ids = logits.max(dim=-1)[1] # [seq_count, seq_len]
    output_ids = output_ids.cpu().numpy().tolist()
    return output_ids

def auto_reg_generate(input_ids):
    max_gen_len = 500
    xs = torch.tensor(input_ids).to(device)
    cur_len = xs.shape[1]
    _, logits, presents = model(xs)

    def sample(logits, temperature=0.5): # [num_sample, vocab_size]
      probs = softmax(logits/temperature, dim=-1)
      return torch.multinomial(probs, num_samples=1) # [num_sample, 1]

    # predict first next token
    logits = logits[:,-1,:].detach().clone() # logit of the last token of each sequence [seq_count, 1]
    output_next_ids = sample(logits)  # [seq_count, 1]
    cur_len += 1
    
    # store all predict tokens
    output_ids = output_next_ids.detach().clone()

    while cur_len < max_gen_len:
      position_ids = torch.ones_like(output_next_ids)  # [seq_count, 1]
      position_ids *= (cur_len-1)  # position id of the last token
      xs = output_next_ids
      _, logits, presents = model(xs, past_key_value=presents, position_ids=position_ids)

      # output_next_ids = logits[:,-1,:].max(dim=-1)[1][:,None].clone().detach()  # [seq_count, 1]
      logits = logits[:,-1,:].detach().clone()
      output_next_ids = sample(logits)
      output_ids = torch.cat((output_ids, output_next_ids), dim=-1)
      cur_len += 1

    return output_ids.cpu().numpy().tolist()

def decode_output(output_ids):
    output_texts = []
    for ids in output_ids:
        output = tokenizer.decode(ids)
        output_texts.append(output)
    return output_texts

def evaluate():
  model.eval()
  with torch.no_grad():
    input_texts = ['how are you']
    bpe_tokens, input_ids = tokenize_input(input_texts)
    print(input_ids)
    output_ids = generate(input_ids)
    print(output_ids)
    output_texts = decode_output(output_ids)
    for i, text in enumerate(input_texts):
      output_texts[i] = output_texts[i]

    print('input:')
    print(input_texts)
    print('output:')
    print(output_texts)

evaluate()