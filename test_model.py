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
work_dir = '{}/output/test/{}'.format(base_dir, time.strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(work_dir)
data_dir = '{}/datasets/wikitext-2-raw-v1'.format(base_dir)

tokenizer = BPETokenizer()
tokenizer.load(tokenizer_dir)

test_dataset = WikiFeeder('test', data_dir, tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=16)

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

def eval():
    with torch.no_grad():
        model.eval()
        print('evaluating...')
        eval_loss = meanVar()
        eval_iter = tqdm(test_dataloader, dynamic_ncols=True)
        for idx, (xs, labels) in enumerate(eval_iter):
            xs = xs.to(device)  # bz, 1024
            labels = labels.to(device)  # bz, 1024
            loss = get_loss(xs, labels)
            eval_loss.append(loss.item())
            eval_iter.set_description('[loss: {:.4f}]'.format(loss.item()))

        eval_loss = eval_loss.getMean()
        print('eval loss: {}'.format(eval_loss))

eval()