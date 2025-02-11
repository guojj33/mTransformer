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
work_dir = '{}/output/train/{}'.format(base_dir, time.strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(work_dir)
data_dir = '{}/datasets/wikitext-2-raw-v1'.format(base_dir)

tokenizer = BPETokenizer()
tokenizer.load(tokenizer_dir)

train_dataset = WikiFeeder('train', data_dir, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
# test_dataset = WikiFeeder('test', data_dir, tokenizer)
# test_dataloader = DataLoader(test_dataset, batch_size=16)

device = torch.device('cpu')
if torch.cuda.is_available():
   device = torch.device('cuda:0')
elif torch.mps.is_available():
  device = torch.device('mps:0')

start_epoch = 0
max_epoch = 100
warm_up_epoch = 3

model = mTransformer(tokenizer.vocabulary_size()).to(device)
learning_rate = 5e-5
optimizer = Adam(model.parameters(), lr=learning_rate)
lr_lambda = Cosine_Scheduler(len(train_dataloader), max_epoch, warm_up_epoch, warm_up_iters).get_lambda()
scheduler = LambdaLR(optimizer, lr_lambda)

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

def save_checkpoint(epoch, name='checkpoint'):
   print('saving {}...'.format(name))
   checkpoint = {
      'model': model,
      'optimizer': optimizer.state_dict(),
      'scheduler': scheduler.state_dict(),
      'epoch': epoch
   }
   torch.save(checkpoint, '{}/{}.pth.tar'.format(work_dir, name))
   print('{} saved.'.format(name))

def load_checkpoint(load_path):
  checkpoint = torch.load(load_path, weights_only=False, map_location=device)
  global model, optimizer, scheduler, start_epoch
  model = checkpoint['model']
  optimizer.load_state_dict(checkpoint['optimizer'])
  scheduler.load_state_dict(checkpoint['scheduler'])
  start_epoch = checkpoint['epoch']
  print('resume training: loading from {}'.format(load_path))

load_path = './output/train/2025-02-11_10-17-47/checkpoint.pth.tar'
if not load_path is None:
   load_checkpoint(load_path)

for e in range(start_epoch, max_epoch):
    print('[train epoch {}/{}]'.format(e+1, max_epoch))
    model.train()
    train_loss = meanVar()
    train_iter = tqdm(train_dataloader, dynamic_ncols=True)
    for idx, (xs, labels) in enumerate(train_iter):
        optimizer.zero_grad()
        xs = xs.to(device)  # bz, 1024
        labels = labels.to(device)  # bz, 1024
        loss = get_loss(xs, labels)
        train_loss.append(loss.item())
        train_iter.set_description('[loss: {:.4f}, lr: {:.4f}(1e-4)]'.format(loss.item(), optimizer.param_groups[0]['lr']*1e4))

        loss.backward()

        optimizer.step()
        scheduler.step()

    train_loss = train_loss.getMean()
    print('train epoch: {}, loss: {}'.format(e+1, train_loss))
    save_checkpoint(e+1)
