import torch
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from tqdm import tqdm
import numpy as np

from model import mTransformer
from wiki_feeder import WikiFeeder
from BPETokenizer import BPETokenizer
from lr_schedulers import Cosine_Scheduler

tokenizer = BPETokenizer()
tokenizer.load('./output/tokenizer')

data_dir = './datasets/wikitext-2-raw-v1'
train_dataset = WikiFeeder('train', data_dir, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
# test_dataset = WikiFeeder('test', data_dir, tokenizer)
# test_dataloader = DataLoader(test_dataset, batch_size=16)
device = torch.device('mps:0')
max_epoch = 100
warm_up_epoch = 5

model = mTransformer(tokenizer.vocabulary_size()).to(device)
optimizer = Adam(model.parameters(), lr=5e-5)
lr_lambda = Cosine_Scheduler(len(train_dataloader), max_epoch, warm_up_epoch).get_lambda()
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

for e in range(max_epoch):
    model.train()
    train_loss = meanVar()
    train_iter = tqdm(train_dataloader, dynamic_ncols=True)
    for idx, (xs, labels) in enumerate(train_iter):
        optimizer.zero_grad()

        xs = torch.tensor(xs).to(device)  # bz, 1024
        labels = torch.tensor(labels).to(device)  # bz, 1024
        loss = get_loss(xs, labels)
        train_loss.append(loss.item())
        train_iter.set_description('[loss: {:.4f}, lr: {:.4f}(1e-4)]'.format(loss.item(), optimizer.param_groups[0]['lr']*1e4))

        loss.backward()

        optimizer.step()
        scheduler.step()

    train_loss = train_loss.getMean()
    print('train epoch: {}, loss: {}'.format(e+1, train_loss))
    torch.save(model, './output/train/model.pth.tar')
