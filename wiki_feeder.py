from torch.utils.data import Dataset
import pyarrow.parquet as pq
from itertools import chain
import numpy as np

class WikiFeeder(Dataset):
    '''
    load:
        raw text
    return:
        bpe tokens, token ids
    '''
    def __init__(self, phase, dir, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        
        # load raw text
        dataset_path = '{}/{}-00000-of-00001.parquet'.format(dir, phase)
        raw_data = pq.read_table(dataset_path)['text'].to_pylist()   # debug
        print('loading data from {}, sequence count: {}'.format(dataset_path, len(raw_data)))
        # tokenize
        print('tokenizing...')
        tokenized_data = {'tokens': [], 'ids': []}
        for i in range(len(raw_data)):
            tokens, ids = tokenizer.encode(raw_data[i])
            tokenized_data['tokens'].append(tokens)
            tokenized_data['ids'].append(ids)
        # group into chunks
        self.chunk_size = 1024
        def group(examples):
            concatenated_examples = list(chain(*examples))  # [[11,22],[33,44]] -> [11,22,33,44]
            total_length = len(concatenated_examples)
            total_length = (total_length // self.chunk_size) * self.chunk_size # drop the remainder
            result = [concatenated_examples[i:i+self.chunk_size] for i in range(0, total_length, self.chunk_size)]
            return result
        print('grouping sequences into chunks...')
        self.lm_data = {k: group(v) for k, v in tokenized_data.items()}

    def __len__(self):
        return len(self.lm_data['ids'])

    def __getitem__(self, idx):
        x = np.array(self.lm_data['ids'][idx])
        label = x.copy()
        return x, label