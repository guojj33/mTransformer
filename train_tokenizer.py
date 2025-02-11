import pyarrow.parquet as pq
from BPETokenizer import BPETokenizer

base_dir = '.'

# dataset_path = '{}/datasets/wikitext-2-raw-v1'.format(base_dir)
# train_data = pq.read_table('{}/train-00000-of-00001.parquet'.format(dataset_path))['text'].to_pylist()

dataset_path = '{}/datasets/yourbench-fairytales'.format(base_dir)
train_data = pq.read_table('{}/train-00000-of-00001.parquet'.format(dataset_path))['content'].to_pylist()

tokenizer = BPETokenizer()
tmp_workspace = '{}/output/tmp'.format(base_dir)
try:
    # tokenizer.load(tmp_workspace, load_state=True)
    tokenizer.train(train_data, resume=False)
except Exception as e:
    tokenizer.save(tmp_workspace, save_state=True)
    print(e)
else:
    tokenizer.save(tmp_workspace)