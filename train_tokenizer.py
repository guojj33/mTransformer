import pyarrow.parquet as pq
from BPETokenizer import BPETokenizer

dataset_path = './datasets/wikitext-2-raw-v1'
train_data = pq.read_table('{}/train-00000-of-00001.parquet'.format(dataset_path))['text'].to_pylist()
test_data = pq.read_table('{}/test-00000-of-00001.parquet'.format(dataset_path))['text'].to_pylist()

tokenizer = BPETokenizer()
tmp_workspace = './output/tmp'
try:
    tokenizer.load(tmp_workspace, load_state=True)
    tokenizer.train(train_data, resume=True)
except Exception as e:
    tokenizer.save(tmp_workspace, save_state=True)
    print(e)
else:
    tokenizer.save(tmp_workspace)