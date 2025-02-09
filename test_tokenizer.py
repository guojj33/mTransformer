import pyarrow.parquet as pq
from BPETokenizer import BPETokenizer

dataset_path = './datasets/wikitext-2-raw-v1'
train_data = pq.read_table('{}/train-00000-of-00001.parquet'.format(dataset_path))['text'].to_pylist()
test_data = pq.read_table('{}/test-00000-of-00001.parquet'.format(dataset_path))['text'].to_pylist()

tokenizer = BPETokenizer()
tokenizer.load('./tokenizer')
# input = test_data[100]
input = ' 你好'
bpe_tokens, ids = tokenizer.encode(input)
output = tokenizer.decode(ids)
print(input)
print(bpe_tokens)
print(ids)
print(output)
print(input == output)