# mTransformer

## Dataset

download the following files from [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext/tree/main/wikitext-2-raw-v1) and put them under `./datasets/wikitext-2-raw-v1`

```
test-00000-of-00001.parquet
train-00000-of-00001.parquet
validation-00000-of-00001.parquet
```

## BPETokenizer

see `train_tokenizer.py` and `test_tokenizer.py`

## GPT-2-like Transformer

see `train_model.py`

## References

- [hugginface/transformers/gpt2](https://github.com/huggingface/transformers/tree/main/src/transformers/models/gpt2)