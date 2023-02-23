import random 
from functools import partial
from torch.utils.data import DataLoader


def transform_func(example):
    """
    Samples a subsequence from the document and uses it as the query
    """
    doc_text = example["text"][0]
    split_text = doc_text.split()
    seq_len = min(random.randint(len(split_text)//10, len(split_text)//5), 50)
    seq_begin = random.randint(0,len(split_text)- seq_len)
    query_text = " ".join(split_text[seq_begin:seq_begin+seq_len]) 
    return {"query": [query_text], "doc":[doc_text]}

def custom_collate(batch_data, tokenizer1, tokenizer2):
  """
  Custom collate function to tokenize query and docs
  """
  queries = [dct["query"] for dct in batch_data]
  docs = [dct["doc"] for dct in batch_data]

  q_tokens = tokenizer1(queries, padding = True, return_tensors = "pt", truncation =True)
  d_tokens = tokenizer2(docs, padding = True, return_tensors = "pt", truncation = True )

  return q_tokens, d_tokens

def get_data_loader(dataset, tokenizer1, tokenizer2, batch_size, shuffle = True):
  dataset.set_transform(transform_func)
  collate_fn = partial(custom_collate, tokenizer1 = tokenizer1, tokenizer2 = tokenizer2)
  return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, collate_fn = collate_fn)