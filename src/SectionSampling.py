import re
from tqdm import tqdm
import numpy as np

from torch.utils.data import DataLoader

##### Dataset Generation #####
class SectionSampler:
  """
  Samples anchors and positives from the KILT 2019 Wikipedia dump
  Anchors are extracted from sufficiently long Wikiepedia sections
  Positives are samples from the corresponding sections as well as the first section of the Wiki article -> general information
  """
  def __init(self):
    self.min_sec_len = 300
    self.max_sec_len = 450
    self.min_pos_len = 40
    self.num_positives = 2

  def extract_sections(self, text, pattern = "Section::::[\w ]+\.\\n"):
      tmp = 0 
      for m in re.finditer(pattern, text):
          yield text[tmp:m.start()]
          tmp = m.start()
      yield text[tmp:len(text)]

  def clean_positives(sefl, pos):
      return re.sub("Section::::\w+\.(:[\w ]+.\\n)?", "",pos).replace("\n","")

  def extract_samples(self,dataset):
    final_data= []

    for entry in tqdm(dataset["train"]):
        full_text = " ".join(entry["text"])
        section_gen = self.extract_sections(full_text)
        init_positive = " ".join(next(section_gen).split()[:np.random.randint(self.min_pos_len,100)])
        init_positive = self.clean_positives(init_positive)

        for sec in section_gen:
            sec_tokens = sec.split()
            sec_len = len(sec_tokens)

            if sec_len > self.min_sec_len:
                positives = [init_positive]
                sec_split= min(self.max_sec_len,sec_len //2)
                anchor = " ".join(sec_tokens[:sec_split])
                for _ in range(self.num_positives):
                    positive_len = int(np.random.beta(2, 4) * (sec_split - self.min_pos_len) + self.min_pos_len)
                    positive_start = np.random.randint(0, sec_len - positive_len+1)
                    pos = " ".join(sec_tokens[positive_start:positive_start + positive_len])
                    positives.append(self.clean_positives(pos))
                final_data.append({"anchor":anchor, "positive":positives})
    return final_data

def chunk(input, chunk_size = 3):
  return [input[i:i+chunk_size] for i in range(0, len(input), chunk_size)]

def tokenize_func(sample):
  anchors = sample["anchor"]
  positives = [pos_text for positive in sample["positive"] for pos_text in positive]
  anchors=tokenizer(anchors, truncation = True, return_attention_mask = False)
  positives = tokenizer(positives, truncation = True, return_attention_mask = False) 
  anchors["pos_input_ids"] = chunk(positives["input_ids"])
  return anchors

##### Dataset Loading #####
def outer_collate(tokenizer_doc, tokenizer_query):
  def collate_fn(data):
    doc = [{"input_ids": input_dict["input_ids"]} for input_dict in data]
    query = [{"input_ids":ids } for input_dict in data for ids in input_dict["pos_input_ids"]]
    doc = tokenizer_doc.pad(doc, return_attention_mask=True, return_tensors="pt")
    query = tokenizer_query.pad(query, return_attention_mask=True, return_tensors="pt")
    return doc, query
  return collate_fn

def get_data_loader(dataset, tokenizer1, tokenizer2, batch_size, shuffle = True):
  collate_fn = outer_collate(tokenizer1, tokenizer2)
  return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, collate_fn = collate_fn)