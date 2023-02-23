from data import  get_data_loader
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset




if __name__ == "__main__":
    dataset = load_dataset("csv","../data/train.csv")    