from data import  get_data_loader
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from pytorch_metric_learning import losses, SelfSupervisedLoss
import torch
from torch.optim import AdamW

import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--q_model_name", 
        type=str, 
        default = None, 
        help = "Enter the HF name of the model to be used for query encoding")
    
    parser.add_argument(
        "--d_model_name",
        type=str,
        default = None,
        help = "Enter the HF name of the model to be used for document encoding"
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default = r'C:\Users\Daniel\Documents\RAG_thesis\data\train.csv',
        help = "Enter the path to the dataset (csv file)"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default = 32,
        help = "Enter the batch size"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default = 5,
        help = "Enter the number of epochs"
    )

    args = parser.parse_args()

    if args.dataset_path is not None:
        assert args.dataset_path.endswith(".csv"), "`train_file` should be a csv or a json file."
    else:
        raise ValueError("Need to specify a dataset path")
    
    return args

def mean_pooling(model_output, attention_mask):
    # from: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)



def train(args):
    # Load the dataset
    data = load_dataset("csv", data_files= args.dataset_path)
    # Load the query and document encoders
    q_encoder = AutoModel.from_pretrained(args.q_model_name)
    d_encoder = AutoModel.from_pretrained(args.d_model_name)
    q_tokenizer = AutoTokenizer.from_pretrained(args.q_model_name)
    d_tokenizer = AutoTokenizer.from_pretrained(args.d_model_name)
    # Get the data loader
    train_loader = get_data_loader(data["train"],q_tokenizer, d_tokenizer, args.batch_size)
    # Define the loss function
    loss_func = SelfSupervisedLoss(losses.NTXentLoss(temperature = 0.07))
    # Define the optimizer
    optimizer = AdamW(q_encoder.parameters(), lr = 1e-5)


    for epoch in range(args.epochs):
        for i, (q_inputs, d_inputs) in enumerate(train_loader):
            q_output = q_encoder(**q_inputs)
            d_output = d_encoder(**d_inputs)

            q_embeds = mean_pooling(q_output, q_inputs["attention_mask"])
            d_embeds = mean_pooling(d_output, d_inputs["attention_mask"])

            loss = loss_func(q_embeds, d_embeds)










if __name__ == "__main__":
    args = parse_args()
    train(args)
    data = load_dataset("csv", data_files= args.dataset_path)
    
    print(data)
    for i, data in enumerate(data["train"]):
        if i % 100_000 == 0:
            print(i)