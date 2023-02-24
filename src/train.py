from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from pytorch_metric_learning import losses
from torch.optim import AdamW

from utils import load_args
from data import  get_data_loader
from model import BiEncoder


def train(args):
    # Load the query and document encoders
    q_encoder = AutoModel.from_pretrained(args.q_model_name)
    q_tokenizer = AutoTokenizer.from_pretrained(args.q_model_name)
    # If no document encoder is provided, construct siamese model
    if args.d_model_name is None:
        print("[LOG]: Using the same encoder for query and document")
        d_encoder = q_encoder
        d_tokenizer = q_tokenizer
        optimizer = AdamW(q_encoder.parameters(), lr = args.lr)
    else:
        print("[LOG]: Using different encoders for query and document")
        d_encoder = AutoModel.from_pretrained(args.d_model_name)
        d_tokenizer = AutoTokenizer.from_pretrained(args.d_model_name)
        optimizer = AdamW(list(q_encoder.parameters()) + list(d_encoder.parameters()), lr =args.lr)
    
    if args.mode == "bi-encoder":
        model = BiEncoder(q_encoder, d_encoder)
    elif args.mode == "poly-encoder":
        pass
    
    # Get the data loader
    data = load_dataset("csv", data_files= args.dataset_path)
    train_loader = get_data_loader(data["train"],q_tokenizer, d_tokenizer, args.batch_size)
    val_loader = get_data_loader(data["val"],q_tokenizer, d_tokenizer, args.batch_size)
    # Define the loss function
    loss_func = losses.SelfSupervisedLoss(losses.NTXentLoss(temperature = 0.07))


    for epoch in range(args.epochs):
        model.train()
        for i, inputs in enumerate(train_loader):
            q_embeds, d_embeds = model(inputs)
            loss = loss_func(q_embeds, d_embeds)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % 100 == 0:
                print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss}")
                
        model.eval()
        for i, inputs in enumerate(val_loader):
            q_embeds, d_embeds = model(inputs)
            loss = loss_func(q_embeds, d_embeds)
            if i % 100 == 0:
                print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss}")


if __name__ == "__main__":
    args = load_args()
    train(args)