from model import PolyEncoder, PooledEncoder
from training_wrapper import PolyEncoderWrapper


from utils import load_args
import torch
from transformers import AutoTokenizer
from torch.optim import AdamW
from datasets import load_from_disk
from SectionSampling import get_data_loader
from torch.cuda.amp import GradScaler
from utils import get_time
from torch import autocast
from loss import AverageMeterDict, AverageMeter


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    faiss_device = torch.device(args.evaluation.faiss_device)
    q_model = PolyEncoder.from_pretrained(args.q_model_name, num_heads = 100)
    d_model = PooledEncoder.from_pretrained(args.d_model_name)
    model = PolyEncoderWrapper(d_model, q_model, args)
    # Load the query and document encoders
    q_tokenizer = AutoTokenizer.from_pretrained(args.q_model.path)
    # If no document encoder is provided, construct siamese model
    d_tokenizer = AutoTokenizer.from_pretrained(args.d_model.path)
    optimizer = AdamW(list(q_model.parameters()) + list(d_model.parameters()), lr =args.training.lr)

    assert q_model.config.hidden_size == d_model.config.hidden_size, "Query and document encoders must have the same hidden size"

    model.train()
    data = load_from_disk(args.paths.dataset_path)
    train_loader = get_data_loader(data["train"],q_tokenizer, d_tokenizer, args.training.batch_size)
    val_loader = get_data_loader(data["test"],q_tokenizer, d_tokenizer, args.evaluation.batch_size)
    # Set the loss function
   
    # Logging
    print_every = args.logging.print_freq  * args.training.accumulation_steps

    # Mixed precision training - Scaler
    scaler = GradScaler()
    av_train = AverageMeter()


    # Training loop
    print(f"[{get_time()}] [LOG]: Starting training")
    for epoch in range(args.training.epochs):
        av_train.reset()
        for i, inputs in enumerate(train_loader,1):
            for input in inputs: input.to(device)
            with autocast(device_type='cuda', dtype=torch.float16):
                loss = model(inputs)
                loss = loss / args.training.accumulation_steps
            
            scaler.scale(loss).backward()
            if i % args.training.accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            av_train.update(loss.item())
            if i % print_every== 0:
                print(f"[{get_time()}] [{epoch}/{args.training.epochs}, {i // args.training.accumulation_steps}/{num_batches // args.training.accumulation_steps}], Loss: {av_train}")
        
            #if i % eval_every == 0:
            #    best_val_loss = evaluate_during_train()
            #    model.train()







if __name__ == "__main__":
    args = load_args()
    train(args)