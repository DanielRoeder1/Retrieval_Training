from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset, load_from_disk
from pytorch_metric_learning import losses
from torch.optim import AdamW
import torch
import wandb
import os

# Mixed precision training
from torch.cuda.amp import GradScaler
from torch import autocast

from utils import load_args, AverageMeter, get_eval_steps, get_time, AverageMeterDict
#from data import  get_data_loader
from SectionSampling import get_data_loader
from model import BiEncoder
from loss import CrossBatchMemoryWrapper, CustomAccuracyCalc, LabelLossWrapper


def train(args):
    if args.wandb.use:
        wandb.login(key = args.wandb.api_key)
        wandb.init(args.wandb.project_name, config=args.wandb_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the query and document encoders
    q_encoder = AutoModel.from_pretrained(args.q_model.path)
    q_tokenizer = AutoTokenizer.from_pretrained(args.q_model.path)
    # If no document encoder is provided, construct siamese model
    if args.d_model.path is None:
        print(f"[{get_time()}] [LOG]: Using the same encoder for query and document")
        d_encoder = q_encoder
        d_tokenizer = q_tokenizer
        optimizer = AdamW(q_encoder.parameters(), lr = args.training.lr)
    else:
        print(f"[{get_time()}] [LOG]: Using different encoders for query and document")
        d_encoder = AutoModel.from_pretrained(args.d_model.path)
        d_tokenizer = AutoTokenizer.from_pretrained(args.d_model.path)
        optimizer = AdamW(list(q_encoder.parameters()) + list(d_encoder.parameters()), lr =args.training.lr)
    assert d_encoder.config.hidden_size == q_encoder.config.hidden_size, "Query and document encoders must have the same hidden size"
    # Set model type
    if args.training.mode == "bi-encoder":
        model = BiEncoder(q_encoder, d_encoder).to(device)
    elif args.training.mode == "poly-encoder":
        pass
    model.train()
    assert args.training.mode in ["bi-encoder", "poly-encoder"], "Invalid model type"
    # Get the data loader
    #data = load_dataset("csv", data_files= args.dataset_path)
    data = load_from_disk(args.paths.dataset_path)
    train_loader = get_data_loader(data["train"],q_tokenizer, d_tokenizer, args.training.batch_size)
    val_loader = get_data_loader(data["test"],q_tokenizer, d_tokenizer, args.training.batch_size)
    # Set the loss function
    if args.training.accumulation_steps < 1: args.training.accumulation_steps = 1
    if args.cross_batch_memory.use:
        loss_func = CrossBatchMemoryWrapper(q_encoder.config.hidden_size, device, memory_size=args.cross_batch_memory.buffer_size, warmup = args.cross_batch_memory.warmup, acc_steps= args.training.accumulation_steps, num_pos =args.training.num_pos)
    else:
        loss_func = LabelLossWrapper(losses.NTXentLoss(temperature = 0.07), num_pos = args.training.num_pos, device = device)
    # Accuracy metrics for evaluation
    acc_calc = CustomAccuracyCalc()
    # Logging
    print_every = args.logging.print_freq  * args.training.accumulation_steps
    av_train = AverageMeter()
    av_val = AverageMeter()
    av_val_acc = AverageMeterDict()
    num_batches = len(train_loader)
    eval_every = get_eval_steps(args.training.eval_freq,num_batches)
    best_val_loss = float("inf")

    # Mixed precision training - Scaler
    scaler = GradScaler()

    # TODO: fix accuracy calculation
    # TODO: fix best_val_loss referenced before assignment
    def evaluate_during_train():
        print(f"[{get_time()}] [LOG]: Evaluating model")             
        model.eval()
        av_val.reset()
        av_val_acc.reset()

        for inputs in val_loader:
            with torch.no_grad():
                for input in inputs: input.to(device)
                q_embeds, d_embeds = model(inputs)
                loss = loss_func(q_embeds, d_embeds)
                av_val.update(loss.item())
            acc_metrics = acc_calc.get_acc_wrapper(q_embeds, d_embeds)
            av_val_acc.update(acc_metrics)

        print(f"[{get_time()}] Epoch: {epoch}, Average Loss {av_val},  \n Average Metrics: {av_val_acc.get_avg()}")
        avg_loss = av_val.get_avg()
        if avg_loss < best_val_loss:
            print(f"[{get_time()}] [LOG]: Saving model")
            model_name = "checkpoint_epoch_{epoch}_steps{i}_loss_{avg_loss}.pt"
            torch.save(model.state_dict(), os.path.join(args.paths.save_path, model_name))
            return avg_loss
        else:
            return best_val_loss


    # Training loop
    print(f"[{get_time()}] [LOG]: Starting training")
    for epoch in range(args.training.epochs):
        av_train.reset()
        for i, inputs in enumerate(train_loader,1):
            for input in inputs: input.to(device)
            with autocast(device_type='cuda', dtype=torch.float16):
                q_embeds, d_embeds = model(inputs)
                loss = loss_func(q_embeds, d_embeds)
                loss = loss / args.training.accumulation_steps
            
            scaler.scale(loss).backward()
            if i % args.training.accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            av_train.update(loss.item())
            if i % print_every== 0:
                if args.wandb.use: wandb.log({"Avg train Loss": av_train.get_avg(), "loss":av_train.val})
                print(f"[{get_time()}] [{epoch}/{args.training.epochs}, {i // args.training.accumulation_steps}/{num_batches // args.training.accumulation_steps}], Loss: {av_train}")
        
            if i % eval_every == 0:
                best_val_loss = evaluate_during_train()
                model.train()
                
        
if __name__ == "__main__":
    args = load_args()
    train(args)