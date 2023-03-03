from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset, load_from_disk
from pytorch_metric_learning import losses
from torch.optim import AdamW
import torch
import wandb

# Mixed precision training
from torch.cuda.amp import GradScaler
from torch import autocast

from utils import load_args, AverageMeter, get_eval_steps, get_time, AverageMeterDict
from data import  get_data_loader
from model import BiEncoder
from loss import CrossBatchMemoryWrapper, CustomAccuracyCalc


def train(args):
    if args.wandb.use:
        wandb.login(args.wandb.api_key)
        wandb.init(args.wandb.project_name, config=args.config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the query and document encoders
    q_encoder = AutoModel.from_pretrained(args.q_model_name)
    q_tokenizer = AutoTokenizer.from_pretrained(args.q_model_name)
    # If no document encoder is provided, construct siamese model
    if args.d_model_name is None:
        print(f"[{get_time()}] [LOG]: Using the same encoder for query and document")
        d_encoder = q_encoder
        d_tokenizer = q_tokenizer
        optimizer = AdamW(q_encoder.parameters(), lr = args.lr)
    else:
        print(f"[{get_time()}] [LOG]: Using different encoders for query and document")
        d_encoder = AutoModel.from_pretrained(args.d_model_name)
        d_tokenizer = AutoTokenizer.from_pretrained(args.d_model_name)
        optimizer = AdamW(list(q_encoder.parameters()) + list(d_encoder.parameters()), lr =args.lr)
    assert d_encoder.config.hidden_size == q_encoder.config.hidden_size, "Query and document encoders must have the same hidden size"
    # Set model type
    if args.mode == "bi-encoder":
        model = BiEncoder(q_encoder, d_encoder).to(device)
    elif args.mode == "poly-encoder":
        pass
    assert args.mode in ["bi-encoder", "poly-encoder"], "Invalid model type"
    # Get the data loader
    #data = load_dataset("csv", data_files= args.dataset_path)
    data = load_from_disk(args.dataset_path)
    train_loader = get_data_loader(data["train"],q_tokenizer, d_tokenizer, args.batch_size)
    val_loader = get_data_loader(data["test"],q_tokenizer, d_tokenizer, args.batch_size)
    # Set the loss function
    if args.accumulation_steps < 1: args.accumulation_steps = 1
    if args.cross_batch_memory.use:
        loss_func = CrossBatchMemoryWrapper(q_encoder.config.hidden_size, device, memory_size=args.cross_batch_memory.buffer_size, warmup = args.cross_batch_memory.warmup, acc_steps= args.accumulation_steps)
    else:
        loss_func = losses.SelfSupervisedLoss(losses.NTXentLoss(temperature = 0.07))
    # Accuracy metrics for evaluation
    acc_calc = CustomAccuracyCalc()
    acc_labels = torch.arange(args.batch_size).to(device)
    # Logging
    print_every = args.print_freq  * args.accumulation_steps
    av_train = AverageMeter()
    av_val = AverageMeter()
    av_val_acc = AverageMeterDict()
    num_batches = len(train_loader)
    eval_every = get_eval_steps(args.eval_freq,num_batches)
    best_val_loss = float('inf')
    # Mixed precision training - Scaler
    scaler = GradScaler()

    def evaluate_during_train():
        print(f"[{get_time()}] [LOG]: Evaluating model")             
        model.eval()
        av_val.reset()
        av_val_acc.reset()

        for i, inputs in enumerate(val_loader):
            with torch.no_grad():
                for input in inputs: input.to(device)
                q_embeds, d_embeds = model(inputs)
                loss = loss_func(q_embeds, d_embeds)
                av_val.update(loss.item())
            acc_metrics = acc_calc.get_accuracy(query = q_embeds, reference = d_embeds,query_labels =  acc_labels, reference_labels = acc_labels)
            av_val_acc.update(acc_metrics)

        print(f"[{get_time()}] Epoch: {epoch}, Average Loss {av_val},  \n Average Metrics: {av_val_acc.get_avg()}")
        if av_val.get_avg() < best_val_loss:
            print(f"[{get_time()}] [LOG]: Saving model")
            torch.save(model.state_dict(), args.save_path)
            best_val_loss = av_val.avg


    # Training loop
    print(f"[{get_time()}] [LOG]: Starting training")
    for epoch in range(args.epochs):
        av_train.reset()
        model.train()
        for i, inputs in enumerate(train_loader):
            for input in inputs: input.to(device)
            with autocast(device_type='cuda', dtype=torch.float16):
                q_embeds, d_embeds = model(inputs)
                loss = loss_func(q_embeds, d_embeds)
                loss = loss / args.accumulation_steps
            
            scaler.scale(loss).backward()
            if (i + 1) % args.accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            av_train.update(loss.item())
            if i % (print_every)== 0:
                print(f"[{get_time()}] [{epoch}/{args.epochs}, {i // args.accumulation_steps}/{num_batches // args.accumulation_steps}], Loss: {av_train}")
        
            if (i+1) % eval_every == 0:
                evaluate_during_train()
                
        
if __name__ == "__main__":
    args = load_args()
    train(args)