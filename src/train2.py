import torch
import wandb
from torch.cuda.amp import GradScaler
from torch import autocast
from utils import load_args, AverageMeter, get_time
from training_utils import get_data_loaders, load_model,get_logging_vars, save_model
from evaluation import Evaluator

def train(args):
    if args.wandb.use:
        wandb.login(key = args.wandb.api_key)
        wandb.init(args.wandb.project_name, config=args.wandb_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    faiss_device = torch.device(args.evaluation.faiss_device)

    model, q_tokenizer, d_tokenizer, optimizer = load_model(args, device)
    train_loader, val_loader = get_data_loaders(args, q_tokenizer, d_tokenizer, True)
    print_every, num_batches, eval_every = get_logging_vars(args, len(train_loader))
    eval = Evaluator(args, val_loader, device, faiss_device, model.q_model.config.hidden_size)
    eval_func = getattr(eval, model.eval_type)
    
    av_train = AverageMeter()
    best_val_loss = float("inf")
    scaler = GradScaler()

    # Training loop
    model.train()
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
                if args.wandb.use: wandb.log({"Avg train Loss": av_train.get_avg(), "loss":av_train.val}, step = i + epoch * num_batches)
                print(f"[{get_time()}] [{epoch}/{args.training.epochs}, {i // args.training.accumulation_steps}/{num_batches // args.training.accumulation_steps}], Loss: {av_train}")
        
            if i % eval_every == 0:
                avg_loss, avg_acc = eval_func(model)
                if args.wandb.use: wandb.log(avg_acc, step = i + epoch * num_batches)
                print(f"[{get_time()}] Epoch: {epoch}, Average Loss {avg_loss},  \n Average Metrics: {avg_acc}")

                if avg_loss < best_val_loss:
                    save_model(model, args.paths.save_path, epoch, i, avg_loss)
                    best_val_loss = avg_loss
                model.train()
                
        
if __name__ == "__main__":
    args = load_args()
    train(args)