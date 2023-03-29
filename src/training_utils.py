from datasets import load_from_disk
from torch.utils.data import DataLoader
from SectionSampling import outer_collate

from transformers import AutoTokenizer
from model import PooledEncoder, PolyEncoder
from model_training_wrapper import BiEncoderWrapper, PolyEncoderWrapper 
import torch
from torch.optim import AdamW

from utils import get_time, get_eval_steps
import os

def get_data_loaders(args, q_tokenizer, d_tokenizer, shuffle = True):
  data = load_from_disk(args.paths.dataset_path)
  collate_fn = outer_collate(q_tokenizer, d_tokenizer)
  train_loader = DataLoader(data["train"], batch_size = args.train.batch_size, shuffle = shuffle, collate_fn = collate_fn)
  val_loader = DataLoader(data["test"], batch_size = args.evaluation.batch_size, shuffle = shuffle, collate_fn = collate_fn)
  return train_loader, val_loader


def load_model(args, device):
    # Selecting architeture
    if args.training.mode == "bi-encoder":
        q_model = PooledEncoder
        d_model = PooledEncoder
        model = BiEncoderWrapper
    elif args.training.mode == "poly-encoder":
        q_model = PolyEncoder
        d_model = PooledEncoder
        model = PolyEncoderWrapper
        assert hasattr(args.q_model.model_args,"num_poly_codes"), "Must provide num_poly_codes in q_model.model_args when using PolyEncoder"
        
    
    # Query Encoder
    q_model = q_model.from_pretrained(args.q_model.path, args.q_model.model_args).to(device)
    q_tokenizer = AutoTokenizer.from_pretrained(args.q_model.path)
    q_tokenizer.add_special_tokens({'additional_special_tokens': args.q_model.special_tokens})
    q_model.resize_token_embeddings(len(q_tokenizer))

    # Document Encoder
    if args.d_model.path is None:
        print(f"[{get_time()}] [LOG]: Using the same encoder for query and document")
        d_model = d_model(q_model.config, q_model.enc_model).to(device)
        d_tokenizer = q_tokenizer
        optimizer = AdamW(q_model.parameters(), lr = args.training.lr)
    else:
        print(f"[{get_time()}] [LOG]: Using different encoders for query and document")
        d_model = d_model.from_pretrained(args.d_model.path, args.d_model.model_args).to(device)
        d_tokenizer = AutoTokenizer.from_pretrained(args.d_model.path)
        optimizer = AdamW(list(q_model.parameters()) + list(d_model.parameters()), lr =args.training.lr)
    d_tokenizer.add_special_tokens({'additional_special_tokens': args.d_model.special_tokens})
    d_model.resize_token_embeddings(len(d_tokenizer))

    assert q_model.config.hidden_size == d_model.config.hidden_size, "Query and document encoders must have the same hidden size"
    model = model(d_model, q_model, args)

    if args.training.use_torch_compile:
        model.q_model = torch.compile(model.q_model)
        model.d_model = torch.compile(model.d_model)
    
    return model, q_tokenizer, d_tokenizer, optimizer

def get_logging_vars(args, num_batches):
    print_every = args.logging.print_freq  * args.training.accumulation_steps
    eval_every = get_eval_steps(args.evaluation.eval_freq,num_batches)
    return print_every, num_batches, eval_every

def save_model(model, path, epoch, i, avg_loss):
    print(f"[{get_time()}] [LOG]: Saving model")
    chkpt_name = f"checkpoint_epoch_{epoch}_steps{i}_loss_{avg_loss:.4f}"
    chkpt_dir = os.path.join(path, chkpt_name)
    q_model_dir = os.path.join(chkpt_dir, "q_model")
    d_model_dir = os.path.join(chkpt_dir, "d_model")
    os.makedirs(q_model_dir)
    os.makedirs(d_model_dir)
    model.q_model.save_pretrained(q_model_dir)
    model.d_model.save_pretrained(d_model_dir)

