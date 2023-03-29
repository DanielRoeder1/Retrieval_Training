import argparse
import yaml
from types import SimpleNamespace
import os
from datetime import datetime
import torch
from collections import Counter

############### Loading args ###############
def load_args():
    args = parse_args()
    # Keep config_path in args for wandb config
    if args.config_path is not None:
        dir_path = os.path.dirname(__file__)
        if args.config_path == 'default':
            args.config_path = os.path.join(dir_path, "configs/config.yaml")
        elif args.config_path == 'colab':
            args.config_path = os.path.join(dir_path, "configs/config_colab.yaml")
        
        tmp = args.config_path
        args = load_config(args.config_path)
        args.config_path = tmp
    
   
    #Config params automatically set 
    #args.training.eval_freq = determine_type(args.training.eval_freq)

    # Get wandb login
    if args.wandb.use:
            if args.wandb.credential_path == 'default':
                args.wandb.credential_path = os.path.join(dir_path, "configs/wandb_key.txt")
            with open(args.wandb.credential_path, 'r') as f:
                args.wandb.api_key = f.read()
            # Have config in dict format again for wandb
            with open(args.config_path, 'r') as f:
                args.wandb_config = yaml.safe_load(f)
    if args.d_model.path == "": args.d_model.path = None
    if args.training.accumulation_steps < 1: args.training.accumulation_steps = 1
    if args.evaluation.eval_accumulation < 1: args.evaluation.eval_accumulation = 1
    if args.training.use_torch_compile: assert torch.__version__ >= "2.0", "Torch version must be greater than 2.0 to use torch.compile(),install torch > 2.0 or diable in config"
    assert args.training.mode in ["bi-encoder", "poly-encoder"], "Invalid model type"
    assert args.q_model.path != '', "Must provide query model name, for Siamese networks only the q_model is used"
    return args


# YAML -> SimpleNamespace
# from: https://gist.github.com/jdthorpe/313cafc6bdaedfbc7d8c32fcef799fbf
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=Loader)
    return config

class Loader(yaml.Loader):
    pass

def _construct_mapping(loader, node):
    loader.flatten_mapping(node)
    return SimpleNamespace(**dict(loader.construct_pairs(node)))

Loader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default = None,
        help = "If path provided will use config to overwrite args, pass 'path'|'default'"
    )
    args = parser.parse_args()
    return args
############################################

def determine_type(str_in):
    """
    eval_freq argument can be either a number of steps, a float between 0 and 1, or "epoch"
    """
    if str_in.isdigit():
        return int(str_in)
    elif str_in.replace('.', '', 1).isdigit():
        return float(str_in)
    else:
        return str_in

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n

    def get_avg(self):
        return self.sum / self.count
    
    def __str__(self) -> str:
        return f"Avergage loss: {self.get_avg():.4f}, Current loss: {self.val:.4f}"
    
class AverageMeterDict:
    def __init__ (self):
        self.reset()
    def reset(self):
        self.sum = Counter()
        self.count = 0
    def update(self, val, n=1):
        self.sum += Counter(val)
        self.count += n
    def get_avg(self):
        avg = {}
        for k in self.sum: avg[k] =  self.sum[k] / self.count
        return avg


def get_eval_steps(eval_freq, total_batches, acc_steps):
    """
    Determine number of steps that trigger evaluatiuon
    """ 
    if eval_freq == "epoch":
        eval_steps = total_batches
    elif isinstance(eval_freq, int):
        eval_steps = eval_freq * acc_steps
    elif isinstance(eval_freq, float):
        eval_steps = int(total_batches * eval_freq)
    
    return eval_steps

def get_time():
    return datetime.now().strftime("%H:%M:%S")