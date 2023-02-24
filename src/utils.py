import argparse
import yaml
from types import SimpleNamespace
import os

############### Loading args ###############
def load_args():
    args = parse_args()
    if args.config_path is not None:
        if args.config_path == 'default':
            dir_path = os.path.dirname(__file__)
            args = load_config(os.path.join(dir_path, "configs/config.yaml"))
        else:
            args = load_config(args.config_path)
    assert args.q_model_name is not None, "Must provide query model name, for Siamese networks only the q_model is used"
    return args


# YAML -> Namespace
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
#

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_path",
        type=str,
        default = None,
        help = "If path provided will use config to overwrite args, pass 'path'|'default'"
    )

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

    parser.add_argument(
        "--lr",
        type=float,
        default = 1e-5,
        help = "Enter the learning rate"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default = None,
        help = "Enter the mode of the model, either 'bi-encoder' or 'poly-encoder'"
    )

    args = parser.parse_args()

    if args.dataset_path is not None:
        assert args.dataset_path.endswith(".csv"), "train_file` should be a csv file"
    else:
        raise ValueError("Need to specify a dataset path")
    
    return args
############################################
