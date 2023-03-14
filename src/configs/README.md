# Config Params explained

#### Paths:
- dataset_path: Path to the hugginface dataset (str)
- save_path: Path to folder in which model checkpoints are saved (str)
#### Q_model:
- path: Hugginface name or local path to query encoder model (required) (str)
- special_tokens: The special tokens that are supposed to be addded to the tokenizer List(str)
#### D_model:
- see q_model with the difference that it is not required
#### Training:
- batch_size: batch size used during training (int)
- num_pos: The number of positive samples per anchor found in the dataset (int)
- epochs: Number of training epochs (int)
- lr: learning rate (float)
- mode: The type of achitecture (Bi-Enoder|Poly-Encoder) the q and d_model are implemented in (str)
- use_torch_compile: Whether to use the torch 2.0 feature compile to speed up gpu training
- accumulation_steps: How many gradient accumulation steps to take
#### Evaluation:
- eval_freq: How often evaluation is performed, "epoch"= end of epoch, int = after x steps, float = after x percent of epoch (string|int|float)
- eval_accumulation: We accumulate the embeddings of x eval batches and calculate the accuracy metrics of the accumulated batch. (int)
- batch_size: Batch size used during evaluation (int)
- faiss_device: We use the accuracy calculator of pytorch metric learning which makes use of faiss. Run this on cpu to save gpu memory during evaluation. (str)
#### Logging:
- print_freq: How often to log the loss (both printing and wandb). Note that this relates to gradient update steps. Thus if accumulations_steps = 5 and print_freq = 5 then the loss will be logged each 25 steps. (int)
#### CrossBatchMemory:
- use: Whether to use CrossBatchMemory (bool)
- buffer_size: How big the buffer of the cross batch memory is (int)
- warmup: The number of gradient update steps that need to be executed before the buffer is filled (suggested by paper) (int)
#### Wandb:
- use: Whether to use wandb for reporting (bool)
- credential_path: Path to a txt file that contains the wandb api key of the user (str)
- project_name: The project name that the wandb logging related to (str)

