# Retrieval training pipeline

![alt text](https://github.com/DanielRoeder1/RAG_thesis/blob/main/visualization/thesis_diagram.drawio.png?raw=true)

Modular training pipeline for Bi-Encoder and Poly-Encoder fully connected to Hugginface + using loss from PytorchMetricLearning.


Features:
- Contrastive loss adapted to PolyEncoder (NtXentLoss)
- Wandb logging
- Code for creating self-supervised retrieval dataset from KILT-Wiki dump
- Mixed precision training & gradient accumulation & CrossBatchMemory (Bi-Encoder)

ToDos:
- Implement traditional CrossEntropy loss for models
- Onboard datasets
- CrossBatchMemory for Poly-Encoder
- Support torch.compile() (currently seems to slow down training)

Usage:
- Adapt parameters in config.yaml and run train.py with --config_path set. 
(if not set config_colab from configs folder is used by default)
