import torch
from pytorch_metric_learning import losses

class CrossBatchMemoryWrapper():
    """
    Allows CrossBatchMemory to be called like SelfSupervisedLoss
    """
    def __init__(self, embedding_size,device,  memory_size=400):
        self.loss_fn = losses.CrossBatchMemory(losses.NTXentLoss(temperature=0.07), embedding_size, memory_size=memory_size, miner=None)
        self.prev_max_label = -1
        self.device = device

    
    def get_labels(self, batch_size):
        labels = torch.arange(0, batch_size)
        labels = torch.cat((labels, labels)).to(self.device)
        labels += self.prev_max_label+1
        enqueue_mask = torch.zeros(len(labels)).bool()
        enqueue_mask[batch_size:] = True
        return labels , enqueue_mask
    
        
    def __call__(self, q_embeds, d_embeds):
        all_enc = torch.cat([q_embeds, d_embeds], dim=0)
        labels, enqueue_mask = self.get_labels(q_embeds.shape[0])
        self.prev_max_label = labels.max()
        loss = self.loss_fn(all_enc, labels, enqueue_mask)
        return loss