import torch
from pytorch_metric_learning import losses
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

class CrossBatchMemoryWrapper():
    """
    Allows CrossBatchMemory to be called like SelfSupervisedLoss
    """
    def __init__(self, embedding_size,device,warmup,acc_steps, num_pos, memory_size=400):
        self.loss_fn = losses.CrossBatchMemory(losses.NTXentLoss(temperature=0.07), embedding_size, memory_size=memory_size, miner=None)
        self.prev_max_label = -1
        self.device = device

        # Take gradient accumulation into account:
        # While we are within the warmup period, build the buffer for acc_steps as no model updates are done
        self.warmup = warmup *acc_steps
        self.iteration = 1
        self.accumulation_steps = acc_steps
        self.num_pos = num_pos

    # TODO decide whether to save query or documens in cross batch memory
    def get_labels(self, batch_size):
        labels = torch.arange(0, batch_size)
        labels = torch.cat((labels, labels.repeat_interleave(self.num_pos))).to(self.device)
        labels += self.prev_max_label+1
        enqueue_mask = torch.zeros(len(labels)).bool()
        enqueue_mask[batch_size:] = True
        return labels , enqueue_mask
    
        
    def __call__(self, q_embeds, d_embeds):
        all_enc = torch.cat([d_embeds,q_embeds], dim=0)
        labels, enqueue_mask = self.get_labels(d_embeds.shape[0])
        self.prev_max_label = labels.max()
        loss = self.loss_fn(embeddings = all_enc, labels = labels, enqueue_mask = enqueue_mask)
        if self.iteration < self.warmup and self.iteration % self.accumulation_steps == 0:
            self.loss_fn.reset_queue()
        self.iteration += 1
        return loss
    

class LabelLossWrapper:
    """
    Similar to SelfSupervisedLoss but allows for multiple positive samples per document
    Note: This expects a constant number of positive samples per document
    """
    def __init__(self, loss_fn, num_pos,device):
        self.loss = loss_fn
        self.num_pos = num_pos
        self.device = device
    
    def __call__(self, q_embeds, d_embeds):
        
        d_labels = torch.arange(0, d_embeds.shape[0]).to(self.device)
        q_lables = d_labels.repeat_interleave(self.num_pos).to(self.device)

        return self.loss(embeddings = d_embeds, labels = d_labels, ref_emb =q_embeds, ref_labels = q_lables)

    
class CustomAccuracyCalc(AccuracyCalculator):
    """
    Added function for calculating the average rank of the first positive example
    """
    def calculate_avg_rank_first(self, knn_labels, query_labels, **kwargs):
        q_labels =  query_labels[:, None]
        is_same_label= torch.eq(q_labels, knn_labels)*1
        zero_remove_mask = is_same_label.sum(-1) == 0
        indices = torch.argmax(is_same_label, 1) + 1.0
        return torch.mean(indices[~zero_remove_mask]).item()

    def requires_knn(self):
        return super().requires_knn() + ["avg_rank_first"] 
    
    def get_acc_wrapper(self, q_embeds, d_embeds):
        d_labels = torch.arange(0, d_embeds.shape[0]).to(self.device)
        q_labels = d_labels.repeat_interleave(q_embeds.shape[0]//d_embeds.shape[0]).to(self.device)

        return super().get_accuracy(query = q_embeds, reference = d_embeds,query_labels =  q_labels, reference_labels = d_labels)