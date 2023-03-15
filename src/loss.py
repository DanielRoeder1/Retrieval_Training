import torch
from pytorch_metric_learning import losses
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

class CrossBatchMemoryWrapper():
    """
    Allows CrossBatchMemory to be called like SelfSupervisedLoss
    """
    def __init__(self, embedding_size,device,warmup,acc_steps, num_pos, memory_size):
        # As gradient accumulation results in the model weights only being update every acc_steps
        # we can set the buffer size to be acc_steps thus profiting from additional samples during warmup
        # After warmup, the buffer size is set to the intended buffer size
        self.loss_fn = losses.CrossBatchMemory(losses.NTXentLoss(temperature=0.07), embedding_size, memory_size=acc_steps, miner=None)
        self.prev_max_label = -1
        self.device = device

        # Take gradient accumulation into account:
        self.warmup = warmup *acc_steps
        self.iteration = 1
        self.accumulation_steps = acc_steps
        self.num_pos = num_pos
        self.orig_memory_size = memory_size

    # We are saving the queries in the cross batch buffer 
    # CrossBatchMemory sets the buffere as reference -> pos/neg samples
    # The non buffered batch is set as embeddings -> anchors
    def get_labels(self, batch_size):
        labels = torch.arange(batch_size)
        labels = torch.cat((labels, labels.repeat_interleave(self.num_pos))).to(self.device)
        labels += self.prev_max_label+1
        enqueue_mask = torch.zeros(len(labels)).bool()
        enqueue_mask[batch_size:] = True
        return labels , enqueue_mask
    
    def __call__(self, q_embeds, d_embeds):
        if self.iteration == self.warmup:
            # Set buffer size to intended buffer_size
            self.loss_fn.memory_size = self.orig_memory_size
            self.loss_fn.reset_queue()
        self.iteration += 1

        all_enc = torch.cat([d_embeds,q_embeds], dim=0)
        labels, enqueue_mask = self.get_labels(d_embeds.shape[0])
        self.prev_max_label = labels.max()
        loss = self.loss_fn(embeddings = all_enc, labels = labels, enqueue_mask = enqueue_mask)
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
    

class EmbedBuffer:
    def __init__(self, buffer_size, embedding_size, device, num_pos, batch_size):
        self.buffer_size = buffer_size * batch_size
        self.buffer_q = torch.zeros((self.buffer_size * num_pos, embedding_size)).to(device)
        self.buffer_d = torch.zeros((self.buffer_size, embedding_size)).to(device)
        self.num_pos = num_pos
    def add(self, q_embed, d_embed, index):
        batch_size = d_embed.shape[0]
        tmp_d = index * batch_size
        tmp_q = tmp_d * self.num_pos

        d_idx = torch.arange(tmp_d, tmp_d + batch_size)% self.buffer_size
        q_idx = torch.arange(tmp_q, tmp_q + batch_size * self.num_pos)% (self.buffer_size*self.num_pos)
        self.buffer_q[q_idx] = q_embed
        self.buffer_d[d_idx] = d_embed