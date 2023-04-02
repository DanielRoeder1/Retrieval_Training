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


from pytorch_metric_learning.losses import NTXentLoss
from pytorch_metric_learning.utils import common_functions as c_f
import torch 

class CustomNTXentLoss(NTXentLoss):
    """
    Overwriting the NTXentLoss loss calculation to allow for a custom n_per_p 
    (i.e. selecting which negatives belong to which positives)
    _comput_loss copied from super, only n_per_p is changed
    """
    def __init__(self, temperature):
        super().__init__(temperature)
    
    def _compute_loss(self, pos_pairs, neg_pairs, indices_tuple):
            #print("%%%%% USING CUSTOM LOSS %%%%%")
            a1, p, a2, _ = indices_tuple
            #print(f"{pos_pairs=}")
            #print(f"{neg_pairs=}")
            
            if len(a1) > 0 and len(a2) > 0:
                dtype = neg_pairs.dtype
                # if dealing with actual distances, use negative distances
                if not self.distance.is_inverted:
                    pos_pairs = -pos_pairs
                    neg_pairs = -neg_pairs

                pos_pairs = pos_pairs.unsqueeze(1) / self.temperature
                neg_pairs = neg_pairs / self.temperature
                # We set a custom n_per_p which decides which negative pairs are mixed with the positives
                #n_per_p = c_f.to_dtype(a2.unsqueeze(0) == a1.unsqueeze(1), dtype=dtype)
                num_pos_pairs = pos_pairs.shape[0]
                num_neg_pairs = neg_pairs.shape[0]
                neg_per_pos = num_neg_pairs // num_pos_pairs
                n_per_p = torch.zeros((num_pos_pairs,num_neg_pairs))
                i0 = torch.arange(num_pos_pairs).repeat_interleave(neg_per_pos)
                i1 = torch.arange(num_neg_pairs)
                n_per_p[i0,i1] = 1
                
                n_per_p = n_per_p.to(torch.device("cuda:0"))
                ###################
                #print(f"{n_per_p=}")
                neg_pairs = neg_pairs * n_per_p
                neg_pairs[n_per_p == 0] = c_f.neg_inf(dtype)
                #print(f"{neg_pairs.shape=}")
                max_val = torch.max(
                    pos_pairs, torch.max(neg_pairs, dim=1, keepdim=True)[0]
                ).detach()
                numerator = torch.exp(pos_pairs - max_val).squeeze(1)

                denominator = torch.sum(torch.exp(neg_pairs - max_val), dim=1) + numerator
                #print(f"{numerator=}")
                #print(f"{denominator=}")
                log_exp = torch.log((numerator / denominator) + c_f.small_val(dtype))
                #print(f"{log_exp=}")
                return {
                    "loss": {
                        "losses": -log_exp,
                        "indices": (a1, p),
                        "reduction_type": "pos_pair",
                    }
                }
            return self.zero_losses()