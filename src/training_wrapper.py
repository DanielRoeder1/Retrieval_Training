import torch
from test_loss import CustomNTXentLoss
import numpy as np

class PolyEncoderWrapper(torch.nn.Module):
    def __init__(self, pool_enc, cond_enc, args):
        super().__init__()
        self.pool_enc = pool_enc
        self.cond_enc = cond_enc
        self.loss_func = CustomNTXentLoss(temperature=0.07)

        self.indices_tuples = self.get_indices_tuple(args.training.batch_size, args.training.num_pos)
    
    def forward(self, inputs):
        q_inputs, d_inputs = inputs
        d_embeds = self.pool_enc(d_inputs)
        q_embeds = self.cond_enc(q_inputs, d_embeds)
        q_embeds = q_embeds.reshape(-1, q_embeds.shape[-1])
        loss = self.loss_func(embeddings = d_embeds,ref_emb = q_embeds,  indices_tuple = self.indices_tuples)
        return loss
        
    def get_indices_tuple(self, num_doc, num_pos):
        """
        Creating custom indces_tuples for pytorch_metric_learning loss function
        Used for building the positive and negative pairs
        """
        num_q = num_doc * num_pos
        doc_pos = torch.arange(num_doc).repeat_interleave(num_pos)
        pos_id = doc_pos + torch.arange(num_q) * num_doc
        neg_id = torch.from_numpy(np.delete(np.arange(num_q*num_doc),pos_id)).long()
        doc_neg = neg_id % num_doc
        return doc_pos, pos_id, doc_neg, neg_id
