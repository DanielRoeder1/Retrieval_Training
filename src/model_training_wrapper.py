import torch
from test_loss import CustomNTXentLoss
import numpy as np

from loss import CrossBatchMemoryWrapper, LabelLossWrapper
from pytorch_metric_learning import losses 

class PolyEncoderWrapper(torch.nn.Module):
    def __init__(self, q_model, d_model, args):
        super().__init__()
        self.q_model = q_model
        self.d_model = d_model
        self.loss_func = CustomNTXentLoss(temperature=0.07)
        self.eval_type = "eval_conditional"

        self.indices_tuples = self.get_indices_tuple(args.training.batch_size, args.training.num_pos)
    
    def forward(self, inputs):
        q_inputs, d_inputs = inputs
        d_embeds = self.d_model(d_inputs)
        q_embeds = self.q_model(q_inputs, d_embeds)
        q_embeds = q_embeds.reshape(-1, q_embeds.shape[-1])
        loss = self.loss_func(embeddings = d_embeds,ref_emb = q_embeds,  indices_tuple = self.indices_tuples)
        return loss if self.training else loss, q_embeds, d_embeds
        
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
    
    
class BiEncoderWrapper(torch.nn.Module):
    def __init__(self, q_model, d_model, args):
        super().__init__()
        self.q_model = q_model
        self.d_model = d_model
        self.eval_type = "evaluate"

        if args.cross_batch_memory.use:
            self.loss_func = CrossBatchMemoryWrapper(q_model.config.hidden_size, device = self.q_model.device, memory_size=args.cross_batch_memory.buffer_size, warmup = args.cross_batch_memory.warmup, acc_steps= args.training.accumulation_steps, num_pos =args.training.num_pos)
        else:
            self.loss_func = LabelLossWrapper(losses.NTXentLoss(temperature = 0.07), num_pos = args.training.num_pos, device = self.q_model.device)

    def __call__(self, inputs):
        q_inputs, d_inputs = inputs
        q_embeds = self.q_model(**q_inputs)
        d_embeds = self.d_model(**d_inputs)
        loss = self.loss_func(q_embeds, d_embeds)
        return loss if self.training else loss, q_embeds, d_embeds

