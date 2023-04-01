from utils import get_time,AverageMeter, AverageMeterDict
from loss import CustomAccuracyCalc
from torch import autocast

import faiss
from pytorch_metric_learning.utils.inference import FaissKNN

import torch

class EmbedBuffer:
    def __init__(self, buffer_dim, device):
        self.buffer_size = buffer_dim[0]
        self.buffer = torch.zeros(buffer_dim).to(device)
        self.reset()
        
    def add(self, embeds):
        batch_size = embeds.shape[0]
        tmp = self.max_fill_idx
        self.max_fill_idx += batch_size
        d_idx = torch.arange(tmp, self.max_fill_idx)% self.buffer_size
        self.buffer[d_idx] = embeds
    
    def reset(self):
        self.max_fill_idx = 0 # index of the last filled element
    
    def get(self):
        return self.buffer[:self.max_fill_idx]
    
def cosine_ranking(q_embed, doc_embed, num_pos, i = 0):
    """
    q_embed: (num_query, num_doc, dim)
    doc_embed: (num_doc, dim)
    num_pos: queries per doc 
    Assumes that the first num_pos queries relate to doc 0 ...
    """
    q_docs = q_embed.shape[0] // num_pos
    a = torch.nn.functional.normalize(q_embed, dim = 2)
    b = torch.nn.functional.normalize(doc_embed, dim = 1)
    index_gt = torch.arange(q_docs).repeat_interleave(num_pos) + i *q_docs
    cos_sim = torch.sum(a*b, dim=-1)
    ranking = cos_sim.argsort(descending=True)
    _, rank = torch.where(ranking.to("cpu") == index_gt.unsqueeze(1))
    return rank.float()+1
    

class Evaluator:
    def __init__(self, args, val_loader, device, faiss_device, hidden_size):
        d_buff_size = args.evaluation.eval_accumulation * args.evaluation.batch_size
        q_buff_size = d_buff_size * args.training.num_pos
        self.d_buff = EmbedBuffer((d_buff_size, hidden_size), device)
        self.q_buff = EmbedBuffer((q_buff_size, hidden_size), device)
        self.av_val = AverageMeter()
        self.av_val_acc = AverageMeterDict()   
        # Use inner prodcut instead of default L2 
        custom_knn = FaissKNN(index_init_fn=faiss.IndexFlatIP)
        self.acc_calc = CustomAccuracyCalc(exclude=("AMI","NMI"), device = faiss_device,  knn_func = custom_knn)
        self.val_loader = val_loader
        self.device = device
        self.args = args
    
    def reset(self):
        self.d_buff.reset()
        self.q_buff.reset()
        self.av_val.reset()
        self.av_val_acc.reset()

    def evaluate(self, model):
        print(f"[{get_time()}] [LOG]: Evaluating model")  
        self.reset()
        model.eval()
        for idx, inputs in enumerate(self.val_loader):
            with autocast(device_type='cuda', dtype=torch.float16):
                with torch.no_grad():
                    for input in inputs: input.to(self.device)
                    loss, q_embeds, d_embeds = model(inputs)
                    self.av_val.update(loss.item())
                    self.q_buff.add(q_embeds)
                    self.d_buff.add(d_embeds)
        
            if (idx+1) % self.args.evaluation.eval_accumulation == 0:
                acc_metrics = self.acc_calc.get_acc_wrapper(self.q_buff.get(), self.d_buff.get())
                self.av_val_acc.update(acc_metrics)
        
        return self.av_val.get_avg(), self.av_val_acc.get_avg()
    
    def eval_conditional(self, model):
        print(f"[{get_time()}] [LOG]: Evaluating model")  
        self.reset()
        model.eval()
        iter_loader = iter(self.val_loader)

        for i , inputs in enumerate(self.val_loader):
            _, pool_inputs = inputs
            pool_inputs.to(self.device)
            with autocast(device_type='cuda', dtype=torch.float16):
                with torch.no_grad():
                    pool_emb = model.d_model(pool_inputs)
                    self.d_buff.add(pool_emb)
                    
            if (i+1) % self.args.evaluation.eval_accumulation == 0:
                pool_embed = self.d_buff.get()
                for i in range(self.args.evaluation.eval_accumulation):
                    cond_inputs, pool_inputs = next(iter_loader)
                    cond_inputs.to(self.device)
                    with autocast(device_type='cuda', dtype=torch.float16):
                        with torch.no_grad():
                            cond_emb = model.q_model(cond_inputs, pool_embed)
                            ranks = cosine_ranking(cond_emb, pool_embed, self.args.training.num_pos, i)
                            indices_tuple = model.get_indices_tuple(self.args.evaluation.batch_size,self.args.training.num_pos)
                            cond_loss_embed = cond_emb[:,:self.args.evaluation.batch_size,:].reshape(-1,cond_emb.shape[-1])
                            pool_loss_embed = pool_embed[:self.args.evaluation.batch_size,:]
                            loss = model.loss_func(embeddings= pool_loss_embed, indices_tuple= indices_tuple, ref_emb = cond_loss_embed)
                            self.av_val.update(loss.item())
                            self.av_val_acc.update({"mean_rank_first":ranks.mean().item()})
        return self.av_val.get_avg(), self.av_val_acc.get_avg()