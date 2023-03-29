from utils import get_time,AverageMeter, AverageMeterDict
from loss import CustomAccuracyCalc
from torch import autocast

import torch
class EmbedBuffer:
    def __init__(self, buffer_size, embedding_size, device):
        self.buffer_size = buffer_size
        self.buffer = torch.zeros((self.buffer_size, embedding_size)).to(device)
        
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
    

class Evaluator:
    def __init__(self, args, val_loader, device, faiss_device, hidden_size):
        d_buff_size = args.evaluation.eval_accumulation * args.evaluation.batch_size
        q_buff_size = d_buff_size * args.training.num_pos
        self.d_buff = EmbedBuffer(d_buff_size, hidden_size, device)
        self.q_buff = EmbedBuffer(q_buff_size, hidden_size, device)
        self.av_val = AverageMeter()
        self.av_val_acc = AverageMeterDict()    
        self.acc_calc = CustomAccuracyCalc(exclude=("AMI","NMI"), device = faiss_device)
        self.val_loader = val_loader
        self.devie = device
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