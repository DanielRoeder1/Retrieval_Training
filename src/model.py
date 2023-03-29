import torch
import torch.nn as nn

class BiEncoder(nn.Module):
    def __init__(self, q_model = None, d_model = None):
        super(BiEncoder, self).__init__()
        self.q_model = q_model
        self.d_model = d_model
    def __call__(self, inputs):
        q_inputs, d_inputs = inputs
        q_output = self.q_model(**q_inputs)
        d_output = self.d_model(**d_inputs)
        q_embeds = mean_pooling(q_output, q_inputs['attention_mask'])
        d_embeds = mean_pooling(d_output, d_inputs['attention_mask'])

        return q_embeds, d_embeds

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# ------------------ New Model architectures ------------------ #
from transformers import AutoModel, AutoConfig, PreTrainedModel
import torch.nn.functional as F

class GeneralizedPreTrainedModel(PreTrainedModel):
    """
    Abstract class that allows to load a pretrained model without specifying the model class.
    """
    config_class = AutoConfig
    base_model_prefix = "enc_model"
    def _init_weights(self, module):
        return getattr(self, self.base_model_prefix, self)._init_weights(module)

class PooledEncoder(GeneralizedPreTrainedModel):
    def __init__(self, config, args, enc_model=None):
        super().__init__(config)
        self.enc_model = AutoModel.from_config(config) if enc_model is None else enc_model

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def forward(self, inputs):
        model_output = self.enc_model(**inputs)
        return self.mean_pooling(model_output, inputs['attention_mask'])
    
class PolyEncoder(GeneralizedPreTrainedModel):
    def __init__(self, config, args, enc_model=None):
        super().__init__(config)
        self.enc_model = AutoModel.from_config(config) if enc_model is None else enc_model
        #self.register_parameter('poly_codes', torch.normal(0,1,(args.num_poly_codes, config.hidden_size), requires_grad=True))
        self.poly_codes = nn.Parameter(torch.normal(0,1,(args.num_poly_codes, config.hidden_size)), requires_grad=True)
        self.poly_m = args.num_poly_codes

    def dot_attention(self, q, k, v):
        attn_weights = torch.matmul(q, k.transpose(2, 1)) 
        attn_weights = F.softmax(attn_weights, -1)
        output = torch.matmul(attn_weights, v)
        return output
    
    def forward(self, inputs, pooled_embeds):
        q_embeds = self.enc_model(**inputs)[0] # [bs, length, dim]
        poly_codes = self.poly_codes.expand(q_embeds.shape[0],-1,-1) # [bs, poly_m, dim]
        poly_heads = self.dot_attention(poly_codes,q_embeds, q_embeds) # [bs, poly_m, dim]
        outputs = self.dot_attention(pooled_embeds, poly_heads, poly_heads) # [doc_cnt, q_cnt, dim]
        return outputs