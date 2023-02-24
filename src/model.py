import torch
import torch.nn as nn

class BiEncoder(nn.Module):
    def __init__(self, q_model, d_model):
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