from pytorch_metric_learning.losses import NTXentLoss
from pytorch_metric_learning.utils import common_functions as c_f

class CustomNTXentLoss(NTXentLoss):
    """
    Overwriting the NTXentLoss loss calculation to allow for a custom n_per_p 
    (i.e. selecting which negatives belong to which positives)
    _comput_loss copied from super, only n_per_p is changed
    """
    def __init__(self, temperature):
        super().__init__(temperature)
    
    def _compute_loss(self, pos_pairs, neg_pairs, indices_tuple):
            print("%%%%% USING CUSTOM LOSS %%%%%")
            a1, p, a2, _ = indices_tuple
            print(f"{pos_pairs=}")
            print(f"{neg_pairs=}")
            
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
                ###################
                print(f"{n_per_p=}")
                neg_pairs = neg_pairs * n_per_p
                neg_pairs[n_per_p == 0] = c_f.neg_inf(dtype)
                print(f"{neg_pairs.shape=}")
                max_val = torch.max(
                    pos_pairs, torch.max(neg_pairs, dim=1, keepdim=True)[0]
                ).detach()
                numerator = torch.exp(pos_pairs - max_val).squeeze(1)

                denominator = torch.sum(torch.exp(neg_pairs - max_val), dim=1) + numerator
                print(f"{numerator=}")
                print(f"{denominator=}")
                log_exp = torch.log((numerator / denominator) + c_f.small_val(dtype))
                print(f"{log_exp=}")
                return {
                    "loss": {
                        "losses": -log_exp,
                        "indices": (a1, p),
                        "reduction_type": "pos_pair",
                    }
                }
            return self.zero_losses()
