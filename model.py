import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import GCNConv
from tqdm import tqdm
import math
from GCN_EN import self_loop_attention_GCN

def extract(tensor, t, shape):
    tensor_t = tensor[t]     
    return tensor_t
import torch
import math

class TimeEmbedding(nn.Module):
    def __init__(self, embed_size):
        super(TimeEmbedding, self).__init__()
        self.embed_size = embed_size

    def forward(self, t):
        device = t.device
        half_dim = self.embed_size // 2
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embed_size % 2 == 1:
            emb = F.pad(emb, (0, 1), "constant", 0)
        return emb  


def get_time_embedding(t, dimension, device='cpu'):  
    position = torch.tensor([t], dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dimension, 2, device=device) * -(math.log(10000.0) / dimension))
    pos_embedding = torch.zeros((1, dimension), device=device)
    
    pos_embedding[:, 0::2] = torch.sin(position * div_term)  
    pos_embedding[:, 1::2] = torch.cos(position * div_term)  
    return pos_embedding
class DiffusionModule_GAT(nn.Module):    
    def __init__(self, feature_hidden_dim=14, feature_dim=14, self_head=4, GAT=True, PE=True):   
        super(DiffusionModule_GAT, self).__init__()
        self.feature_hidden_dim = feature_hidden_dim
        self.PE = PE
        self.loop_att_gcn = self_loop_attention_GCN(feature_dim, self_head)
        self.de_MLP = nn.Linear(feature_dim, 2)


    def forward(self, timestamp, graph_topo, features):         
        if self.PE:
            position_em_hidden = get_time_embedding(timestamp, self.feature_hidden_dim)  
        enhanced_fea = self.loop_att_gcn(graph_topo, features)
        decoder_output = self.de_MLP(enhanced_fea)

        return decoder_output


class DiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        self.model = model
        self.T = T
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).float())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        self.register_buffer('glo_var_coeff',  ((torch.arange(1, T + 1) - 1) * alphas_bar_prev) / (
                    (torch.arange(1, T + 1) - 1) * alphas_bar_prev + 1))

        self.register_buffer('main_xt', ((torch.arange(1, T + 1) - 1) * alphas_bar_prev + 1) / (
                    torch.sqrt(alphas) * (torch.arange(1, T + 1) - 1) * alphas_bar_prev))
       
        self.register_buffer('additional_term', (
                (torch.sqrt(alphas) - 1) / torch.sqrt(alphas) +
                (torch.sqrt(alphas) - 1) / (torch.sqrt(alphas) * (torch.arange(1, T + 1) - 1) * alphas_bar_prev)
        ))
        self.register_buffer('sigma_coeff',
                             torch.sqrt(alphas_bar_prev) / (torch.arange(1, T + 1) - 1) / alphas_bar_prev * torch.sqrt(
                                 1 - alphas_bar[1:]))

    def predict_xt_prev_mean_from_pred_noise(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                self.coeff1[t] * x_t - self.coeff2[t] * eps
        )

    def p_mean_variance(self, x_t, time_step, prior):
        xt_prev_mean = self.main_xt[time_step] * x_t
        xt_prev_mean += self.additional_term[time_step] * prior
        
        sigma_t_prev = self.sigma_coeff[time_step] * torch.sqrt(1 - self.alphas_bar[time_step])

        return xt_prev_mean, sigma_t_prev

    def forward(self, x_T, graph_var, prior, CR_RW_list=None): 
        self.T = prior.shape[0]
        x_t = x_T
        for t_idx, time_step in enumerate(reversed(range(self.T))):
            time_step = torch.tensor([time_step])
            if CR_RW_list != None:
                
                x_t = (x_t + CR_RW_list[t_idx]) / 2
            mean, sigma_t_prev = self.p_mean_variance(x_t, time_step, prior)
            if time_step > 0:
                mean += sigma_t_prev * graph_var[time_step]
                glo_noise_var = time_step /prior.shape[0] / prior.shape[0] * self.glo_var_coeff
            else:
                glo_noise_var = 0
            noise = torch.randn_like(x_t)
            x_t = mean + torch.sqrt(glo_noise_var) * noise
        x_0 = x_t
        return torch.sigmoid(x_0)
