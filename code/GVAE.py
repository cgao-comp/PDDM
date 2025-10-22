import torch
import tqdm
from torch import nn
from torch_geometric.nn import GATConv
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
import torch.optim as optim
import numpy as np

class GVAEv3(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels):
        super(GVAEv3, self).__init__()       
        self.gat1 = GATConv(in_channels, hidden_channels, heads=4, concat=True)
  
        self.gat2 = GATConv(hidden_channels * 4, hidden_channels, heads=4, concat=True)      
        self.fc_mean = nn.Sequential(
            nn.Linear(hidden_channels * 4, latent_channels),
            nn.ReLU(),
            nn.Linear(latent_channels, latent_channels)
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(hidden_channels * 4, latent_channels),
            nn.ReLU(),
            nn.Linear(latent_channels, latent_channels)
        )

        self.feature_trans = nn.Linear(in_channels, 2 * hidden_channels)
        self.attn_trans = nn.Sequential(
            nn.Linear(latent_channels, hidden_channels),
            nn.ReLU()
        )
        self.decode_attn_layer = nn.Linear(2 * hidden_channels, 1)

    def encode(self, x, adj):        
        edge_index = adj.nonzero(as_tuple=False).t().contiguous()        
        h = torch.relu(self.gat1(x, edge_index))        
        h = torch.relu(self.gat2(h, edge_index))       
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
            

    def decode(self, z):
        adj_recon = torch.sigmoid(torch.matmul(z, z.t()))
        return adj_recon

    def decode_attn(self, z):
        adj_recon = torch.zeros(z.shape[0], z.shape[0])
        for i in range(z.shape[0]):
            for j in range(z.shape[0]):
                if i != j:
                    
                    adj_recon[i][j] = torch.sigmoid(self.decode_attn_layer(torch.concat((self.attn_trans(z[i]),
                                                                                         self.attn_trans(z[j])))))
        return adj_recon

    def decode_attn_fast(self, z):
        z_expanded = self.attn_trans(z)
        V = z_expanded.shape[0]
        E = z_expanded.shape[1]       
        z_expanded = z_expanded.unsqueeze(1)  
        z_combined = z_expanded.expand(-1, V, -1)  
        z_combined = torch.cat((z_combined, z_combined.transpose(0, 1)), dim=2)     
        adj_recon = torch.sigmoid(self.decode_attn_layer(z_combined.view(-1, 2 * E))).view(V, V)    
        adj_recon = torch.where(torch.eye(V, device=adj_recon.device) == 1, torch.tensor(0.0, device=adj_recon.device),
                                adj_recon)

        return adj_recon

    def decode_attn_fast_combind_fea(self, z, x):
        z_trans = self.attn_trans(z)
        V = z_trans.shape[0]
        E = z_trans.shape[1]

        X_trans = self.feature_trans(x)
        X_expanded = X_trans.unsqueeze(1).expand(-1, V, -1)
       
        z_expanded = z_trans.unsqueeze(1)  
        z_combined = z_expanded.expand(-1, V, -1)  
        z_combined = torch.cat((z_combined, z_combined.transpose(0, 1)), dim=2)  

        z_combined = 0.5 * z_combined + 0.5 * X_expanded
      
        adj_recon = torch.sigmoid(self.decode_attn_layer(z_combined.view(-1, 2 * E))).view(V, V)
        
        adj_recon = torch.where(torch.eye(V, device=adj_recon.device) == 1, torch.tensor(0.0, device=adj_recon.device),
                                adj_recon)

        return adj_recon

    def forward(self, x, adj):
        mean, logvar = self.encode(x, adj)
        z = self.reparameterize(mean, logvar)
        adj_recon = self.decode_attn_fast(z)
        return adj_recon, mean, logvar


class RandomGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(RandomGraphDataset, self).__init__(root, transform, pre_transform)
        self.data_list = self.generate_random_graphs()
        self.data, self.slices = self.collate(self.data_list)

    def generate_random_graphs(self):
        data_list = []
        for _ in range(NUM_GRAPHS):
            
            A = np.random.rand(NUM_NODES, NUM_NODES)
            A = (A + A.T) / 2  
            A[A < 0.9] = 0     
            A[A >= 0.9] = 1
            np.fill_diagonal(A, 0)  

            X = np.random.rand(NUM_NODES, NUM_FEATURES)

            x = torch.tensor(X, dtype=torch.float)
            adj = torch.tensor(A, dtype=torch.float)
            data = Data(x=x)
            data.adj = adj
            data.num_nodes = NUM_NODES  
            data_list.append(data)
        return data_list
