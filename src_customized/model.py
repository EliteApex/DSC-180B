import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineGAE(nn.Module):
    def __init__(self, num_nodes, latent_dim):
        super(BaselineGAE, self).__init__()
        self.encoder = nn.Linear(num_nodes, latent_dim)

    def forward(self, adj_matrix):
        # Encoder: Latent embeddings
        Z = F.relu(self.encoder(adj_matrix))
        # Decoder: Reconstruct adjacency matrix
        adj_reconstructed = torch.sigmoid(torch.mm(Z, Z.t()))
        return adj_reconstructed, Z