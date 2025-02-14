import numpy as np
import pandas as pd 
import torch
import torch.nn.functional as F
from torch_geometric.nn import VGAE
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import torch.nn as nn
from torch_geometric.utils import from_scipy_sparse_matrix, train_test_split_edges, add_self_loops
from scipy.sparse import coo_matrix
from sklearn.preprocessing import StandardScaler
from model import GCNEncoder, train, test
import argparse
from sklearn.preprocessing import StandardScaler


# Argument Parser
parser = argparse.ArgumentParser(description="Run VGAE on different versions of protein data")
parser.add_argument("--version", type=int, choices=[1, 2, 3], required=True,
                    help="Select version: 1 for Combined Features, 2 for RNA Features, 3 for Protein Expression Features")
args = parser.parse_args()

# Load Data
ppi_df = pd.read_csv('9606.protein.links.v12.0.txt', delimiter=' ')
rna_pca_df = pd.read_csv('../Data/PPI_RNA_seq_10PCs.csv')
pe_pca_df = pd.read_csv('../Data/PPI_protein_expression_10PCs.csv')
combined_feature_df = pd.read_csv('../Data/PPI_RNA_Protein_combined.csv')


# Select Version
if args.version == 1:
    print("Running Version 1: Combined Features (RNA + Protein Expression)")
    valid_nodes = set(combined_feature_df['Protein'])
elif args.version == 2:
    print("Running Version 2: RNA Features Only")
    valid_nodes = set(rna_pca_df['Protein'])
elif args.version == 3:
    print("Running Version 3: Protein Expression Features Only")
    valid_nodes = set(pe_pca_df['Protein'])


# Filter PPI Data
ppi_df_filtered = ppi_df[ppi_df['protein1'].isin(valid_nodes) & ppi_df['protein2'].isin(valid_nodes)]
filtered_nodes = pd.concat([ppi_df_filtered['protein1'], ppi_df_filtered['protein2']]).unique()
node_to_idx = {node: i for i, node in enumerate(filtered_nodes)}

ppi_df_filtered['protein1_idx'] = ppi_df_filtered['protein1'].map(node_to_idx)
ppi_df_filtered['protein2_idx'] = ppi_df_filtered['protein2'].map(node_to_idx)


scaler = StandardScaler()
ppi_df_filtered['standardized_score'] = scaler.fit_transform(ppi_df_filtered[['combined_score']])

adj_matrix = coo_matrix(
    (ppi_df_filtered['standardized_score'], (ppi_df_filtered['protein1_idx'], ppi_df_filtered['protein2_idx'])),
    shape=(len(filtered_nodes), len(filtered_nodes)))

# Prepare Feature Matrix
if args.version == 1:
    combined_feature_df['node_idx'] = combined_feature_df['Protein'].map(node_to_idx)
    feature_df_filtered = combined_feature_df.dropna(subset=['node_idx']).sort_values(by='node_idx')
elif args.version == 2:
    rna_pca_df['node_idx'] = rna_pca_df['Protein'].map(node_to_idx)
    feature_df_filtered = rna_pca_df.dropna(subset=['node_idx']).sort_values(by='node_idx')
elif args.version == 3:
    pe_pca_df['node_idx'] = pe_pca_df['Protein'].map(node_to_idx)
    feature_df_filtered = pe_pca_df.dropna(subset=['node_idx']).sort_values(by='node_idx')

feature_matrix = feature_df_filtered.iloc[:, 1:-1].values  # Exclude 'Protein' and 'node_idx'
feature_matrix = StandardScaler().fit_transform(feature_matrix)

assert adj_matrix.shape[0] == feature_matrix.shape[0]

# Convert to PyG Data Format
edge_index, edge_attr = from_scipy_sparse_matrix(adj_matrix)
data = Data(edge_index=edge_index, edge_attr=edge_attr, x=torch.tensor(feature_matrix, dtype=torch.float))

# Split Edges
data = train_test_split_edges(data, val_ratio=0.2, test_ratio=0.2)
data.train_pos_edge_index, _ = add_self_loops(data.train_pos_edge_index)

# Define Model
in_channels = data.x.shape[1]  
out_channels = 32
model = VGAE(GCNEncoder(in_channels, out_channels))
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

# Train Model
metrics_dict = train(model, data, optimizer)



# Convert metrics to NumPy arrays
auc_values = np.array(metrics_dict["AUC"])  # AUC values per epoch
ap_values = np.array(metrics_dict["AP"])  # AP values per epoch

# Precision and Recall are lists of NumPy arrays (different lengths)
precision_values = np.array(metrics_dict["Precision"], dtype=object)  # Store as object array
recall_values = np.array(metrics_dict["Recall"], dtype=object)  # Store as object array

# Determine filename based on version
if args.version == 1:
    metrics_filename = "metrics_combined.npz"
elif args.version == 2:
    metrics_filename = "metrics_rna.npz"
elif args.version == 3:
    metrics_filename = "metrics_pe.npz"

# Save everything in a single npz file with `allow_pickle=True`

#use savez instead of compressed 
# save_txt
np.savez_compressed(metrics_filename, 
                    auc_values=auc_values, 
                    ap_values=ap_values, 
                    precision_values=precision_values, 
                    recall_values=recall_values, 
                    allow_pickle=True)

print(f"All metrics saved to {metrics_filename}")
