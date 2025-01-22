import numpy as np
import pandas as pd

def load_data(file_path):
    # Load the data
    data = pd.read_csv(file_path)
    data.columns = ['protein1', 'protein2', 'combined_score']

    # Build adjacency matrix
    proteins = set(data['protein1']).union(set(data['protein2']))
    protein_to_idx = {protein: idx for idx, protein in enumerate(proteins)}

    num_nodes = len(proteins)
    adj_matrix = np.zeros((num_nodes, num_nodes))

    for _, row in data.iterrows():
        idx1 = protein_to_idx[row['protein1']]
        idx2 = protein_to_idx[row['protein2']]
        adj_matrix[idx1, idx2] = 1  # Positive links
        adj_matrix[idx2, idx1] = 1  # Symmetric for undirected graph

    return adj_matrix, protein_to_idx