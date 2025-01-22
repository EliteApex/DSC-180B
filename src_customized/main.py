from Input import load_data
from trainGCN import train_baseline_gae

# Load data
adj_matrix, protein_to_idx = load_data('/home/xgui/private/Q2_GAE/filtered_PPI.csv')
print("Loaded data!")

# Train baseline GAE
train_baseline_gae(adj_matrix, latent_dim=16, num_epochs=10, lr=0.01)