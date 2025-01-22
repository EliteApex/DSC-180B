import torch
from model import BaselineGAE
from preprocessing import prepare_train_test_data

def train_baseline_gae(adj_matrix, latent_dim=16, num_epochs=100, lr=0.01):
    num_nodes = adj_matrix.shape[0]
    model = BaselineGAE(num_nodes, latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCELoss()

    # Prepare training and testing data
    edges_train, edges_test, labels_train, labels_test = prepare_train_test_data(adj_matrix)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Reconstruct adjacency matrix
        adj_reconstructed, _ = model(torch.tensor(adj_matrix, dtype=torch.float32))

        # Compute loss using training edges
        train_loss = 0
        for edge, label in zip(edges_train, labels_train):
            i, j = edge
            pred = adj_reconstructed[i, j]
            train_loss += loss_fn(pred, torch.tensor(label, dtype=torch.float32))

        train_loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {train_loss.item()}")