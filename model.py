import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, VGAE, GATConv
import torch.nn as nn
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, average_precision_score



class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, 2 * out_channels)  # Extra message passing
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))  # Additional message passing
        x = self.dropout(x)
        mu = self.conv_mu(x, edge_index)
        logstd = self.conv_logstd(x, edge_index)
        return mu, logstd
    

def train(model, data, optimizer, patience=60, delta=0.001, checkpoint_path="best_model.pth"):
    best_auc = float('-inf')
    counter = 0
    num_epochs = 200
    
    auc_values = []
    ap_values = []
    precision_values = []
    recall_values = []
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        z = model.encode(data.x, data.train_pos_edge_index)
        loss = model.recon_loss(z, data.train_pos_edge_index)
        loss += (1 / data.num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()

        # Compute validation AUC, AP, Precision, Recall
        auc_score, ap_score, precision, recall = test(model, data)
        auc_values.append(auc_score)
        ap_values.append(ap_score)
        precision_values.append(precision)
        recall_values.append(recall)

        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss:.4f}, AUC: {auc_score:.4f}, AP: {ap_score:.4f}")

        # Early stopping logic
        if auc_score > best_auc + delta:
            best_auc = auc_score
            counter = 0
            torch.save(model.state_dict(), checkpoint_path)  # Save best model
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}. Best AUC: {best_auc:.4f}")
                break

    # Load the best model
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"Best model restored with AUC: {best_auc:.4f}")

    # Store metrics in a dictionary
    metrics_dict = {
        "AUC": auc_values,
        "AP": ap_values,
        "Precision": precision_values,
        "Recall": recall_values
    }
    
    return metrics_dict


def test(model, data):
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.train_pos_edge_index)

        # Compute predicted scores
        pos_edge_scores = model.decoder(z, data.test_pos_edge_index).sigmoid().cpu().numpy()
        neg_edge_scores = model.decoder(z, data.test_neg_edge_index).sigmoid().cpu().numpy()

        # Ground truth labels (1 for positive edges, 0 for negative edges)
        y_true = np.concatenate([np.ones(pos_edge_scores.shape[0]), np.zeros(neg_edge_scores.shape[0])])
        y_scores = np.concatenate([pos_edge_scores, neg_edge_scores])

        # Compute AUC and AP
        auc_score = roc_auc_score(y_true, y_scores)
        ap_score = average_precision_score(y_true, y_scores)

        # Compute Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_scores)

    return auc_score, ap_score, precision, recall