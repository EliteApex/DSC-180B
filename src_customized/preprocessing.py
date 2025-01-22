from sklearn.model_selection import train_test_split
import numpy as np

def prepare_train_test_data(adj_matrix, test_size=0.2):
    num_nodes = adj_matrix.shape[0]
    positive_edges = np.array(np.nonzero(np.triu(adj_matrix))).T  # Extract upper triangular edges
    all_possible_edges = np.array([(i, j) for i in range(num_nodes) for j in range(i)])
    negative_edges = np.array(
        [edge for edge in all_possible_edges if adj_matrix[edge[0], edge[1]] == 0]
    )

    # Sample negatives to match the number of positives
    negative_edges = negative_edges[np.random.choice(len(negative_edges), len(positive_edges), replace=False)]

    # Combine positive and negative samples
    edges = np.vstack([positive_edges, negative_edges])
    labels = np.hstack([np.ones(len(positive_edges)), np.zeros(len(negative_edges))])

    # Train-test split
    edges_train, edges_test, labels_train, labels_test = train_test_split(edges, labels, test_size=test_size)
    return edges_train, edges_test, labels_train, labels_test