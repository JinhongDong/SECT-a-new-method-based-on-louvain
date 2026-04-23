# gcn.py
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score
)
from sklearn.cluster import KMeans
import igraph as ig

# ---------------------------------------------------------------------------#
# 0. Load network
# ---------------------------------------------------------------------------#

def load_graph_with_attributes(node_file_path, edge_file_path):
    G = nx.Graph()
    with open(node_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                node_id, comm = parts
                G.add_node(int(node_id), actual_community=int(comm))
    with open(edge_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                n1, n2 = parts
                G.add_edge(int(n1), int(n2))
    return G

# ---------------------------------------------------------------------------#
# 1. GCN function
# ---------------------------------------------------------------------------#

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout_rate = dropout_rate
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def prepare_pyg_data(nx_graph, node_order):
    """Convert NetworkX graph to PyTorch Geometric data"""
    # Node features (using node degree as feature)
    node_degrees = torch.tensor([d for _, d in nx_graph.degree(node_order)], dtype=torch.float).view(-1, 1)
    
    # Edge index
    edge_list = []
    for edge in nx_graph.edges():
        u_idx = node_order.index(edge[0])
        v_idx = node_order.index(edge[1])
        edge_list.append([u_idx, v_idx])
        edge_list.append([v_idx, u_idx])  # Undirected graph
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # Get true labels
    true_labels = [nx_graph.nodes[n]['actual_community'] for n in node_order]
    y = torch.tensor(true_labels, dtype=torch.long)
    
    return Data(x=node_degrees, edge_index=edge_index, y=y)

def best_map(true_labels, pred_labels):
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    
    # Ensure labels are non-negative integers
    true_labels = true_labels.astype(int)
    pred_labels = pred_labels.astype(int)
    
    D = max(pred_labels.max(), true_labels.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    
    for i in range(pred_labels.size):
        w[pred_labels[i], true_labels[i]] += 1
    
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    mapping = {int(row): int(col) for row, col in zip(row_ind, col_ind)}
    return np.array([mapping[label] for label in pred_labels])

# ---------------------------------------------------------------------------#
# 3. Example entry
# ---------------------------------------------------------------------------#

if __name__ == "__main__":

    #tree
    file_name = "tree"  
    input_dir = os.path.join('..', 'norm_dataset', file_name)
    node_file_path = os.path.join(input_dir, f'{file_name}_nodes.txt')
    edge_file_path = os.path.join(input_dir, f'{file_name}_edges.txt')
    gcn_hidden_channels = 16
    gcn_dropout_rate = 0.2
    gcn_out_channels = 5
    training_epochs = 100
    learning_rate = 0.001
    optimizer_type = "Adam"
    loss_function = "MSE"
    n_clusters = None  
    kmeans_random_state = 42
    kmeans_n_init = 10
    in_channels = 1

    # lol
    # file_name = "lol" 
    # input_dir = os.path.join('..', 'norm_dataset', file_name)
    # node_file_path = os.path.join(input_dir, f'{file_name}_nodes.txt')
    # edge_file_path = os.path.join(input_dir, f'{file_name}_edges.txt')
    # gcn_hidden_channels = 16
    # gcn_dropout_rate = 0.2
    # gcn_out_channels = 10
    # training_epochs = 100
    # learning_rate = 0.001
    # optimizer_type = "Adam"
    # loss_function = "CrossEntropy"
    # n_clusters = None  
    # kmeans_random_state = 42
    # kmeans_n_init = 10
    # in_channels = 1

    # LFR_base
    # file_name = "LFR_base" 
    # input_dir = os.path.join('..', 'norm_dataset', file_name)
    # node_file_path = os.path.join(input_dir, f'{file_name}_nodes.txt')
    # edge_file_path = os.path.join(input_dir, f'{file_name}_edges.txt')
    # gcn_hidden_channels = 16
    # gcn_dropout_rate = 0.2
    # gcn_out_channels = 10
    # training_epochs = 100
    # learning_rate = 0.001
    # optimizer_type = "Adam"
    # loss_function = "MSE"
    # n_clusters = None  
    # kmeans_random_state = 42
    # kmeans_n_init = 10
    # in_channels = 1
    
    # email-Eu-core
    # file_name = "email-Eu-core" 
    # input_dir = os.path.join('..', 'norm_dataset', file_name)
    # node_file_path = os.path.join(input_dir, f'{file_name}_nodes.txt')
    # edge_file_path = os.path.join(input_dir, f'{file_name}_edges.txt')
    # gcn_hidden_channels = 16
    # gcn_dropout_rate = 0.2
    # gcn_out_channels = 20
    # training_epochs = 100
    # learning_rate = 0.001
    # optimizer_type = "Adam"
    # loss_function = "MSE"
    # n_clusters = None  
    # kmeans_random_state = 42
    # kmeans_n_init = 10
    # in_channels = 1

    # facebook
    # file_name = "facebook" 
    # input_dir = os.path.join('..', 'norm_dataset', file_name)
    # node_file_path = os.path.join(input_dir, f'{file_name}_nodes.txt')
    # edge_file_path = os.path.join(input_dir, f'{file_name}_edges.txt')
    # gcn_hidden_channels = 128
    # gcn_dropout_rate = 0.2
    # gcn_out_channels = 30
    # training_epochs = 100
    # learning_rate = 0.001
    # optimizer_type = "Adam"
    # loss_function = "MSE"
    # n_clusters = None  
    # kmeans_random_state = 42
    # kmeans_n_init = 10
    # in_channels = 1

    # com-youtube_largest_deliso
    # file_name = "com-youtube_largest_deliso" 
    # input_dir = os.path.join('..', 'norm_dataset', file_name)
    # node_file_path = os.path.join(input_dir, f'{file_name}_nodes.txt')
    # edge_file_path = os.path.join(input_dir, f'{file_name}_edges.txt')
    # gcn_hidden_channels = 64
    # gcn_dropout_rate = 0.2
    # gcn_out_channels = 32
    # training_epochs = 100
    # learning_rate = 0.001
    # optimizer_type = "Adam"
    # loss_function = "MSE"
    # n_clusters = None  
    # kmeans_random_state = 42
    # kmeans_n_init = 10
    # in_channels = 1

    
    G = load_graph_with_attributes(node_file_path, edge_file_path)
    player_names = sorted(G.nodes())
    true_labels = [G.nodes[n]['actual_community'] for n in player_names]
    
    if n_clusters is None:
        n_clusters = len(set(true_labels))
    
    data = prepare_pyg_data(G, player_names)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    
    model = GCN(in_channels, gcn_hidden_channels, gcn_out_channels, gcn_dropout_rate).to(device)
    
    if optimizer_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Prepare training data
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    # Train GCN
    model.train()
    losses = []
    for epoch in range(training_epochs):
        optimizer.zero_grad()
        
        embeddings = model(x, edge_index)
        
        # Loss function
        if loss_function == "MSE":
            loss = F.mse_loss(embeddings, torch.zeros_like(embeddings))
        elif loss_function == "CrossEntropy":
            loss = F.cross_entropy(embeddings, y)
        else:
            loss = F.mse_loss(embeddings, torch.zeros_like(embeddings))
        
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    # Get node embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model(x, edge_index)
        embeddings_np = embeddings.cpu().numpy()
    
    # Perform clustering using K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=kmeans_random_state, n_init=kmeans_n_init)
    pred_labels = kmeans.fit_predict(embeddings_np)
    
    # Label alignment
    pred_labels = best_map(true_labels, pred_labels)
    
    # Calculate modularity
    n_nodes = len(player_names)
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(player_names)}
    
    ig_G = ig.Graph(directed=False)
    ig_G.add_vertices(n_nodes)
    
    edges = []
    for u, v in G.edges():
        u_idx = node_id_to_idx[u]
        v_idx = node_id_to_idx[v]
        edges.append((u_idx, v_idx))
    
    if edges:
        ig_G.add_edges(edges)
    
    modularity_score = ig_G.modularity(pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    
    print(f"{file_name} GCN_result: Modularity: {modularity_score:.6f}, ARI: {ari:.6f}, NMI: {nmi:.6f}")
