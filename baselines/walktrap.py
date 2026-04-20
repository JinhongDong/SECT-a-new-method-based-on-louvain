# walktrap.py
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
import networkx as nx
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score
)
import igraph as ig

# ---------------------------------------------------------------------------#
# 0. Read network
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
    # tree
    file_name = "g22"  
    input_dir = os.path.join('..', 'norm_dataset', file_name)
    node_file_path = os.path.join(input_dir, f'{file_name}_nodes.txt')
    edge_file_path = os.path.join(input_dir, f'{file_name}_edges.txt')
    walktrap_steps = 10           
    
    # lol
    # file_name = "lol" 
    # input_dir = os.path.join('..', 'norm_dataset', file_name)
    # node_file_path = os.path.join(input_dir, f'{file_name}_nodes.txt')
    # edge_file_path = os.path.join(input_dir, f'{file_name}_edges.txt')
    # walktrap_steps = 5
    
    # g22
    # file_name = "g22" 
    # input_dir = os.path.join('..', 'norm_dataset', file_name)
    # node_file_path = os.path.join(input_dir, f'{file_name}_nodes.txt')
    # edge_file_path = os.path.join(input_dir, f'{file_name}_edges.txt')
    # walktrap_steps = 10
    
    # email-Eu-core
    # file_name = "email-Eu-core" 
    # input_dir = os.path.join('..', 'norm_dataset', file_name)
    # node_file_path = os.path.join(input_dir, f'{file_name}_nodes.txt')
    # edge_file_path = os.path.join(input_dir, f'{file_name}_edges.txt')
    # walktrap_steps = 10
    
    # facebook
    # file_name = "facebook" 
    # input_dir = os.path.join('..', 'norm_dataset', file_name)
    # node_file_path = os.path.join(input_dir, f'{file_name}_nodes.txt')
    # edge_file_path = os.path.join(input_dir, f'{file_name}_edges.txt')
    # walktrap_steps = 10
    
    # com-youtube_largest_deliso
    # file_name = "com-youtube_largest_deliso" 
    # input_dir = os.path.join('..', 'norm_dataset', file_name)
    # node_file_path = os.path.join(input_dir, f'{file_name}_nodes.txt')
    # edge_file_path = os.path.join(input_dir, f'{file_name}_edges.txt')
    # walktrap_steps = 5
    
    nx_G = load_graph_with_attributes(node_file_path, edge_file_path)
    player_names = sorted(nx_G.nodes())
    true_labels = [nx_G.nodes[n]['actual_community'] for n in player_names]
    
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(player_names)}
    idx_to_node_id = {idx: node_id for idx, node_id in enumerate(player_names)}
    
    ig_G = ig.Graph(directed=False)
    
    n_nodes = len(player_names)
    ig_G.add_vertices(n_nodes)
    
    edges = []
    for u, v in nx_G.edges():
        u_idx = node_id_to_idx[u]
        v_idx = node_id_to_idx[v]
        edges.append((u_idx, v_idx))
    
    if edges:
        ig_G.add_edges(edges)
    
    walktrap_result = ig_G.community_walktrap(
        steps=walktrap_steps,
    )
    
    dendrogram = walktrap_result.as_clustering()
    membership = dendrogram.membership
    
    pred_labels = [membership[node_id_to_idx[node_id]] for node_id in player_names]

    pred_labels = best_map(true_labels, pred_labels)
    
    modularity_score = ig_G.modularity(membership)
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    
    print(f"{file_name} walktrap_result: Modularity: {modularity_score:.6f}, ARI: {ari:.6f}, NMI: {nmi:.6f}")
