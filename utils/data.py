import torch
import pandas as pd
import numpy as np
import random
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

def load_graph_data(edge_file, feature_dim=64, max_nodes=None, ensure_connected=True):
    """
    Load graph data from edge list file with option to reduce to a smaller connected graph.
    
    Args:
        edge_file: Path to the CSV edge list
        feature_dim: Dimension of node features to generate
        max_nodes: Maximum number of nodes to include (if None, use all)
        ensure_connected: Ensure the sampled graph is connected
        
    Returns:
        Data object with graph information
    """
    try:
        # Try to read edge list
        df = pd.read_csv(edge_file)
    except FileNotFoundError:
        # If file not found, create a simple test graph
        print(f"Edge file {edge_file} not found. Creating a test graph.")
        # Create a simple grid graph
        G = nx.grid_2d_graph(10, 10)
        # Relabel nodes to integers
        G = nx.convert_node_labels_to_integers(G)
        # Create edge list
        edges = list(G.edges())
        df = pd.DataFrame(edges, columns=['id1', 'id2'])
    
    if max_nodes is not None:
        # Create networkx graph for connectivity analysis
        G = nx.Graph()
        for _, row in df.iterrows():
            G.add_edge(row['id1'], row['id2'])
        
        if ensure_connected:
            # Find largest connected component if needed
            largest_cc = max(nx.connected_components(G), key=len)
            
            # If the largest component is smaller than max_nodes, use the whole component
            if len(largest_cc) < max_nodes:
                print(f"Warning: Largest connected component has only {len(largest_cc)} nodes")
                print(f"Using all {len(largest_cc)} nodes from largest component")
                selected_nodes = list(largest_cc)
            else:
                # Start with a random node from the largest component
                start_node = random.choice(list(largest_cc))
                selected_nodes = [start_node]
                frontier = list(nx.neighbors(G, start_node))
                
                # Breadth-first selection to ensure connectivity
                while len(selected_nodes) < max_nodes and frontier:
                    next_node = frontier.pop(0)
                    if next_node not in selected_nodes:
                        selected_nodes.append(next_node)
                        # Add its neighbors to the frontier
                        for neighbor in nx.neighbors(G, next_node):
                            if neighbor not in selected_nodes and neighbor not in frontier:
                                frontier.append(neighbor)
        else:
            # Simply select random nodes
            selected_nodes = random.sample(list(G.nodes()), min(max_nodes, len(G.nodes())))
        
        # Filter edges to only include selected nodes
        filtered_edges = []
        for _, row in df.iterrows():
            if row['id1'] in selected_nodes and row['id2'] in selected_nodes:
                filtered_edges.append((row['id1'], row['id2']))
                
        # Create a new dataframe with the filtered edges
        df = pd.DataFrame(filtered_edges, columns=['id1', 'id2'])
        
        print(f"Reduced graph to {len(selected_nodes)} nodes and {len(filtered_edges)} edges")

    # Get unique node IDs
    node_ids = np.unique(df[['id1', 'id2']].values.flatten())
    num_nodes = len(node_ids)
    
    # Create mapping from original IDs to consecutive indices
    node_mapping = {node_id: idx for idx, node_id in enumerate(node_ids)}
    
    # Convert edges to use consecutive indices
    edge_index = torch.tensor([
        [node_mapping[id1] for id1 in df['id1'].values],
        [node_mapping[id2] for id2 in df['id2'].values]
    ], dtype=torch.long)
    
    # Make the graph undirected
    edge_index = to_undirected(edge_index)
    
    # Generate random node features (would be replaced with actual features in practice)
    x = torch.randn((num_nodes, feature_dim))
    
    # Create a PyG Data object
    data = Data(x=x, edge_index=edge_index)
    
    # Store the mapping for reference
    data.node_mapping = node_mapping
    data.reverse_mapping = {idx: node_id for node_id, idx in node_mapping.items()}
    
    # Create an adjacency list for faster BFS and neighbor lookup
    adj_list = {}
    for i in range(edge_index.size(1)):
        source = edge_index[0, i].item()
        target = edge_index[1, i].item()
        
        if source not in adj_list:
            adj_list[source] = []
        if target not in adj_list:
            adj_list[target] = []
            
        adj_list[source].append(target)
    
    data.adj_list = adj_list
    
    return data


def neighbor_sampler(nodes, edge_index, num_hops=2, num_neighbors=10):
    """
    Sample neighbors for a batch of nodes.
    This is a simplified version of neighbor sampling with control over breadth and depth.
    
    Args:
        nodes: Tensor of node indices
        edge_index: Edge indices tensor [2, E]
        num_hops: Number of hops to expand
        num_neighbors: Max number of neighbors to sample per node
        
    Returns:
        sampled_nodes: Tensor of sampled node indices
    """
    sampled_nodes = set(nodes.tolist())
    current_nodes = set(nodes.tolist())
    
    for _ in range(num_hops):
        next_nodes = set()
        for node in current_nodes:
            # Find neighbors of the current node
            neighbors = edge_index[1][edge_index[0] == node].tolist()
            
            # Sample a subset if there are too many neighbors
            if len(neighbors) > num_neighbors:
                neighbors = random.sample(neighbors, num_neighbors)
            
            next_nodes.update(neighbors)
            sampled_nodes.update(neighbors)
        
        current_nodes = next_nodes
    
    return torch.tensor(list(sampled_nodes), dtype=torch.long)