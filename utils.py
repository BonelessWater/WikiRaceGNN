import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np 
import torch
import random
from collections import deque
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_undirected
from scipy.sparse.csgraph import connected_components

def visualize_results(results, summary):
    """Visualize comparison results"""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Path lengths
    plt.subplot(2, 2, 1)
    plt.bar(['GNN', 'BFS'], 
            [summary['avg_gnn_path_length'], summary['avg_bfs_path_length']], 
            color=['blue', 'green'])
    plt.title('Average Path Length')
    plt.ylabel('Number of nodes')
    
    # Plot 2: Nodes explored
    plt.subplot(2, 2, 2)
    plt.bar(['GNN', 'BFS'], 
            [summary['avg_gnn_nodes_explored'], summary['avg_bfs_nodes_explored']], 
            color=['blue', 'green'])
    plt.title('Average Nodes Explored')
    plt.ylabel('Number of nodes')
    
    # Plot 3: Success rates
    plt.subplot(2, 2, 3)
    plt.bar(['Success Rate', 'Shortest Path Rate'], 
            [summary['gnn_success_rate'], summary['shortest_path_rate']], 
            color=['blue', 'green'])
    plt.title('GNN Performance')
    plt.ylabel('Rate')
    plt.ylim(0, 1)
    
    # Plot 4: Nodes explored per path length
    plt.subplot(2, 2, 4)
    plt.scatter(results['gnn_path_length'], results['gnn_nodes_explored'], 
                color='blue', label='GNN', alpha=0.7)
    plt.scatter(results['bfs_path_length'], results['bfs_nodes_explored'], 
                color='green', label='BFS', alpha=0.7)
    plt.title('Nodes Explored vs Path Length')
    plt.xlabel('Path length')
    plt.ylabel('Nodes explored')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('plots/traversal_comparison.png')
    plt.close()
    
    # Create a more detailed comparison plot
    plt.figure(figsize=(15, 8))
    
    indices = range(len(results['source']))
    width = 0.35
    
    plt.subplot(2, 1, 1)
    plt.bar([i - width/2 for i in indices], results['gnn_path_length'], width, label='GNN')
    plt.bar([i + width/2 for i in indices], results['bfs_path_length'], width, label='BFS')
    plt.xlabel('Test Case')
    plt.ylabel('Path Length')
    plt.title('Path Length Comparison by Test Case')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.bar([i - width/2 for i in indices], results['gnn_nodes_explored'], width, label='GNN')
    plt.bar([i + width/2 for i in indices], results['bfs_nodes_explored'], width, label='BFS')
    plt.xlabel('Test Case')
    plt.ylabel('Nodes Explored')
    plt.title('Nodes Explored Comparison by Test Case')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('plots/detailed_comparison.png')
    
    print(f"Visualization saved to 'traversal_comparison.png' and 'detailed_comparison.png'")
    
    return summary

def plot_graph_sample(data, max_nodes=50):
    """
    Plot a sample of the graph to visualize its structure
    """
    # Create networkx graph
    G = nx.Graph()
    
    # Add edges (limit to manageable size for visualization)
    edge_index = data.edge_index.numpy()
    for i in range(min(edge_index.shape[1], max_nodes * 3)):
        source = int(edge_index[0, i])
        target = int(edge_index[1, i])
        if source < max_nodes and target < max_nodes:
            G.add_edge(source, target)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=500, edge_color='gray', font_weight='bold')
    plt.title(f'Graph Sample (max {max_nodes} nodes)')
    plt.savefig('plots/graph_sample.png')
    plt.close()
    
    print(f"Graph sample visualization saved to 'graph_sample.png'")

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
    # Read edge list
    df = pd.read_csv(edge_file)
    
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
    
    # Make the graph undirected (for GraphSAGE)
    edge_index = to_undirected(edge_index)
    
    # Generate random node features for demonstration
    # In practice, you would use actual Wikipedia article features
    x = torch.randn((num_nodes, feature_dim))
    
    # Create a PyG Data object
    data = Data(x=x, edge_index=edge_index)
    
    # Store the mapping for reference
    data.node_mapping = node_mapping
    data.reverse_mapping = {idx: node_id for node_id, idx in node_mapping.items()}
    
    # Create an adjacency list for faster BFS
    adj_list = {}
    for i in range(edge_index.size(1)):
        source = edge_index[0, i].item()
        target = edge_index[1, i].item()
        
        if source not in adj_list:
            adj_list[source] = []
        if target not in adj_list:
            adj_list[target] = []
            
        adj_list[source].append(target)
        adj_list[target].append(source)  # For undirected graph
    
    data.adj_list = adj_list
    
    # Verify connectivity
    if ensure_connected:
        # Create a simple adjacency matrix
        adj_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(edge_index.size(1)):
            source = edge_index[0, i].item()
            target = edge_index[1, i].item()
            adj_matrix[source, target] = 1
            adj_matrix[target, source] = 1  # For undirected graph
            
        n_components, labels = connected_components(adj_matrix, directed=False)
        if n_components > 1:
            print(f"Warning: Graph has {n_components} connected components")
        else:
            print("Graph is fully connected")
    
    return data

def neighbor_sampler(nodes, edge_index, num_hops=2, num_neighbors=10):
    """
    Sample neighbors for a batch of nodes.
    This is a simplified version of neighbor sampling.
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
                neighbors = np.random.choice(neighbors, num_neighbors, replace=False).tolist()
            
            next_nodes.update(neighbors)
            sampled_nodes.update(neighbors)
        
        current_nodes = next_nodes
    
    return torch.tensor(list(sampled_nodes), dtype=torch.long)

def bidirectional_bfs(data, start_idx, target_idx):
    """
    Perform bidirectional BFS to find the shortest path between two nodes.
    
    Returns:
        path: List of nodes in the path (or empty list if no path exists)
        nodes_explored: Number of nodes explored during search
    """
    if start_idx == target_idx:
        return [start_idx], 1
    
    # Initialize forward and backward searches
    forward_queue = deque([(start_idx, [start_idx])])
    backward_queue = deque([(target_idx, [target_idx])])
    
    forward_visited = {start_idx: [start_idx]}
    backward_visited = {target_idx: [target_idx]}
    
    nodes_explored = 1  # Start node
    
    while forward_queue and backward_queue:
        # Forward search
        curr_node, curr_path = forward_queue.popleft()
        
        for neighbor in data.adj_list.get(curr_node, []):
            nodes_explored += 1
            if neighbor in backward_visited:
                # Found intersection, construct the path
                forward_path = curr_path
                backward_path = backward_visited[neighbor]
                
                # Combine paths (reverse the backward path)
                full_path = forward_path + backward_path[::-1][1:]
                return full_path, nodes_explored
            
            if neighbor not in forward_visited:
                new_path = curr_path + [neighbor]
                forward_visited[neighbor] = new_path
                forward_queue.append((neighbor, new_path))
        
        # Backward search
        curr_node, curr_path = backward_queue.popleft()
        
        for neighbor in data.adj_list.get(curr_node, []):
            nodes_explored += 1
            if neighbor in forward_visited:
                # Found intersection, construct the path
                forward_path = forward_visited[neighbor]
                backward_path = curr_path
                
                # Combine paths (reverse the backward path)
                full_path = forward_path + backward_path[::-1][1:]
                return full_path, nodes_explored
            
            if neighbor not in backward_visited:
                new_path = curr_path + [neighbor]
                backward_visited[neighbor] = new_path
                backward_queue.append((neighbor, new_path))
    
    # No path found
    return [], nodes_explored
