import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from sklearn.manifold import TSNE
from matplotlib.colors import LinearSegmentedColormap

def plot_graph_sample(data, max_nodes=50):
    """
    Plot a sample of the graph to visualize its structure.
    
    Args:
        data: Graph data object
        max_nodes: Maximum number of nodes to include in the visualization
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


def visualize_embeddings(model, data, device, sample_size=500):
    """
    Visualize node embeddings using t-SNE.
    
    Args:
        model: Trained GNN model
        data: Graph data object
        device: Computation device
        sample_size: Number of nodes to sample for visualization
    """
    # Sample nodes
    num_nodes = data.x.size(0)
    if num_nodes > sample_size:
        indices = np.random.choice(num_nodes, sample_size, replace=False)
    else:
        indices = np.arange(num_nodes)
    
    # Get embeddings
    model.eval()
    with torch.no_grad():
        x = data.x[indices].to(device)
        edge_index = data.edge_index.to(device)
        batch = torch.zeros(len(indices), dtype=torch.long).to(device)
        embeddings = model(x, edge_index, batch).cpu().numpy()
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(12, 10))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, s=50)
    
    # Add node indices for some points
    for i in range(min(50, len(indices))):
        plt.annotate(str(indices[i]), (embeddings_2d[i, 0], embeddings_2d[i, 1]))
    
    plt.title('t-SNE Visualization of Node Embeddings')
    plt.savefig('plots/embeddings_tsne.png')
    plt.close()
    
    print("Embedding visualization saved to 'embeddings_tsne.png'")


def visualize_path(data, path, algorithm_name="Algorithm"):
    """
    Visualize a path through the graph.
    
    Args:
        data: Graph data object
        path: List of node indices forming the path
        algorithm_name: Name of the algorithm used to find the path
    """
    if not path:
        print("No path to visualize")
        return
    
    # Convert to internal indices if needed
    if isinstance(path[0], type(data.reverse_mapping[0])):  # Check if path contains IDs instead of indices
        path = [data.node_mapping[node_id] for node_id in path if node_id in data.node_mapping]
    
    # Create networkx graph for visualization
    G = nx.Graph()
    
    # Add all nodes from the path and their immediate neighbors
    nodes_to_include = set(path)
    for node in path:
        # Add neighbors
        neighbors = data.adj_list.get(node, [])
        nodes_to_include.update(neighbors)
    
    # Add edges connecting these nodes
    edge_index = data.edge_index.numpy()
    for i in range(edge_index.shape[1]):
        source = int(edge_index[0, i])
        target = int(edge_index[1, i])
        if source in nodes_to_include and target in nodes_to_include:
            G.add_edge(source, target)
    
    # Create plot
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)
    
    # Draw all nodes and edges
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray')
    
    # Draw nodes with different colors
    path_nodes = set(path)
    other_nodes = nodes_to_include - path_nodes
    
    # Draw path nodes (highlighted)
    nx.draw_networkx_nodes(G, pos, nodelist=list(path_nodes), 
                          node_color='red', node_size=600)
    
    # Draw other nodes
    nx.draw_networkx_nodes(G, pos, nodelist=list(other_nodes), 
                          node_color='lightblue', node_size=300, alpha=0.7)
    
    # Draw path edges with a different color and width
    path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=3, 
                          edge_color='red', arrows=True)
    
    # Add labels for path nodes
    path_labels = {node: str(node) for node in path}
    nx.draw_networkx_labels(G, pos, labels=path_labels, font_weight='bold')
    
    # Draw start and end nodes with special markers
    start_node = path[0]
    end_node = path[-1]
    
    nx.draw_networkx_nodes(G, pos, nodelist=[start_node], node_color='green', 
                          node_size=800, node_shape='s')
    nx.draw_networkx_nodes(G, pos, nodelist=[end_node], node_color='purple', 
                          node_size=800, node_shape='s')
    
    # Add special labels for start and end
    plt.annotate("START", xy=pos[start_node], xytext=(-40, 40),
                textcoords="offset points", fontsize=12, fontweight='bold',
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    
    plt.annotate("END", xy=pos[end_node], xytext=(40, 40),
                textcoords="offset points", fontsize=12, fontweight='bold',
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    
    plt.title(f'Path Found by {algorithm_name} (Length: {len(path)})')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'plots/path_{algorithm_name.lower().replace(" ", "_")}.png')
    plt.close()
    
    print(f"Path visualization saved to 'path_{algorithm_name.lower().replace(' ', '_')}.png'")


def compare_paths_visualization(data, paths_dict, title="Path Comparison"):
    """
    Visualize multiple paths from different algorithms.
    
    Args:
        data: Graph data object
        paths_dict: Dictionary of {algorithm_name: path}
        title: Title for the visualization
    """
    if not paths_dict:
        print("No paths to visualize")
        return
    
    # Convert paths to internal indices if needed
    processed_paths = {}
    for name, path in paths_dict.items():
        if not path:
            continue
            
        if isinstance(path[0], type(data.reverse_mapping[0])):  # Check if path contains IDs
            processed_paths[name] = [data.node_mapping[node_id] for node_id in path 
                                    if node_id in data.node_mapping]
        else:
            processed_paths[name] = path
    
    # Create a combined set of all nodes in all paths
    all_path_nodes = set()
    for path in processed_paths.values():
        all_path_nodes.update(path)
    
    # Add neighbors for context
    nodes_to_include = set(all_path_nodes)
    for node in all_path_nodes:
        neighbors = data.adj_list.get(node, [])
        for neighbor in neighbors:
            if len(nodes_to_include) < 100:  # Limit total nodes for clarity
                nodes_to_include.add(neighbor)
    
    # Create networkx graph
    G = nx.Graph()
    
    # Add edges
    edge_index = data.edge_index.numpy()
    for i in range(edge_index.shape[1]):
        source = int(edge_index[0, i])
        target = int(edge_index[1, i])
        if source in nodes_to_include and target in nodes_to_include:
            G.add_edge(source, target)
    
    # Create plot
    plt.figure(figsize=(14, 12))
    pos = nx.spring_layout(G, seed=42)
    
    # Draw background edges
    nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='gray')
    
    # Draw background nodes
    background_nodes = nodes_to_include - all_path_nodes
    nx.draw_networkx_nodes(G, pos, nodelist=list(background_nodes), 
                          node_color='lightgray', node_size=200, alpha=0.5)
    
    # Colors for different algorithms
    colors = plt.cm.tab10(np.linspace(0, 1, len(processed_paths)))
    
    # Create a custom color map for path nodes
    path_node_cmap = LinearSegmentedColormap.from_list('path_cmap', 
                                                     ['lightyellow', 'orange', 'red'])
    
    # Draw paths with different colors
    for i, (name, path) in enumerate(processed_paths.items()):
        # Draw path edges
        path_edges = [(path[j], path[j+1]) for j in range(len(path)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=2+i, 
                              edge_color=colors[i], arrows=True, 
                              label=f"{name} (length: {len(path)})")
        
        # Draw path nodes with color indicating position
        for j, node in enumerate(path):
            # Color based on position in the path (start=yellow, end=red)
            color = path_node_cmap(j / (len(path)-1) if len(path) > 1 else 0.5)
            nx.draw_networkx_nodes(G, pos, nodelist=[node], 
                                  node_color=[color], node_size=300+i*50)
        
        # Draw start and end nodes specially
        if path:
            start_node, end_node = path[0], path[-1]
            nx.draw_networkx_nodes(G, pos, nodelist=[start_node], 
                                  node_color='lime', node_size=500, node_shape='s')
            nx.draw_networkx_nodes(G, pos, nodelist=[end_node], 
                                  node_color='magenta', node_size=500, node_shape='s')
    
    # Add labels for all path nodes
    labels = {node: str(node) for node in all_path_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_weight='bold')
    
    plt.title(title)
    plt.legend(loc='upper right')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'plots/path_comparison.png')
    plt.close()
    
    print("Path comparison visualization saved to 'path_comparison.png'")


def visualize_node_exploration(data, algorithm_explorations, max_nodes=100):
    """
    Visualize which nodes different algorithms explored.
    
    Args:
        data: Graph data object
        algorithm_explorations: Dictionary of {algorithm_name: set_of_explored_nodes}
        max_nodes: Maximum number of nodes to include
    """
    # Combine all explored nodes
    all_explored = set()
    for nodes in algorithm_explorations.values():
        all_explored.update(nodes)
    
    # Limit to max_nodes
    if len(all_explored) > max_nodes:
        all_explored = set(list(all_explored)[:max_nodes])
    
    # Create networkx graph
    G = nx.Graph()
    G.add_nodes_from(all_explored)
    
    # Add edges connecting these nodes
    edge_index = data.edge_index.numpy()
    for i in range(edge_index.shape[1]):
        source = int(edge_index[0, i])
        target = int(edge_index[1, i])
        if source in all_explored and target in all_explored:
            G.add_edge(source, target)
    
    # Create plot
    plt.figure(figsize=(14, 12))
    pos = nx.spring_layout(G, seed=42)
    
    # Draw base graph
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray')
    
    # Colors for different algorithms
    colors = plt.cm.tab10(np.linspace(0, 1, len(algorithm_explorations)))
    
    # Draw exploration patterns for each algorithm
    for i, (name, nodes) in enumerate(algorithm_explorations.items()):
        nodes_subset = [n for n in nodes if n in all_explored]
        color = colors[i]
        alpha = 0.6
        
        nx.draw_networkx_nodes(G, pos, nodelist=nodes_subset,
                              node_color=[color]*len(nodes_subset),
                              alpha=alpha, label=f"{name} ({len(nodes)} nodes)",
                              node_size=100)
    
    # Add labels
    labels = {node: str(node) for node in all_explored}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    
    plt.title("Node Exploration Patterns by Algorithm")
    plt.legend(loc='upper right')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'plots/exploration_patterns.png')
    plt.close()
    
    print("Exploration patterns visualization saved to 'exploration_patterns.png'")


def visualize_path_distances(path_lengths, algorithm_names):
    """
    Create a histogram of path lengths for different algorithms.
    
    Args:
        path_lengths: Dictionary of {algorithm_name: list_of_path_lengths}
        algorithm_names: List of algorithm names to include
    """
    plt.figure(figsize=(12, 8))
    
    # Colors for different algorithms
    colors = plt.cm.tab10(np.linspace(0, 1, len(algorithm_names)))
    
    # Create histograms
    bins = range(0, max(max(lengths) for lengths in path_lengths.values() if lengths) + 2)
    
    for i, name in enumerate(algorithm_names):
        plt.hist(path_lengths[name], bins=bins, alpha=0.7, 
                label=f"{name} (mean: {np.mean([l for l in path_lengths[name] if l > 0]):.2f})",
                color=colors[i])
    
    plt.title("Distribution of Path Lengths by Algorithm")
    plt.xlabel("Path Length")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('plots/path_length_distribution.png')
    plt.close()
    
    print("Path length distribution visualization saved to 'path_length_distribution.png'")


def visualize_performance_by_difficulty(difficulty_stats, algorithm_names):
    """
    Visualize algorithm performance across different difficulty levels.
    
    Args:
        difficulty_stats: Dictionary from analyze_by_path_difficulty
        algorithm_names: List of algorithm names to include
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    difficulties = ['short', 'medium', 'long']
    metrics = ['success_rate', 'avg_nodes', 'avg_time']
    titles = ['Success Rate', 'Average Nodes Explored', 'Average Time (s)']
    
    # Colors for different algorithms
    colors = plt.cm.tab10(np.linspace(0, 1, len(algorithm_names)))
    
    # For each metric
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        
        # Set up positions for grouped bars
        x = np.arange(len(difficulties))
        width = 0.8 / len(algorithm_names)
        
        # Plot bars for each algorithm
        for j, name in enumerate(algorithm_names):
            values = [difficulty_stats[diff][name][metric] for diff in difficulties]
            pos = x + (j - len(algorithm_names)/2 + 0.5) * width
            bars = ax.bar(pos, values, width, label=name if i == 0 else None, color=colors[j])
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.03,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([d.capitalize() for d in difficulties])
        ax.grid(axis='y', alpha=0.3)
        
        # Only add legend to the first plot
        if i == 0:
            ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('plots/performance_by_difficulty.png')
    plt.close()
    
    print("Performance by difficulty visualization saved to 'performance_by_difficulty.png'")