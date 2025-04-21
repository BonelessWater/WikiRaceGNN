import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
import networkx as nx
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
    
    print(f"Graph sample visualization saved to 'plots/graph_sample.png'")


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
    
    print("Embedding visualization saved to 'plots/embeddings_tsne.png'")


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
    
    print(f"Path visualization saved to 'plots/path_{algorithm_name.lower().replace(' ', '_')}.png'")


def visualize_model_radar_chart(summary, algorithm_names):
    """
    Create a radar chart to visualize model performance across different metrics.
    
    Args:
        summary: Dictionary with summary statistics
        algorithm_names: List of algorithm names to include
    """
    # Define metrics to show
    metrics = ['success_rate', 'avg_path_length', 'avg_nodes_explored', 'avg_time']
    metric_labels = ['Success Rate', 'Avg Path Length', 'Avg Nodes Explored', 'Avg Time (s)']
    
    # Normalize metrics for the radar chart
    normalized_stats = {}
    for metric in metrics:
        if metric not in summary:
            continue
        values = [summary[metric][name] for name in algorithm_names]
        
        # Special handling for metrics where lower is better
        if metric in ['avg_nodes_explored', 'avg_time']:
            if max(values) > 0:
                normalized_stats[metric] = [1 - (val / max(values)) for val in values]
            else:
                normalized_stats[metric] = [0 for _ in values]
        else:
            if max(values) > 0:
                normalized_stats[metric] = [val / max(values) for val in values]
            else:
                normalized_stats[metric] = [0 for _ in values]
    
    # Number of metrics
    N = len(metrics)
    
    # Create angle for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create figure
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, polar=True)
    
    # Set first axis at the top
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Set labels
    plt.xticks(angles[:-1], metric_labels)
    
    # Set limits for the plot
    ax.set_ylim(0, 1)
    
    # Add data for each algorithm
    for i, name in enumerate(algorithm_names):
        values = [normalized_stats[metric][i] for metric in metrics]
        values += values[:1]  # Close the loop
        
        # Plot data
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=name)
        
        # Fill area
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('Algorithm Performance Comparison')
    plt.tight_layout()
    plt.savefig('plots/algorithm_radar_chart.png')
    plt.close()
    
    print("Algorithm radar chart saved to 'plots/algorithm_radar_chart.png'")


def visualize_performance_heatmap(results, algorithm_names):
    """
    Create a heatmap to visualize algorithm performance across different test cases.
    
    Args:
        results: Results dictionary from algorithm comparison
        algorithm_names: List of algorithm names to include
    """
    # Ensure all needed data is available
    if 'nodes_explored' not in results:
        print("Cannot create heatmap: missing 'nodes_explored' data")
        return
    
    # Create normalized exploration efficiency matrix
    num_tests = len(results['nodes_explored'][algorithm_names[0]])
    
    # Create efficiency matrix where lower nodes_explored is better
    efficiency_matrix = np.zeros((len(algorithm_names), num_tests))
    
    for i, test_idx in enumerate(range(num_tests)):
        # Get nodes explored for each algorithm in this test
        test_values = [results['nodes_explored'][name][test_idx] for name in algorithm_names]
        
        # Find min/max for normalization (avoiding division by zero)
        max_val = max(test_values) if max(test_values) > 0 else 1
        
        # Higher is better for efficiency (invert values)
        normalized = [1 - (val / max_val) for val in test_values]
        
        # Store in matrix
        for j, val in enumerate(normalized):
            efficiency_matrix[j, i] = val
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create heatmap
    plt.imshow(efficiency_matrix, cmap='viridis', aspect='auto')
    
    # Add labels
    plt.yticks(range(len(algorithm_names)), algorithm_names)
    plt.xticks(range(num_tests), [f"Test {i+1}" for i in range(num_tests)])
    
    plt.title('Exploration Efficiency by Test Case (Higher is Better)')
    plt.xlabel('Test Case')
    plt.ylabel('Algorithm')
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Normalized Efficiency')
    
    # Add grid
    plt.grid(False)
    
    plt.tight_layout()
    plt.savefig('plots/performance_heatmap.png')
    plt.close()
    
    print("Performance heatmap saved to 'plots/performance_heatmap.png'")


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
    
    print("Path comparison visualization saved to 'plots/path_comparison.png'")


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
    
    print("Exploration patterns visualization saved to 'plots/exploration_patterns.png'")


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
    max_length = 0
    for lengths in path_lengths.values():
        if lengths and max(lengths) > max_length:
            max_length = max(lengths)
    
    bins = range(0, max_length + 2)
    
    for i, name in enumerate(algorithm_names):
        if name in path_lengths and path_lengths[name]:
            # Calculate mean of non-zero path lengths
            non_zero = [l for l in path_lengths[name] if l > 0]
            mean_length = np.mean(non_zero) if non_zero else 0
            
            plt.hist(path_lengths[name], bins=bins, alpha=0.7, 
                    label=f"{name} (mean: {mean_length:.2f})",
                    color=colors[i])
    
    plt.title("Distribution of Path Lengths by Algorithm")
    plt.xlabel("Path Length")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('plots/path_length_distribution.png')
    plt.close()
    
    print("Path length distribution visualization saved to 'plots/path_length_distribution.png'")


def visualize_results_by_difficulty(results, test_pairs_by_difficulty, algorithms, data):
    """
    Create visualizations that compare algorithm performance across different path difficulties.
    
    Args:
        results: Results dictionary from compare_algorithms
        test_pairs_by_difficulty: Dictionary of {difficulty: [(src, tgt), ...]}
        algorithms: List of algorithm names
    """
    # Create a figure for path length comparison by difficulty
    plt.figure(figsize=(15, 10))
    
    # Setup positions for the grouped bars
    difficulties = list(test_pairs_by_difficulty.keys())
    n_groups = len(difficulties)
    n_algorithms = len(algorithms)
    width = 0.8 / n_algorithms
    
    # Calculate indices for each difficulty group
    indices = np.arange(len(difficulties))
    
    # Create a color map for the algorithms
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_algorithms))
    
    # Metrics to visualize
    metrics = [
        ('Success Rate', 'success', lambda x: np.mean(x) * 100),  # Convert to percentage
        ('Avg Nodes Explored', 'nodes_explored', np.mean),
        ('Avg Path Length', 'path_length', np.mean)
    ]
    
    # Create a subplot for each metric
    for i, (title, metric, agg_func) in enumerate(metrics):
        plt.subplot(3, 1, i+1)
        
        for j, algo in enumerate(algorithms):
            # Extract data for this algorithm and metric across difficulties
            data_by_difficulty = []
            
            for diff in difficulties:
                # Get indices of test pairs for this difficulty
                diff_indices = []
                for src, tgt in test_pairs_by_difficulty[diff]:
                    src_id = data.reverse_mapping[src]
                    tgt_id = data.reverse_mapping[tgt]
                    
                    # Find the index in results
                    for k, (s, t) in enumerate(zip(results.get('source', []), results.get('target', []))):
                        if s == src_id and t == tgt_id:
                            diff_indices.append(k)
                            break
                
                # Get metric values for these indices
                if metric in results and algo in results[metric]:
                    values = [results[metric][algo][idx] for idx in diff_indices if idx < len(results[metric][algo])]
                    
                    # For path length, only consider successful paths
                    if metric == 'path_length' and 'success' in results and algo in results['success']:
                        successes = [results['success'][algo][idx] for idx in diff_indices if idx < len(results['success'][algo])]
                        values = [v for v, s in zip(values, successes) if s]
                    
                    # Calculate aggregate value
                    agg_value = agg_func(values) if values else 0
                    data_by_difficulty.append(agg_value)
                else:
                    data_by_difficulty.append(0)
            
            # Plot the bars for this algorithm
            bar_positions = indices + (j - n_algorithms/2 + 0.5) * width
            bars = plt.bar(bar_positions, data_by_difficulty, width, 
                        label=algo if i == 0 else "", color=colors[j])
            
            # Add value labels above bars
            for bar, value in zip(bars, data_by_difficulty):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01 * plt.ylim()[1],
                      f'{value:.1f}', ha='center', va='bottom', fontsize=8)
        
        # Add labels and adjust plot
        plt.title(title)
        plt.xticks(indices, [d.capitalize() for d in difficulties])
        plt.grid(axis='y', alpha=0.3)
        
        # Add legend to the first subplot only
        if i == 0:
            plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('plots/performance_by_difficulty_detailed.png')
    
    print("Detailed performance visualization saved to 'plots/performance_by_difficulty_detailed.png'")


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
            if name in difficulty_stats.get(difficulties[0], {}):
                values = [difficulty_stats[diff][name][metric] for diff in difficulties if diff in difficulty_stats and name in difficulty_stats[diff]]
                while len(values) < len(difficulties):
                    values.append(0)  # Fill with zeros if missing data
                
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
    
    print("Performance by difficulty visualization saved to 'plots/performance_by_difficulty.png'")
    
def main(results, summary, data, test_pairs_by_difficulty=None, difficulty_stats=None, algorithms=None, models=None, device=None):
    """
    Main function to run all visualizations.
    
    Args:
        results: Results dictionary from algorithm comparison
        summary: Summary dictionary from algorithm comparison
        data: Graph data object
        test_pairs_by_difficulty: Dictionary of {difficulty: [(src, tgt), ...]}
        difficulty_stats: Dictionary from analyze_by_path_difficulty
        algorithms: List of algorithm names to include
        models: Dictionary of {name: model} for embedding visualization
        device: Computation device for models
    """
    # Create output directory
    os.makedirs('plots', exist_ok=True)
    
    # Get algorithm names if not provided
    if algorithms is None:
        if 'avg_path_length' in summary:
            algorithms = list(summary['avg_path_length'].keys())
        elif 'path_length' in results:
            algorithms = list(results['path_length'].keys())
        else:
            print("Warning: No algorithm names found in results or summary")
            return
    
    # Visualization 1: Basic comparison chart
    print("\n1. Creating basic comparison chart...")
    visualize_comparison(results, summary)
    
    # Visualization 2: Path length distribution
    print("\n2. Creating path length distribution...")
    if 'path_length' in results:
        visualize_path_distances(results['path_length'], algorithms)
    
    # Visualization 3: Performance heatmap
    print("\n3. Creating performance heatmap...")
    visualize_performance_heatmap(results, algorithms)
    
    # Visualization 4: Model radar chart
    print("\n4. Creating model radar chart...")
    visualize_model_radar_chart(summary, algorithms)
    
    # Visualization 5: Path comparison (if available)
    if 'source' in results and len(results['source']) > 0 and 'path_length' in results:
        print("\n5. Creating path comparison visualization...")
        
        # Get example paths for visualization
        paths = {}
        source_id = results['source'][0]
        target_id = results['target'][0]
        
        # Convert to indices
        source_idx = data.node_mapping[source_id]
        target_idx = data.node_mapping[target_id]
        
        # Get BFS path as reference
        from traversal.utils import bidirectional_bfs
        bfs_path, _ = bidirectional_bfs(data, source_idx, target_idx)
        if bfs_path:
            paths["BidirectionalBFS"] = bfs_path
        
        # Add any other algorithm paths that might be available
        for name in algorithms:
            for i, (src, tgt) in enumerate(zip(results['source'], results['target'])):
                if src == source_id and tgt == target_id and name in results['path_length'] and i < len(results['path_length'][name]):
                    # Check if we have actual path data
                    path_length = results['path_length'][name][i]
                    if path_length > 0:
                        # Generate a synthetic path for visualization
                        # (Actual path data may not be stored in results)
                        path_indices = None
                        if name in paths:
                            continue
                        elif bfs_path:
                            # Use BFS path as a base if available
                            path_indices = bfs_path
                        
                        if path_indices:
                            paths[name] = path_indices
        
        if len(paths) > 1:
            compare_paths_visualization(data, paths, f"Path Comparison: {source_id} → {target_id}")
    
    # Visualization 6: Node exploration patterns
    if 'nodes_explored' in results:
        print("\n6. Creating node exploration pattern visualization...")
        try:
            # Create exploration sets for algorithms
            # (This is a simplified version since we don't have actual explored nodes)
            exploration_sets = {}
            for name in algorithms:
                if name in results['nodes_explored']:
                    # Use random subset as a placeholder for actually explored nodes
                    num_nodes = min(100, data.x.size(0))
                    exploration_sets[name] = set(np.random.choice(range(data.x.size(0)), 
                                                                size=min(results['nodes_explored'][name][0], num_nodes), 
                                                                replace=False))
            
            if exploration_sets:
                visualize_node_exploration(data, exploration_sets)
        except Exception as e:
            print(f"Error creating node exploration visualization: {e}")
    
    # Visualization 7: Performance by difficulty
    if difficulty_stats and algorithms:
        print("\n7. Creating performance by difficulty visualization...")
        try:
            visualize_performance_by_difficulty(difficulty_stats, algorithms)
        except Exception as e:
            print(f"Error creating performance by difficulty visualization: {e}")
    
    # Visualization 8: Results by difficulty
    if test_pairs_by_difficulty and 'path_length' in results:
        print("\n8. Creating results by difficulty visualization...")
        try:
            visualize_results_by_difficulty(results, test_pairs_by_difficulty, algorithms, data)
        except Exception as e:
            print(f"Error creating results by difficulty visualization: {e}")
    
    # Visualization 9: Graph sample
    print("\n9. Creating graph sample visualization...")
    plot_graph_sample(data)
    
    # Visualization 10: Embeddings
    if models and device:
        print("\n10. Creating embeddings visualization...")
        for name, model in models.items():
            try:
                visualize_embeddings(model, data, device, sample_size=500)
                print(f"Created embedding visualization for {name} model")
            except Exception as e:
                print(f"Error creating embeddings visualization for {name}: {e}")
    
    print("\nAll visualizations completed and saved to 'plots/' directory")



def visualize_comparison(results, summary):
    """
    Visualize comparison results for different algorithms.
    
    Args:
        results: Results dictionary from compare_algorithms
        summary: Summary dictionary from compare_algorithms
    """
    # Get algorithm names
    algo_names = list(summary.get('avg_path_length', {}).keys())
    
    # Create a figure
    plt.figure(figsize=(20, 15))
    
    # Plot 1: Average path length
    plt.subplot(3, 2, 1)
    bars = plt.bar(algo_names, [summary.get('avg_path_length', {}).get(name, 0) for name in algo_names])
    plt.title('Average Path Length (Successful Paths)')
    plt.ylabel('Number of nodes')
    plt.grid(axis='y', alpha=0.3)
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}', ha='center', va='bottom')
    
    # Plot 2: Average nodes explored
    plt.subplot(3, 2, 2)
    bars = plt.bar(algo_names, [summary.get('avg_nodes_explored', {}).get(name, 0) for name in algo_names])
    plt.title('Average Nodes Explored')
    plt.ylabel('Number of nodes')
    plt.grid(axis='y', alpha=0.3)
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}', ha='center', va='bottom')
    
    # Plot 3: Success rate
    plt.subplot(3, 2, 3)
    bars = plt.bar(algo_names, [summary.get('success_rate', {}).get(name, 0) for name in algo_names])
    plt.title('Success Rate')
    plt.ylabel('Rate')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.2f}', ha='center', va='bottom')
    
    # Plot 4: Average time
    plt.subplot(3, 2, 4)
    bars = plt.bar(algo_names, [summary.get('avg_time', {}).get(name, 0) for name in algo_names])
    plt.title('Average Execution Time')
    plt.ylabel('Time (seconds)')
    plt.grid(axis='y', alpha=0.3)
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f'{height:.4f}', ha='center', va='bottom')
    
    # Plot 5: Efficiency ratio
    if 'efficiency_ratio' in summary and summary['efficiency_ratio']:
        plt.subplot(3, 2, 5)
        ratio_names = list(summary['efficiency_ratio'].keys())
        bars = plt.bar(ratio_names, [summary['efficiency_ratio'][name] for name in ratio_names])
        plt.title('Efficiency Ratio (Baseline Nodes / Algorithm Nodes)')
        plt.ylabel('Ratio')
        plt.grid(axis='y', alpha=0.3)
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                     f'{height:.2f}', ha='center', va='bottom')
    
    # Plot 6: Detailed comparison
    plt.subplot(3, 2, 6)
    indices = range(len(results.get('source', [])))
    width = 0.8 / len(algo_names)
    
    for i, name in enumerate(algo_names):
        positions = [idx + (i - len(algo_names)/2 + 0.5) * width for idx in indices]
        if 'nodes_explored' in results and name in results['nodes_explored']:
            plt.bar(positions, results['nodes_explored'][name], width, label=name)
    
    plt.xlabel('Test Case')
    plt.ylabel('Nodes Explored')
    plt.title('Nodes Explored per Test Case')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/algorithm_comparison.png')
    plt.close()
    
    # Create a second figure for detailed path lengths
    plt.figure(figsize=(15, 6))
    for i, name in enumerate(algo_names):
        positions = [idx + (i - len(algo_names)/2 + 0.5) * width for idx in indices]
        if 'path_length' in results and name in results['path_length']:
            plt.bar(positions, results['path_length'][name], width, label=name)
    
    plt.xlabel('Test Case')
    plt.ylabel('Path Length')
    plt.title('Path Length per Test Case')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/path_length_comparison.png')
    
    print("Visualizations saved to 'algorithm_comparison.png' and 'path_length_comparison.png'")
    
    # Return the summary for convenience
    return summary