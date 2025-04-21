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

def visualize_model_radar_chart(summary, algorithm_names):
    """
    Create a radar chart to compare models across multiple metrics.
    
    Args:
        summary: Dictionary with summary statistics from compare_algorithms
        algorithm_names: List of algorithm names to include
    """
    # Select metrics to include in the radar chart
    metrics = [
        ('Success Rate', 'success_rate', lambda x: x * 100),
        ('Path Optimality', 'avg_path_optimality', lambda x: x * 100),
        ('Efficiency', 'efficiency_ratio', lambda x: x * 50),  # Normalize to similar scale
        ('Speed', 'avg_time', lambda x: 100 / (x + 0.01))  # Invert time (lower is better)
    ]
    
    # Number of metrics
    N = len(metrics)
    
    # Create radar chart
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    
    # Set the angle for each metric (evenly spaced)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Set the labels for each metric
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m[0] for m in metrics])
    
    # Add grid lines and set y-axis limits
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="grey", size=7)
    plt.ylim(0, 100)
    
    # Plot each algorithm
    for i, name in enumerate(algorithm_names):
        # Extract values for each metric
        values = []
        for _, metric_key, normalizer in metrics:
            if metric_key in summary and name in summary[metric_key]:
                value = normalizer(summary[metric_key][name])
                values.append(min(value, 100))  # Cap at 100 for visualization
            else:
                values.append(0)
        
        # Close the loop
        values += values[:1]
        
        # Plot the algorithm
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=name)
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title("Algorithm Performance Comparison", size=15, y=1.1)
    plt.savefig('plots/radar_chart_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Radar chart saved to 'plots/radar_chart_comparison.png'")

def visualize_performance_heatmap(results, algorithm_names):
    """
    Create a heatmap showing the performance of different algorithms across all test cases.
    
    Args:
        results: Results dictionary from compare_algorithms
        algorithm_names: List of algorithm names to include
    """
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Create a composite score for each test case and algorithm
    num_test_cases = len(results['source'])
    score_matrix = np.zeros((len(algorithm_names), num_test_cases))
    
    # Formula: Score = success * (1 + (optimal_length / path_length) - (nodes_explored / max_nodes_explored))
    # This rewards success, path optimality, and exploration efficiency
    
    for i, algo in enumerate(algorithm_names):
        for j in range(num_test_cases):
            # Base score for success
            success = results['success'][algo][j]
            
            if success:
                # Path optimality component (optimal_length / actual_length)
                path_length = results['path_length'][algo][j]
                optimal_length = results['optimal_path_length'][j] if 'optimal_path_length' in results else path_length
                path_ratio = optimal_length / path_length if path_length > 0 else 0
                
                # Exploration efficiency component
                nodes_explored = results['nodes_explored'][algo][j]
                max_nodes = max([results['nodes_explored'][name][j] for name in algorithm_names])
                explore_ratio = nodes_explored / max_nodes if max_nodes > 0 else 1
                
                # Composite score
                score_matrix[i, j] = success * (1 + path_ratio - explore_ratio)
            else:
                score_matrix[i, j] = 0
    
    # Create heatmap
    ax = plt.gca()
    im = ax.imshow(score_matrix, cmap='YlGn')
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Performance Score')
    
    # Set labels
    ax.set_yticks(np.arange(len(algorithm_names)))
    ax.set_yticklabels(algorithm_names)
    ax.set_xticks(np.arange(num_test_cases))
    ax.set_xticklabels([f'Test {i+1}' for i in range(num_test_cases)])
    
    # Rotate x labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add grid
    ax.set_xticks(np.arange(-.5, num_test_cases, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(algorithm_names), 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1, alpha=0.1)
    
    # Add title and labels
    plt.title("Algorithm Performance Across Test Cases")
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('plots/performance_heatmap.png', dpi=150)
    plt.close()
    
    print("Performance heatmap saved to 'plots/performance_heatmap.png'")

def compare_paths_visualization(data, paths_dict, title="Path Comparison"):
    """
    Visualize multiple paths from different algorithms with improved legend.
    
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
    plt.figure(figsize=(14, 14))  # Square figure for better readability
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
                              edge_color=[colors[i]], arrows=True, 
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
    
    # Create a color legend
    custom_lines = [plt.Line2D([0], [0], color=colors[i], lw=4) for i in range(len(processed_paths))]
    plt.legend(custom_lines, [f"{name} (length: {len(path)})" for name, path in processed_paths.items()],
               loc='upper right', title="Algorithm Paths")
    
    # Add a smaller legend for node types
    plt.figtext(0.02, 0.02, "Node Types:", fontweight='bold')
    plt.figtext(0.02, 0.01, "■ Start (Green)  ■ End (Purple)  ○ Path Node (Yellow→Red)  ○ Other Node (Gray)",
               bbox=dict(facecolor='white', alpha=0.8))
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'plots/path_comparison.png', dpi=300, bbox_inches='tight')
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
                    for k, (s, t) in enumerate(zip(results['source'], results['target'])):
                        if s == src_id and t == tgt_id:
                            diff_indices.append(k)
                            break
                
                # Get metric values for these indices
                values = [results[metric][algo][idx] for idx in diff_indices]
                
                # For path length, only consider successful paths
                if metric == 'path_length':
                    successes = [results['success'][algo][idx] for idx in diff_indices]
                    values = [v for v, s in zip(values, successes) if s]
                
                # Calculate aggregate value
                agg_value = agg_func(values) if values else 0
                data_by_difficulty.append(agg_value)
            
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

def visualize_comparison(results, summary):
    """
    Visualize comparison results for different algorithms with enhanced plots.
    
    Args:
        results: Results dictionary from compare_algorithms
        summary: Summary dictionary from compare_algorithms
    """
    # Get algorithm names
    algo_names = list(summary['avg_path_length'].keys())
    
    # Create a figure
    plt.figure(figsize=(20, 25))  # Increased height for more plots
    
    # Plot 1: Average path length
    plt.subplot(5, 2, 1)
    bars = plt.bar(algo_names, [summary['avg_path_length'][name] for name in algo_names])
    plt.title('Average Path Length (Successful Paths)')
    plt.ylabel('Number of nodes')
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}', ha='center', va='bottom')
    
    # Plot 2: Average nodes explored
    plt.subplot(5, 2, 2)
    bars = plt.bar(algo_names, [summary['avg_nodes_explored'][name] for name in algo_names])
    plt.title('Average Nodes Explored')
    plt.ylabel('Number of nodes')
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}', ha='center', va='bottom')
    
    # Plot 3: Success rate
    plt.subplot(5, 2, 3)
    bars = plt.bar(algo_names, [summary['success_rate'][name] for name in algo_names])
    plt.title('Success Rate')
    plt.ylabel('Rate')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.2f}', ha='center', va='bottom')
    
    # Plot 4: Average time
    plt.subplot(5, 2, 4)
    bars = plt.bar(algo_names, [summary['avg_time'][name] for name in algo_names])
    plt.title('Average Execution Time')
    plt.ylabel('Time (seconds)')
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f'{height:.4f}', ha='center', va='bottom')
    
    # Plot 5: Efficiency ratio
    if 'efficiency_ratio' in summary and summary['efficiency_ratio']:
        plt.subplot(5, 2, 5)
        ratio_names = list(summary['efficiency_ratio'].keys())
        bars = plt.bar(ratio_names, [summary['efficiency_ratio'][name] for name in ratio_names])
        plt.title('Efficiency Ratio (Baseline Nodes / Algorithm Nodes)')
        plt.ylabel('Ratio')
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                     f'{height:.2f}', ha='center', va='bottom')
    
    # Plot 6: Path optimality (if available)
    if 'optimal_rate' in summary:
        plt.subplot(5, 2, 6)
        opt_rates = [summary['optimal_rate'].get(name, 0) for name in algo_names]
        bars = plt.bar(algo_names, opt_rates)
        plt.title('Optimal Path Rate')
        plt.ylabel('Rate')
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                     f'{height:.2f}', ha='center', va='bottom')
    
    # Plot 7: Path length distribution
    plt.subplot(5, 2, 7)
    # Create box plots for path lengths
    path_lengths_data = [results['path_length'][name] for name in algo_names]
    plt.boxplot(path_lengths_data, labels=algo_names)
    plt.title('Path Length Distribution')
    plt.ylabel('Path Length')
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    # Plot 8: Nodes explored distribution
    plt.subplot(5, 2, 8)
    # Create box plots for nodes explored
    nodes_explored_data = [results['nodes_explored'][name] for name in algo_names]
    plt.boxplot(nodes_explored_data, labels=algo_names)
    plt.title('Nodes Explored Distribution')
    plt.ylabel('Nodes Explored')
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    # Plot 9: Success rate by test case
    plt.subplot(5, 2, 9)
    indices = range(len(results['source']))
    width = 0.8 / len(algo_names)
    
    for i, name in enumerate(algo_names):
        positions = [idx + (i - len(algo_names)/2 + 0.5) * width for idx in indices]
        plt.bar(positions, results['success'][name], width, label=name)
    
    plt.xlabel('Test Case')
    plt.ylabel('Success (0/1)')
    plt.title('Success by Test Case')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(range(len(indices)), [f'{i+1}' for i in indices], rotation=45)
    
    # Plot 10: Detailed comparison of nodes explored
    plt.subplot(5, 2, 10)
    indices = range(len(results['source']))
    width = 0.8 / len(algo_names)
    
    for i, name in enumerate(algo_names):
        positions = [idx + (i - len(algo_names)/2 + 0.5) * width for idx in indices]
        plt.bar(positions, results['nodes_explored'][name], width, label=name)
    
    plt.xlabel('Test Case')
    plt.ylabel('Nodes Explored')
    plt.title('Nodes Explored per Test Case')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(range(len(indices)), [f'{i+1}' for i in indices], rotation=45)
    
    plt.tight_layout()
    plt.savefig('plots/algorithm_comparison.png', dpi=150)
    plt.close()
    
    # Create a second figure for detailed path lengths
    plt.figure(figsize=(15, 6))
    for i, name in enumerate(algo_names):
        positions = [idx + (i - len(algo_names)/2 + 0.5) * width for idx in indices]
        plt.bar(positions, results['path_length'][name], width, label=name)
    
    plt.xlabel('Test Case')
    plt.ylabel('Path Length')
    plt.title('Path Length per Test Case')
    plt.legend(loc='upper right')
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(range(len(indices)), [f'{i+1}' for i in indices], rotation=45)
    
    plt.tight_layout()
    plt.savefig('plots/path_length_comparison.png')
    
    print("Visualizations saved to 'algorithm_comparison.png' and 'path_length_comparison.png'")
    
    # Return the summary for convenience
    return summary