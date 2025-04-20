import torch
import numpy as np
import random
import os
import time
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import WikiGraphSAGE, EnhancedWikiGraphSAGE
from traversal.enhanced import ImprovedGraphTraverser, EnhancedGNNTraverser, SmartGraphTraverser
from traversal.utils import bidirectional_bfs
from utils import (
    load_graph_data,
    visualize_comparison,
    analyze_by_path_difficulty,
    visualize_path_distances,
    visualize_performance_by_difficulty,
    visualize_path,
    compare_paths_visualization
)

def test_improved_traversers(data, device, max_steps=100, num_pairs=10):
    """
    Test improved traversers against the baseline BFS algorithm
    """
    print("\nTesting improved traversers against BFS baseline...")
    
    # Load models
    input_dim = data.x.size(1)
    hidden_dim = 256
    output_dim = 64
    
    standard_model = WikiGraphSAGE(input_dim, hidden_dim, output_dim, num_layers=4)
    enhanced_model = EnhancedWikiGraphSAGE(input_dim, hidden_dim, output_dim, num_layers=4)
    
    # Load trained models if available
    if os.path.exists("models/standard_model_final.pt"):
        standard_model.load_state_dict(torch.load("models/standard_model_final.pt", map_location=device))
        print("Loaded standard model")
    else:
        print("Standard model not found, using random weights")
        
    if os.path.exists("models/enhanced_model_final.pt"):
        enhanced_model.load_state_dict(torch.load("models/enhanced_model_final.pt", map_location=device))
        print("Loaded enhanced model")
    else:
        print("Enhanced model not found, using random weights")
    
    # Create traversers
    from traversal.enhanced import ImprovedGraphTraverser
    improved_standard = ImprovedGraphTraverser(standard_model, data, device, beam_width=3)
    improved_enhanced = ImprovedGraphTraverser(enhanced_model, data, device, beam_width=5)
    
    # Get baseline traversers
    baseline_standard = SmartGraphTraverser(standard_model, data, device)
    baseline_enhanced = EnhancedGNNTraverser(enhanced_model, data, device)
    
    # Define algorithms to compare
    algorithms = {
        "BidirectionalBFS": lambda s, t, max_steps: bidirectional_bfs_wrapper(data, s, t, max_steps),
        "BaselineGNN": lambda s, t, max_steps: baseline_standard.traverse(s, t, max_steps),
        "ImprovedGNN": lambda s, t, max_steps: improved_standard.traverse(s, t, max_steps),
        "EnhancedBidirectional": lambda s, t, max_steps: improved_enhanced.traverse(s, t, max_steps)
    }
    
    # Generate test pairs
    all_nodes = list(range(data.x.size(0)))
    test_pairs = []
    
    print("Generating test pairs...")
    for _ in range(num_pairs):
        # Select random source and target
        while True:
            source = random.choice(all_nodes)
            target = random.choice(all_nodes)
            if source != target:
                # Check if there's a path
                path, _ = bidirectional_bfs(data, source, target)
                if path:
                    test_pairs.append((source, target))
                    break
    
    # Run evaluation
    results = {
        'path_length': {name: [] for name in algorithms},
        'nodes_explored': {name: [] for name in algorithms},
        'success': {name: [] for name in algorithms},
        'time': {name: [] for name in algorithms}
    }
    
    for i, (source, target) in enumerate(test_pairs):
        source_id = data.reverse_mapping[source]
        target_id = data.reverse_mapping[target]
        
        print(f"\nTest {i+1}/{len(test_pairs)}: {source_id} → {target_id}")
        
        # Get optimal path using BFS
        bfs_path, bfs_nodes = bidirectional_bfs(data, source, target)
        print(f"Optimal path length: {len(bfs_path)}")
        
        # Run each algorithm
        for name, algo_func in algorithms.items():
            try:
                start_time = time.time()
                path, nodes_explored = algo_func(source_id, target_id, max_steps)
                elapsed_time = time.time() - start_time
                
                success = len(path) > 0
                success_marker = "✓" if success else "✗"
                
                results['path_length'][name].append(len(path) if path else 0)
                results['nodes_explored'][name].append(nodes_explored)
                results['success'][name].append(success)
                results['time'][name].append(elapsed_time)
                
                print(f"{name}: {success_marker} Path: {len(path)}, Nodes: {nodes_explored}, Time: {elapsed_time:.4f}s")
                
            except Exception as e:
                print(f"Error in {name}: {e}")
                results['path_length'][name].append(0)
                results['nodes_explored'][name].append(0)
                results['success'][name].append(False)
                results['time'][name].append(0)
    
    # Calculate summary statistics
    summary = {
        'avg_path_length': {name: np.mean([l for l, s in zip(results['path_length'][name], results['success'][name]) if s]) 
                           if any(results['success'][name]) else 0 
                           for name in algorithms},
        'avg_nodes_explored': {name: np.mean(results['nodes_explored'][name]) for name in algorithms},
        'success_rate': {name: np.mean(results['success'][name]) for name in algorithms},
        'avg_time': {name: np.mean(results['time'][name]) for name in algorithms}
    }
    
    # Print summary
    print("\nEvaluation Results:")
    print("-" * 60)
    
    for name in algorithms:
        print(f"{name}:")
        print(f"  Success Rate: {summary['success_rate'][name]:.2%}")
        print(f"  Avg Path Length: {summary['avg_path_length'][name]:.2f}")
        print(f"  Avg Nodes Explored: {summary['avg_nodes_explored'][name]:.2f}")
        print(f"  Avg Time: {summary['avg_time'][name]:.4f}s")
        print()
    
    # Visualize paths for an example
    if test_pairs:
        source, target = test_pairs[0]
        source_id = data.reverse_mapping[source]
        target_id = data.reverse_mapping[target]
        
        print(f"\nVisualizing paths for {source_id} → {target_id}...")
        
        paths = {}
        for name, algo_func in algorithms.items():
            path, _ = algo_func(source_id, target_id, max_steps)
            if path:
                # Convert to indices for visualization
                paths[name] = [data.node_mapping[node_id] for node_id in path 
                              if node_id in data.node_mapping]
        
        # Create visualization
        if paths:
            os.makedirs('plots', exist_ok=True)
            compare_paths_visualization(
                data, 
                paths, 
                f"Path Comparison: {source_id} → {target_id}"
            )
            print("Path visualization saved to 'plots/path_comparison.png'")
    
    return results, summary

def bidirectional_bfs_wrapper(data, source_id, target_id, max_steps=None):
    """Wrapper for bidirectional BFS to match the traverser interface"""
    try:
        if source_id not in data.node_mapping or target_id not in data.node_mapping:
            return [], 0
        
        source_idx = data.node_mapping[source_id]
        target_idx = data.node_mapping[target_id]
        
        path, nodes_explored = bidirectional_bfs(data, source_idx, target_idx)
        
        # Convert indices back to IDs
        path_ids = [data.reverse_mapping[idx] for idx in path]
        
        return path_ids, nodes_explored
    except Exception as e:
        print(f"Error in bidirectional_bfs_wrapper: {e}")
        return [], 0
        
def generate_test_pairs(data, num_pairs=30):
    """
    Generate test pairs with balanced path lengths: short, medium, and long.
    This ensures comprehensive evaluation across different path difficulties.
    """
    all_nodes = list(range(data.x.size(0)))
    short_pairs = []  # 2-3 hops
    medium_pairs = []  # 4-6 hops
    long_pairs = []   # 7+ hops
    
    attempts = 0
    max_attempts = 5000
    target_per_category = num_pairs // 3
    
    print("Generating test pairs with varied path lengths...")
    
    with tqdm(total=num_pairs) as pbar:
        while (len(short_pairs) < target_per_category or 
              len(medium_pairs) < target_per_category or 
              len(long_pairs) < target_per_category) and attempts < max_attempts:
            
            source = random.choice(all_nodes)
            target = random.choice(all_nodes)
            
            if source == target:
                attempts += 1
                continue
                
            # Find path using BFS
            path, _ = bidirectional_bfs(data, source, target)
            
            if not path:
                attempts += 1
                continue
                
            path_length = len(path)
            
            # Categorize based on path length
            if path_length <= 3 and len(short_pairs) < target_per_category:
                if (source, target) not in short_pairs:
                    short_pairs.append((source, target))
                    pbar.update(1)
            elif 4 <= path_length <= 6 and len(medium_pairs) < target_per_category:
                if (source, target) not in medium_pairs:
                    medium_pairs.append((source, target))
                    pbar.update(1)
            elif path_length >= 7 and len(long_pairs) < target_per_category:
                if (source, target) not in long_pairs:
                    long_pairs.append((source, target))
                    pbar.update(1)
                
            attempts += 1
            
            # Update progress
            pbar.set_postfix({
                'short': len(short_pairs), 
                'medium': len(medium_pairs), 
                'long': len(long_pairs),
                'attempts': attempts
            })
    
    # Combine all pairs
    all_pairs = short_pairs + medium_pairs + long_pairs
    random.shuffle(all_pairs)
    
    if len(short_pairs) < target_per_category:
        print(f"Warning: Could only find {len(short_pairs)} short paths")
    if len(medium_pairs) < target_per_category:
        print(f"Warning: Could only find {len(medium_pairs)} medium paths")
    if len(long_pairs) < target_per_category:
        print(f"Warning: Could only find {len(long_pairs)} long paths")
        
    print(f"Generated {len(short_pairs)} short, {len(medium_pairs)} medium, and {len(long_pairs)} long paths")
    return all_pairs, {'short': short_pairs, 'medium': medium_pairs, 'long': long_pairs}

def compare_algorithms(data, algorithms, test_pairs, max_steps=100):
    """
    Compare different path finding algorithms with detailed metrics.
    
    Args:
        data: Graph data object
        algorithms: Dictionary of {name: algorithm_function}
        test_pairs: List of (source, target) node pairs
        max_steps: Maximum steps for each algorithm
        
    Returns:
        results: Dictionary with detailed comparison results
        summary: Dictionary with summary statistics
    """
    # Initialize results
    results = {
        'source': [],
        'target': [],
        'path_length': {name: [] for name in algorithms},
        'nodes_explored': {name: [] for name in algorithms},
        'success': {name: [] for name in algorithms},
        'time': {name: [] for name in algorithms},
        'optimal_path_length': []
    }
    
    # Test each pair with each algorithm
    for i, (source, target) in enumerate(test_pairs):
        source_id = data.reverse_mapping[source]
        target_id = data.reverse_mapping[target]
        
        print(f"Testing path {i+1}/{len(test_pairs)}: {source_id} → {target_id}")
        
        # Record source and target
        results['source'].append(source_id)
        results['target'].append(target_id)
        
        # Get optimal path length using BFS for reference
        optimal_path, _ = bidirectional_bfs(data, source, target)
        optimal_length = len(optimal_path) if optimal_path else 0
        results['optimal_path_length'].append(optimal_length)
        
        # Run each algorithm
        for name, algo_func in algorithms.items():
            try:
                start_time = time.time()
                
                # Run the algorithm
                path, nodes_explored = algo_func(source_id, target_id, max_steps)
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                # Record results
                path_length = len(path) if path else 0
                results['path_length'][name].append(path_length)
                results['nodes_explored'][name].append(nodes_explored)
                results['success'][name].append(path_length > 0)
                results['time'][name].append(elapsed_time)
                
                # Print status
                success = "✓" if path_length > 0 else "✗"
                length_str = f"{path_length}"
                if path_length > 0 and optimal_length > 0:
                    length_str += f" (optimal: {optimal_length})"
                
                print(f"  {name}: {success} Path: {length_str}, "
                      f"Nodes: {nodes_explored}, Time: {elapsed_time:.4f}s")
                
            except Exception as e:
                print(f"  Error in {name}: {e}")
                results['path_length'][name].append(0)
                results['nodes_explored'][name].append(0)
                results['success'][name].append(False)
                results['time'][name].append(0)
        
        print("-" * 50)
    
    # Calculate summary statistics
    summary = {
        'avg_path_length': {},
        'avg_nodes_explored': {},
        'success_rate': {},
        'avg_time': {},
        'optimal_rate': {},  # Percentage of paths that match the optimal length
        'avg_path_optimality': {}  # How close paths are to optimal (optimal_length/path_length)
    }
    
    for name in algorithms:
        # Calculate averages
        successful_paths = [l for l, s in zip(results['path_length'][name], results['success'][name]) if s]
        summary['avg_path_length'][name] = np.mean(successful_paths) if successful_paths else 0
        summary['avg_nodes_explored'][name] = np.mean(results['nodes_explored'][name])
        summary['success_rate'][name] = np.mean(results['success'][name])
        summary['avg_time'][name] = np.mean(results['time'][name])
        
        # Calculate path optimality
        optimal_matches = 0
        path_optimality = []
        
        for i, success in enumerate(results['success'][name]):
            if success:
                path_length = results['path_length'][name][i]
                optimal_length = results['optimal_path_length'][i]
                
                if path_length == optimal_length:
                    optimal_matches += 1
                    
                # Calculate optimality ratio (higher is better)
                if path_length > 0 and optimal_length > 0:
                    path_optimality.append(optimal_length / path_length)
        
        if len(results['success'][name]) > 0:
            summary['optimal_rate'][name] = optimal_matches / len(results['success'][name])
        else:
            summary['optimal_rate'][name] = 0
            
        summary['avg_path_optimality'][name] = np.mean(path_optimality) if path_optimality else 0
    
    # Calculate efficiency ratios
    baseline = next(iter(algorithms.keys()))  # First algorithm as baseline
    summary['efficiency_ratio'] = {}
    
    for name in algorithms:
        if name != baseline and summary['avg_nodes_explored'][name] > 0:
            # Ratio of nodes explored (baseline / algorithm)
            summary['efficiency_ratio'][name] = summary['avg_nodes_explored'][baseline] / summary['avg_nodes_explored'][name]
    
    return results, summary

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

def analyze_paths(data, algorithms, source_id, target_id, max_steps=100):
    """
    Analyze path finding algorithms on a specific source-target pair.
    Visualizes paths found by different algorithms for detailed comparison.
    
    Args:
        data: Graph data object
        algorithms: Dictionary of {name: algorithm_function}
        source_id: Source node ID
        target_id: Target node ID
        max_steps: Maximum steps for each algorithm
    """
    print(f"\nAnalyzing paths from {source_id} to {target_id}...")
    
    # Convert IDs to indices
    if source_id not in data.node_mapping or target_id not in data.node_mapping:
        print("Invalid source or target ID")
        return
        
    source_idx = data.node_mapping[source_id]
    target_idx = data.node_mapping[target_id]
    
    # Get optimal path using BFS
    optimal_path, optimal_nodes = bidirectional_bfs(data, source_idx, target_idx)
    
    # Paths found by each algorithm
    paths = {}
    
    if optimal_path:
        paths["Optimal (BFS)"] = [data.reverse_mapping[idx] for idx in optimal_path]
        print(f"Optimal path length: {len(optimal_path)}, explored {optimal_nodes} nodes")
    else:
        print("No path found using BFS")
    
    # Run each algorithm
    for name, algo_func in algorithms.items():
        start_time = time.time()
        path, nodes_explored = algo_func(source_id, target_id, max_steps)
        elapsed_time = time.time() - start_time
        
        paths[name] = path
        
        if path:
            print(f"{name}: Path length {len(path)}, explored {nodes_explored} nodes in {elapsed_time:.4f}s")
        else:
            print(f"{name}: No path found, explored {nodes_explored} nodes in {elapsed_time:.4f}s")
    
    # Convert paths to internal indices for visualization
    paths_indices = {}
    for name, path in paths.items():
        if path:
            paths_indices[name] = [data.node_mapping[node_id] for node_id in path 
                                  if node_id in data.node_mapping]
    
    # Visualize paths
    if paths_indices:
        os.makedirs('plots', exist_ok=True)
        compare_paths_visualization(
            data, 
            paths_indices, 
            f"Path Comparison: {source_id} → {target_id}"
        )
        print("Path visualization saved to 'plots/path_comparison.png'")
    else:
        print("No paths to visualize")

def main():
    """Main function for evaluating graph traversal algorithms"""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs('plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Load graph data
    edge_file = "data/wiki_edges.csv"  # Replace with your actual file path
    max_nodes = 1000  # Adjust based on your needs and resources
    data = load_graph_data(edge_file, feature_dim=64, max_nodes=max_nodes, ensure_connected=True)
    print(f"Loaded graph with {data.x.size(0)} nodes and {data.edge_index.size(1) // 2} edges")
    
    # Load models
    input_dim = data.x.size(1)
    hidden_dim = 256
    output_dim = 64
    
    standard_model = WikiGraphSAGE(input_dim, hidden_dim, output_dim, num_layers=4)
    enhanced_model = EnhancedWikiGraphSAGE(input_dim, hidden_dim, output_dim, num_layers=4)
    
    # Load trained models if available
    if os.path.exists("models/standard_model_final.pt"):
        standard_model.load_state_dict(torch.load("models/standard_model_final.pt", map_location=device))
        print("Loaded standard model")
    else:
        print("Standard model not found, using random weights")
        
    if os.path.exists("models/enhanced_model_final.pt"):
        enhanced_model.load_state_dict(torch.load("models/enhanced_model_final.pt", map_location=device))
        print("Loaded enhanced model")
    else:
        print("Enhanced model not found, using random weights")
    
    # Create traversers
    standard_traverser = ImprovedGraphTraverser(standard_model, data, device, beam_width=3)
    enhanced_traverser = ImprovedGraphTraverser(enhanced_model, data, device, beam_width=5)
    
    # Define algorithms to compare
    algorithms = {
        "BidirectionalBFS": lambda s, t, max_steps: bidirectional_bfs_wrapper(data, s, t, max_steps),
        "StandardGNN": lambda s, t, max_steps: standard_traverser.traverse(s, t, max_steps),
        "EnhancedGNN": lambda s, t, max_steps: enhanced_traverser.traverse(s, t, max_steps)
    }
    
    # Generate test pairs with varied path lengths
    all_test_pairs, test_pairs_by_difficulty = generate_test_pairs(data, num_pairs=30)
    
    # Run comparison
    results, summary = compare_algorithms(data, algorithms, all_test_pairs, max_steps=100)
    
    # Print summary results
    print("\nEvaluation complete. Summary results:")
    print("-" * 60)
    
    for name in algorithms:
        print(f"{name}:")
        print(f"  Success Rate: {summary['success_rate'][name]:.2%}")
        print(f"  Avg Path Length: {summary['avg_path_length'][name]:.2f}")
        print(f"  Avg Nodes Explored: {summary['avg_nodes_explored'][name]:.2f}")
        print(f"  Avg Time: {summary['avg_time'][name]:.4f}s")
        print(f"  Optimal Path Rate: {summary['optimal_rate'][name]:.2%}")
        print(f"  Avg Path Optimality: {summary['avg_path_optimality'][name]:.2%}")
        print()
    
    # Visualize results
    visualize_comparison(results, summary)
    
    # Analyze by path difficulty
    difficulty_stats = analyze_by_path_difficulty(results, summary)
    
    # Additional visualizations
    visualize_path_distances(results['path_length'], list(algorithms.keys()))
    visualize_performance_by_difficulty(difficulty_stats, list(algorithms.keys()))
    visualize_results_by_difficulty(results, test_pairs_by_difficulty, list(algorithms.keys()), data)
    
    # Analyze specific cases - look at some challenging paths
    if test_pairs_by_difficulty['long']:
        # Analyze a long path
        source, target = test_pairs_by_difficulty['long'][0]
        source_id = data.reverse_mapping[source]
        target_id = data.reverse_mapping[target]
        analyze_paths(data, algorithms, source_id, target_id)
    
    if test_pairs_by_difficulty['medium']:
        # Analyze a medium path
        source, target = test_pairs_by_difficulty['medium'][0]
        source_id = data.reverse_mapping[source]
        target_id = data.reverse_mapping[target]
        analyze_paths(data, algorithms, source_id, target_id)
    
    print("\nEvaluation complete!")

    print("\n\n" + "="*50)
    print("TESTING IMPROVED TRAVERSERS")
    print("="*50)
    test_improved_traversers(data, device, max_steps=100, num_pairs=5)


if __name__ == "__main__":
    main()