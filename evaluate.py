import torch
import numpy as np
import random
import os
import time
from collections import defaultdict

from models import WikiGraphSAGE, EnhancedWikiGraphSAGE
from traversal import EnhancedGNNTraverser
from traversal import (
    ImprovedGraphTraverser, 
    EnhancedBidirectionalTraverser, 
    OptimizedImprovedGraphTraverser,
    bidirectional_bfs
)
from utils import (
    load_graph_data,
    generate_test_pairs,
    analyze_by_path_difficulty,
    visualize_all
)

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

def test_traversers(data, device, max_steps=100, num_pairs=10):
    """
    Enhanced test function that includes comprehensive visualizations
    
    Args:
        data: Graph data object
        device: Computation device (CPU or GPU)
        max_steps: Maximum steps for traversal
        num_pairs: Number of test pairs to generate
        
    Returns:
        results: Dictionary with evaluation results
        summary: Dictionary with summary statistics
    """
    print("\nTesting traversers with visualizations...")
    
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
    
    standard_model.eval()
    enhanced_model.eval()

    # Create traversers
    improvedStandard = ImprovedGraphTraverser(standard_model, data, device, beam_width=5)
    optimizedImprovedStandard = OptimizedImprovedGraphTraverser(standard_model, data, device)
    standardGNN = EnhancedGNNTraverser(standard_model, data, device)
    enhancedGNN = EnhancedGNNTraverser(enhanced_model, data, device)
    
    standardBidirectional = EnhancedBidirectionalTraverser(standard_model, data, device, beam_width=5)
    enhancedBidirectional = EnhancedBidirectionalTraverser(enhanced_model, data, device, beam_width=5)
    
    # Define algorithms to compare
    algorithms = {
        "BidirectionalBFS": lambda s, t, max_steps: bidirectional_bfs_wrapper(data, s, t, max_steps),
        "ImprovedStandard": lambda s, t, max_steps: improvedStandard.traverse(s, t, max_steps),
        "OptimizedImproved": lambda s, t, max_steps: optimizedImprovedStandard.traverse(s, t, max_steps),
        "StandardGNN": lambda s, t, max_steps: standardGNN.traverse(s, t, max_steps),
        "EnhancedGNN": lambda s, t, max_steps: enhancedGNN.traverse(s, t, max_steps),
        "StandardBidirectional": lambda s, t, max_steps: standardBidirectional.traverse(s, t, max_steps),
        "EnhancedBidirectional": lambda s, t, max_steps: enhancedBidirectional.traverse(s, t, max_steps),
    } 
    
    # Generate test pairs with different path difficulties
    print("Generating test pairs with varied path difficulties...")
    all_test_pairs, test_pairs_by_difficulty = generate_test_pairs(data, num_pairs=num_pairs)
    
    # Count pairs by difficulty
    difficulty_counts = {diff: len(pairs) for diff, pairs in test_pairs_by_difficulty.items()}
    print(f"Test pairs by difficulty: {difficulty_counts}")
    
    # Initialize results tracking
    results = {
        'source': [],
        'target': [],
        'path_length': {name: [] for name in algorithms},
        'nodes_explored': {name: [] for name in algorithms},
        'success': {name: [] for name in algorithms},
        'time': {name: [] for name in algorithms},
        'optimal_path_length': []
    }
    
    # Save actual paths for visualization (only for first few test cases)
    saved_paths = defaultdict(dict)
    
    print("\nRunning evaluations...")
    for i, (source_idx, target_idx) in enumerate(all_test_pairs):
        source_id = data.reverse_mapping[source_idx]
        target_id = data.reverse_mapping[target_idx]
        
        print(f"\nTest {i+1}/{len(all_test_pairs)}: {source_id} → {target_idx}")
        
        # Record source and target
        results['source'].append(source_id)
        results['target'].append(target_id)
        
        # Get optimal path using BFS for reference
        optimal_path, _ = bidirectional_bfs(data, source_idx, target_idx)
        optimal_length = len(optimal_path) if optimal_path else 0
        results['optimal_path_length'].append(optimal_length)
        
        print(f"Optimal path length: {optimal_length}")
        
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
                
                # Save actual path for first few test cases
                if i < 3 and success:
                    if path:
                        path_indices = [data.node_mapping[node_id] for node_id in path if node_id in data.node_mapping]
                        saved_paths[(source_id, target_id)][name] = path_indices
                
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
        'avg_nodes_explored': {name: np.mean([n for n, s in zip(results['nodes_explored'][name], results['success'][name]) if s])
                             if any(results['success'][name]) else 0
                             for name in algorithms},
        'success_rate': {name: np.mean(results['success'][name]) for name in algorithms},
        'avg_time': {name: np.mean([t for t, s in zip(results['time'][name], results['success'][name]) if s])
                   if any(results['success'][name]) else 0
                   for name in algorithms}
    }
    
    # Calculate path optimality (how close to optimal length)
    summary['path_optimality'] = {}
    for name in algorithms:
        optimality = []
        for i, success in enumerate(results['success'][name]):
            if success and results['optimal_path_length'][i] > 0:
                path_length = results['path_length'][name][i]
                optimal_length = results['optimal_path_length'][i]
                optimality.append(optimal_length / path_length)  # Optimal/Actual (lower is better)
        
        summary['path_optimality'][name] = np.mean(optimality) if optimality else 0
    
    # Calculate efficiency ratio
    baseline = "BidirectionalBFS"  # Use BFS as baseline
    summary['efficiency_ratio'] = {}
    
    for name in algorithms:
        if name != baseline and summary['avg_nodes_explored'][name] > 0:
            # Ratio of nodes explored (baseline / algorithm)
            summary['efficiency_ratio'][name] = summary['avg_nodes_explored'][baseline] / summary['avg_nodes_explored'][name]
    
    # Analyze by path difficulty
    difficulty_stats = analyze_by_path_difficulty(results, summary)
    
    # Print summary
    print("\nEvaluation Results:")
    print("-" * 60)
    
    for name in algorithms:
        print(f"{name}:")
        print(f"  Success Rate: {summary['success_rate'][name]:.2%}")
        print(f"  Avg Path Length: {summary['avg_path_length'][name]:.2f}")
        print(f"  Avg Nodes Explored: {summary['avg_nodes_explored'][name]:.2f}")
        print(f"  Avg Time: {summary['avg_time'][name]:.4f}s")
        if name in summary['path_optimality']:
            print(f"  Path Optimality: {summary['path_optimality'][name]:.2f}")
        if name in summary['efficiency_ratio']:
            print(f"  Efficiency Ratio: {summary['efficiency_ratio'][name]:.2f}x")
        print()
    
    # Visualize paths for saved test cases
    print("\nCreating visualizations...")
    os.makedirs('plots', exist_ok=True)
    
    # Visualize all test cases with saved paths
    for (source_id, target_id), paths in saved_paths.items():
        if len(paths) > 1:
            from utils.visualization_manager import compare_paths_visualization
            compare_paths_visualization(
                data, 
                paths,
                f"Path Comparison: {source_id} → {target_id}"
            )
            print(f"Created path comparison for {source_id} → {target_id}")
    
    # Run all visualizations
    models = {
        "Standard": standard_model,
        "Enhanced": enhanced_model
    }
    
    visualize_all(
        results=results,
        summary=summary,
        data=data,
        test_pairs_by_difficulty=test_pairs_by_difficulty,
        difficulty_stats=difficulty_stats,
        algorithms=list(algorithms.keys()),
        models=models,
        device=device
    )
    
    return results, summary, difficulty_stats

def main():
    """Main function to run evaluation with visualizations"""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Load graph data
    edge_file = "data/wiki_edges.csv"  # Replace with your actual file path
    max_nodes = 1000
    data = load_graph_data(
        edge_file, 
        feature_dim=64, 
        max_nodes=max_nodes, 
        ensure_connected=True,
        use_word2vec=True
    )
    print(f"Loaded graph with {data.x.size(0)} nodes and {data.edge_index.size(1) // 2} edges")
    
    # Run evaluation with visualizations
    results, summary, difficulty_stats = test_traversers(
        data=data,
        device=device,
        max_steps=100,
        num_pairs=15  # More test pairs for better statistics
    )
    
    print("\nEvaluation with visualizations complete!")
    print("Check the 'plots/' directory for visualization outputs.")

if __name__ == "__main__":
    main()