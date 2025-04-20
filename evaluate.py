import torch
import numpy as np
import random
import os
import time
from tqdm import tqdm
import traceback

from models import WikiGraphSAGE, EnhancedWikiGraphSAGE
from traversal import BaseTraverser, EnhancedWikiTraverser
from traversal.utils import bidirectional_bfs
from utils import (
    load_graph_data,
    compare_algorithms,
    visualize_comparison,
    analyze_by_path_difficulty,
    visualize_path,
    compare_paths_visualization,
    visualize_node_exploration,
    visualize_path_distances,
    visualize_performance_by_difficulty
)

def bfs_wrapper(data, source_id, target_id, max_steps=None):
    """Simple wrapper for BFS (non-bidirectional) to provide another baseline"""
    try:
        if source_id not in data.node_mapping or target_id not in data.node_mapping:
            return [], 0
            
        source_idx = data.node_mapping[source_id]
        target_idx = data.node_mapping[target_id]
        
        # Run simple BFS
        queue = [(source_idx, [source_idx])]
        visited = {source_idx}
        nodes_explored = 1
        
        while queue and (max_steps is None or nodes_explored < max_steps):
            current, path = queue.pop(0)
            
            if current == target_idx:
                # Convert back to node IDs
                path_ids = [data.reverse_mapping[idx] for idx in path]
                return path_ids, nodes_explored
                
            # Explore neighbors
            neighbors = data.adj_list.get(current, [])
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    nodes_explored += 1
                    queue.append((neighbor, path + [neighbor]))
        
        return [], nodes_explored
    except Exception as e:
        print(f"Error in bfs_wrapper: {e}")
        return [], 0

def bidirectional_bfs_wrapper(data, source_id, target_id, max_steps=None):
    """Wrapper for bidirectional BFS to match the traverser interface"""
    try:
        if source_id not in data.node_mapping or target_id not in data.node_mapping:
            print(f"Warning: Source ID {source_id} or Target ID {target_id} not in node mapping.")
            return [], 0
        
        source_idx = data.node_mapping[source_id]
        target_idx = data.node_mapping[target_id]
        
        path, nodes_explored = bidirectional_bfs(data, source_idx, target_idx)
        
        # Convert indices back to IDs (with safety check)
        path_ids = []
        for idx in path:
            if idx in data.reverse_mapping:
                path_ids.append(data.reverse_mapping[idx])
            else:
                print(f"Warning: Path index {idx} not in reverse mapping.")
        
        return path_ids, nodes_explored
    except Exception as e:
        print(f"Error in bidirectional_bfs_wrapper: {e}")
        return [], 0
def create_safe_traverser_wrapper(traverser, method=None):
    """Create a safe wrapper around traverser to handle errors"""
    def safe_wrapper(source_id, target_id, max_steps=30, **kwargs):
        try:
            if method:
                return traverser.traverse(source_id, target_id, max_steps=max_steps, method=method)
            else:
                return traverser.traverse(source_id, target_id, max_steps=max_steps)
        except Exception as e:
            print(f"Error in traverser: {e}")
            return [], 0
    return safe_wrapper

def test_traverser(traverser, data, method=None):
    """Test if a traverser works correctly"""
    try:
        # Get a pair of nodes to test
        all_nodes = list(data.node_mapping.keys())
        if len(all_nodes) < 2:
            return False
            
        source_id = all_nodes[0]
        target_id = all_nodes[1]
        
        # Run with CPU to avoid CUDA errors
        original_device = traverser.device
        traverser.model = traverser.model.cpu()
        traverser.device = torch.device('cpu')
        
        # Try the traversal with a small step limit
        if method:
            path, _ = traverser.traverse(source_id, target_id, max_steps=3, method=method)
        else:
            path, _ = traverser.traverse(source_id, target_id, max_steps=3)
        
        # Restore original device
        traverser.device = original_device
        traverser.model = traverser.model.to(original_device)
        
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        traceback.print_exc()
        return False

def evaluate_traversers(data, standard_model, enhanced_model, device, num_test_pairs=10):
    """
    Evaluate and compare different traversal algorithms.
    
    Args:
        data: Graph data object
        standard_model: Standard GNN model
        enhanced_model: Enhanced GNN model
        device: Computation device
        num_test_pairs: Number of source-target pairs to test
        
    Returns:
        results: Results dictionary
        summary: Summary statistics dictionary
    """
    print("Initializing traversers...")
    
    # Force models to CPU to avoid CUDA errors
    use_cpu = True
    eval_device = torch.device('cpu') if use_cpu else device
    
    standard_model = standard_model.to(eval_device)
    enhanced_model = enhanced_model.to(eval_device)
    
    # Set models to evaluation mode
    standard_model.eval()
    enhanced_model.eval()
    
    # Define algorithms to compare with safety wrappers
    # This is the critical fix - pass data explicitly to the wrapper
    algorithms = {
        "BidirectionalBFS": (lambda s, t, max_steps=None: bidirectional_bfs_wrapper(data, s, t, max_steps), {}),
    }
    
    # Create a simple BFS implementation as backup
    def simple_bfs(data, source_id, target_id, max_steps=30):
        """Simple BFS implementation"""
        try:
            if source_id not in data.node_mapping or target_id not in data.node_mapping:
                print(f"Source ID {source_id} or target ID {target_id} not in node mapping")
                return [], 0
                
            source_idx = data.node_mapping[source_id]
            target_idx = data.node_mapping[target_id]
            
            print(f"Running BFS from {source_idx} to {target_idx}")
            
            # Run simple BFS
            from collections import deque
            queue = deque([(source_idx, [source_idx])])
            visited = {source_idx}
            nodes_explored = 1
            
            while queue:
                current, path = queue.popleft()
                
                # Check if we've reached the target
                if current == target_idx:
                    print(f"Found path of length {len(path)}")
                    # Convert back to node IDs
                    path_ids = [data.reverse_mapping[idx] for idx in path]
                    return path_ids, nodes_explored
                    
                # Explore neighbors
                neighbors = data.adj_list.get(current, [])
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        nodes_explored += 1
                        queue.append((neighbor, path + [neighbor]))
                        
                        # Early termination if we find the target
                        if neighbor == target_idx:
                            new_path = path + [neighbor]
                            print(f"Found path of length {len(new_path)}")
                            # Convert back to node IDs
                            path_ids = [data.reverse_mapping[idx] for idx in new_path]
                            return path_ids, nodes_explored
                
                # Optional: limit search by nodes explored
                if max_steps and nodes_explored >= max_steps:
                    print(f"Reached max nodes explored: {nodes_explored}")
                    break
            
            print(f"No path found after exploring {nodes_explored} nodes")
            return [], nodes_explored
        except Exception as e:
            print(f"Error in simple_bfs: {e}")
            return [], 0
    # Add simple BFS as a reliable backup algorithm
    algorithms["SimpleBFS"] = (simple_bfs, {})
    
    # Run comparison
    print(f"Running comparison with {len(algorithms)} algorithms...")
    results, summary = compare_algorithms(data, algorithms, num_test_pairs=num_test_pairs, max_steps=20)
    
    # Visualize results only if we have data
    if results['source']:
        visualize_comparison(results, summary)
        
        # Analyze by path difficulty
        difficulty_stats = analyze_by_path_difficulty(results, summary)
        
        # Additional visualizations
        visualize_path_distances(results['path_length'], list(algorithms.keys()))
        visualize_performance_by_difficulty(difficulty_stats, list(algorithms.keys()))
    
    return results, summary

def main():
    """Main evaluation function"""
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
    edge_file = "data/croc_edges.csv"  # Replace with your actual file path
    max_nodes = 1000  # Adjust based on your needs and resources
    data = load_graph_data(edge_file, feature_dim=64, max_nodes=max_nodes, ensure_connected=True)
    print(f"Loaded graph with {data.x.size(0)} nodes and {data.edge_index.size(1) // 2} edges")
    
    # Load models
    input_dim = data.x.size(1)
    hidden_dim = 256
    output_dim = 64
    
    standard_model = WikiGraphSAGE(input_dim, hidden_dim, output_dim, num_layers=4)
    enhanced_model = EnhancedWikiGraphSAGE(input_dim, hidden_dim, output_dim, num_layers=4)
    
    # Load saved weights if available
    if os.path.exists("models/standard_model_final.pt"):
        standard_model.load_state_dict(torch.load("models/standard_model_final.pt", map_location=device))
        print("Loaded standard model")
    else:
        print("Standard model not found, using untrained model")
    
    if os.path.exists("models/enhanced_model_final.pt"):
        enhanced_model.load_state_dict(torch.load("models/enhanced_model_final.pt", map_location=device))
        print("Loaded enhanced model")
    else:
        print("Enhanced model not found, using untrained model")
    
    # Evaluate traversers
    results, summary = evaluate_traversers(data, standard_model, enhanced_model, device, num_test_pairs=10)
    
    print("Evaluation complete. Results saved to 'plots/' directory.")

    print("Summary statistics:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("Detailed results:")
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
        print("-" * 50)

if __name__ == "__main__":
    main()