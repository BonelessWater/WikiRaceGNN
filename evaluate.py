import torch
import numpy as np
import random
import os
import time
from tqdm import tqdm

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

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def load_models(data, device):
    """
    Load trained models.
    
    Args:
        data: Graph data object
        device: Computation device
        
    Returns:
        standard_model: Standard GNN model
        enhanced_model: Enhanced GNN model
    """
    # Initialize models
    input_dim = data.x.size(1)
    hidden_dim = 256
    output_dim = 64
    
    standard_model = WikiGraphSAGE(input_dim, hidden_dim, output_dim, num_layers=4)
    enhanced_model = EnhancedWikiGraphSAGE(input_dim, hidden_dim, output_dim, num_layers=4)
    
    # Load saved weights if available
    if os.path.exists("models/standard_model_final.pt"):
        standard_model.load_state_dict(torch.load("models/standard_model_final.pt"))
        print("Loaded standard model")
    else:
        print("Standard model not found, using untrained model")
    
    if os.path.exists("models/enhanced_model_final.pt"):
        enhanced_model.load_state_dict(torch.load("models/enhanced_model_final.pt"))
        print("Loaded enhanced model")
    else:
        print("Enhanced model not found, using untrained model")
    
    standard_model = standard_model.to(device)
    enhanced_model = enhanced_model.to(device)
    
    return standard_model, enhanced_model


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
    # Create traversers
    standard_traverser = BaseTraverser(standard_model, data, device, num_neighbors=20, num_hops=2)
    enhanced_traverser = EnhancedWikiTraverser(
        enhanced_model, data, device, 
        beam_width=5, 
        heuristic_weight=1.5, 
        num_neighbors=20,
        num_hops=2
    )
    
    # Define algorithms to compare
    algorithms = {
        "BidirectionalBFS": (lambda s, t, max_steps: bidirectional_bfs_wrapper(data, s, t), {}),
        
        # Remove the method parameter since BaseTraverser doesn't support it
        "StandardGNN": (standard_traverser.traverse, {}),
        
        "EnhancedBeam": (enhanced_traverser.traverse, {
            "method": "parallel_beam"
        }),
        
        "EnhancedBidir": (enhanced_traverser.traverse, {
            "method": "bidirectional_guided"
        }),
        
        "EnhancedHybrid": (enhanced_traverser.traverse, {
            "method": "hybrid"
        })
    }
    
    # Run comparison
    results, summary = compare_algorithms(data, algorithms, num_test_pairs=num_test_pairs)
    
    # Visualize results
    visualize_comparison(results, summary)
    
    # Analyze by path difficulty
    difficulty_stats = analyze_by_path_difficulty(results, summary)
    
    # Additional visualizations
    visualize_path_distances(results['path_length'], list(algorithms.keys()))
    visualize_performance_by_difficulty(difficulty_stats, list(algorithms.keys()))
    
    # Visualize specific path examples
    test_visualizations(data, standard_traverser, enhanced_traverser, num_examples=3)
    
    return results, summary
def bidirectional_bfs_wrapper(data, source_id, target_id):
    """Wrapper for bidirectional BFS to match the traverser interface"""
    if source_id not in data.node_mapping or target_id not in data.node_mapping:
        return [], 0
    
    source_idx = data.node_mapping[source_id]
    target_idx = data.node_mapping[target_id]
    
    path, nodes_explored = bidirectional_bfs(data, source_idx, target_idx)
    
    # Convert indices back to IDs
    path_ids = [data.reverse_mapping[idx] for idx in path]
    
    return path_ids, nodes_explored


def test_visualizations(data, standard_traverser, enhanced_traverser, num_examples=3):
    """
    Create visualizations for specific path examples.
    
    Args:
        data: Graph data object
        standard_traverser: Standard GNN traverser
        enhanced_traverser: Enhanced GNN traverser
        num_examples: Number of examples to visualize
    """
    # Get list of all nodes
    all_nodes = list(range(data.x.size(0)))
    
    for i in range(num_examples):
        # Choose random source and target
        while True:
            source = random.choice(all_nodes)
            target = random.choice(all_nodes)

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
    standard_model, enhanced_model = load_models(data, device)
    
    # Evaluate traversers
    results, summary = evaluate_traversers(data, standard_model, enhanced_model, device, num_test_pairs=10)
    
    print("Evaluation complete. Results saved to 'plots/' directory.")

    print("Summary statistics:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("Detailed results:")
    for name, result in results.items():
        print(f"{name}:")
        for k, v in result.items():
            print(f"  {k}: {v}")
        print("-" * 50)


if __name__ == "__main__":
    main()