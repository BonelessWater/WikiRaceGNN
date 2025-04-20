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
        
        "StandardGNN": (standard_traverser.traverse, {
            "method": "bidirectional_guided" if hasattr(standard_traverser, "traverse_bidirectional") else None
        }),
        
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