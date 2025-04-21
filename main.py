import argparse
import torch
import os
import numpy as np
import random

from models import EnhancedWikiGraphSAGE
from traversal import EnhancedWikiTraverser
from utils import load_graph_data, crawl_main
from utils.wikibuilder import create_wiki_edge_list

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced Wikipedia Graph Traversal')
    
    parser.add_argument('--edge_file', type=str, default='data/wiki_edges.csv',
                        help='Path to edge list CSV file')
    parser.add_argument('--max_nodes', type=int, default=1000,
                        help='Maximum number of nodes to include in the graph')
    parser.add_argument('--feature_dim', type=int, default=64,
                        help='Dimension of node features')
    
    parser.add_argument('--mode', type=str, choices=['data','train', 'evaluate', 'traverse', 'pipeline'],
                        default='traverse', help='Operation mode')
    
    parser.add_argument('--model_path', type=str, default='models/enhanced_model_final.pt',
                        help='Path to saved model (for evaluate and traverse modes)')
    
    parser.add_argument('--source_id', type=int, default=None,
                        help='Source node ID for traversal (if None, will be chosen randomly)')
    parser.add_argument('--target_id', type=int, default=None,
                        help='Target node ID for traversal (if None, will be chosen randomly)')
    
    parser.add_argument('--method', type=str, 
                        choices=['parallel_beam', 'bidirectional_guided', 'hierarchical', 
                                'parallel_bidirectional', 'hybrid', 'auto'],
                        default='auto', help='Traversal method to use')
    
    parser.add_argument('--max_steps', type=int, default=30,
                        help='Maximum steps for traversal')
    
    parser.add_argument('--num_neighbors', type=int, default=20,
                        help='Number of neighbors to sample in each step')
    
    parser.add_argument('--beam_width', type=int, default=5,
                        help='Beam width for beam search')
    
    parser.add_argument('--heuristic_weight', type=float, default=1.5,
                        help='Weight for heuristic component in A* search')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the traversal results')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU if available')
    
    return parser.parse_args()

def build_graph(args):
    """Build graph from edge file or create a new one if it doesn't exist"""
    print("Starting data generation...")
    
    # Create the data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Check if the edge file exists
    if not os.path.exists(args.edge_file):
        print(f"Edge file {args.edge_file} not found. Creating a new graph...")
        
        # Create a new edge list
        edge_file_path = create_wiki_edge_list(
            output_dir="data",
            max_nodes=args.max_nodes,
            use_word2vec=True
        )
        
        print(f"Created edge file at {edge_file_path}")
        return edge_file_path
    else:
        print(f"Using existing edge file at {args.edge_file}")
        return args.edge_file

def run_training(args, data, device):
    """Run training mode"""
    from train import train_path_predictor
    
    print("Starting training...")
    
    # Load or initialize the model
    input_dim = data.x.size(1)
    hidden_dim = 256
    output_dim = 64
    
    model = EnhancedWikiGraphSAGE(input_dim, hidden_dim, output_dim, num_layers=4)
    
    # Train the model
    model = train_path_predictor(
        data=data,
        model=model,
        device=device,
        num_epochs=10,
        batch_size=32,
        num_train_pairs=1000,
        learning_rate=0.001,
        weight_decay=0.0001,
        validation_split=0.2
    )
    
    # Save the final model
    torch.save(model.state_dict(), "models/enhanced_model_final.pt")
    print(f"Saved trained model to models/enhanced_model_final.pt")
    
    return model

def run_evaluation(args, data, device, model=None):
    """Run evaluation mode"""
    from evaluate import test_traversers
    
    print("Starting evaluation...")
    
    if model is None:
        # Load model if not provided
        input_dim = data.x.size(1)
        hidden_dim = 256
        output_dim = 64
        
        model = EnhancedWikiGraphSAGE(input_dim, hidden_dim, output_dim, num_layers=4)
        
        if os.path.exists(args.model_path):
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            print(f"Loaded model from {args.model_path}")
        else:
            print(f"Warning: Model file {args.model_path} not found. Using untrained model.")
    
    # Run evaluation
    results, summary, difficulty_stats = test_traversers(
        data=data,
        device=device,
        max_steps=args.max_steps,
        num_pairs=10
    )

    print("\nEvaluation complete!")
    print("Difficulty statistics:")
    for difficulty, stats in difficulty_stats.items():
        print(f"Difficulty {difficulty}: {stats}")
            
    return results, summary

def run_traversal(args, data, device):
    """Run traversal mode"""
    print("Starting traversal...")
    
    # Load model
    input_dim = data.x.size(1)
    hidden_dim = 256
    output_dim = 64
    
    model = EnhancedWikiGraphSAGE(input_dim, hidden_dim, output_dim, num_layers=4)
    
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model from {args.model_path}")
    else:
        print(f"Warning: Model file {args.model_path} not found. Using untrained model.")
    
    model = model.to(device)
    
    # Create traverser
    traverser = EnhancedWikiTraverser(
        model, data, device,
        beam_width=args.beam_width,
        heuristic_weight=args.heuristic_weight,
        num_neighbors=args.num_neighbors,
        num_hops=2
    )
    
    # Choose source and target if not specified
    all_nodes = list(range(data.x.size(0)))
    
    source_idx = None
    if args.source_id is not None:
        if args.source_id in data.node_mapping:
            source_idx = data.node_mapping[args.source_id]
        else:
            print(f"Warning: Source ID {args.source_id} not found in graph. Choosing randomly.")
    
    if source_idx is None:
        source_idx = random.choice(all_nodes)
    
    source_id = data.reverse_mapping[source_idx]
    
    target_idx = None
    if args.target_id is not None:
        if args.target_id in data.node_mapping:
            target_idx = data.node_mapping[args.target_id]
        else:
            print(f"Warning: Target ID {args.target_id} not found in graph. Choosing randomly.")
    
    if target_idx is None:
        # Choose a target that has a path from source
        while True:
            target_idx = random.choice(all_nodes)
            if target_idx != source_idx:
                # Check if there's a path
                from traversal.utils import bidirectional_bfs
                path, _ = bidirectional_bfs(data, source_idx, target_idx)
                if path:
                    break
    
    target_id = data.reverse_mapping[target_idx]
    
    print(f"Traversing from node {source_id} to node {target_id}")
    
    # Run different traversal methods
    from traversal.utils import bidirectional_bfs
    
    # Compare with BFS as baseline
    print("\nRunning bidirectional BFS (baseline)...")
    bfs_start_time = torch.cuda.Event(enable_timing=True)
    bfs_end_time = torch.cuda.Event(enable_timing=True)
    
    bfs_start_time.record()
    bfs_path, bfs_nodes = bidirectional_bfs(data, source_idx, target_idx)
    bfs_end_time.record()
    
    torch.cuda.synchronize()
    bfs_time = bfs_start_time.elapsed_time(bfs_end_time) / 1000  # convert to seconds
    
    bfs_path_ids = [data.reverse_mapping[idx] for idx in bfs_path] if bfs_path else []
    
    # Run enhanced traversal
    print(f"\nRunning enhanced traversal with method: {args.method}...")
    enh_start_time = torch.cuda.Event(enable_timing=True)
    enh_end_time = torch.cuda.Event(enable_timing=True)
    
    enh_start_time.record()
    enhanced_path, enhanced_nodes = traverser.traverse(
        source_id, target_id, max_steps=args.max_steps, method=args.method
    )
    enh_end_time.record()
    
    torch.cuda.synchronize()
    enh_time = enh_start_time.elapsed_time(enh_end_time) / 1000  # convert to seconds
    
    # Print results
    print("\nResults:")
    print(f"BidirectionalBFS: {len(bfs_path_ids)} nodes, explored {bfs_nodes} nodes, time: {bfs_time:.4f}s")
    print(f"EnhancedTraversal: {len(enhanced_path)} nodes, explored {enhanced_nodes} nodes, time: {enh_time:.4f}s")
    
    print("\nBFS Path:")
    for i, node_id in enumerate(bfs_path_ids):
        print(f"Step {i}: Node {node_id}")
    
    print("\nEnhanced Path:")
    for i, node_id in enumerate(enhanced_path):
        print(f"Step {i}: Node {node_id}")
    
    # Calculate improvement metrics
    if bfs_nodes > 0:
        nodes_reduction = (bfs_nodes - enhanced_nodes) / bfs_nodes * 100
        print(f"\nEnhanced traversal explored {nodes_reduction:.2f}% fewer nodes than BFS")
    
    if bfs_time > 0:
        time_speedup = (bfs_time - enh_time) / bfs_time * 100
        print(f"Enhanced traversal was {time_speedup:.2f}% faster than BFS")
    
    # Visualize if requested
    if args.visualize:
        from utils.visualization_manager import visualize_path, compare_paths_visualization
        
        os.makedirs('plots', exist_ok=True)
        
        visualize_path(data, bfs_path, "Bidirectional BFS")
        visualize_path(data, enhanced_path, "Enhanced Traversal")
        
        compare_paths_visualization(data, {
            "BidirectionalBFS": bfs_path_ids,
            "EnhancedTraversal": enhanced_path
        }, f"Path Comparison: Node {source_id} → Node {target_id}")
        
        print("\nVisualizations saved to 'plots/' directory")

def run_pipeline(args, device):
    """Run the full pipeline: data generation, training, and evaluation"""
    print("Starting full pipeline: data generation, training, and evaluation")
    
    # Step 1: Generate data
    edge_file = build_graph(args)
    
    # Step 2: Load graph data
    data = load_graph_data(
        edge_file, 
        feature_dim=args.feature_dim, 
        max_nodes=args.max_nodes, 
        ensure_connected=True,
        use_word2vec=True
    )
    print(f"Loaded graph with {data.x.size(0)} nodes and {data.edge_index.size(1) // 2} edges")
    
    # Step 3: Train the model
    model = run_training(args, data, device)
    
    # Step 4: Evaluate the model
    results, summary = run_evaluation(args, data, device, model)
    
    # Step 5: Run a sample traversal
    run_traversal(args, data, device)
    
    print("\nPipeline complete! Model trained and evaluated successfully.")
    return model, results, summary

def main():
    """Main function"""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Determine device
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Mode-specific operations
    if args.mode == 'data':
        edge_file = build_graph(args)
        print(f"Data generation complete. Edge file created at {edge_file}")
    elif args.mode == 'pipeline':
        run_pipeline(args, device)
    elif args.mode == 'train':
        # Load graph data
        data = load_graph_data(
            args.edge_file, 
            feature_dim=args.feature_dim, 
            max_nodes=args.max_nodes, 
            ensure_connected=True,
            use_word2vec=True
        )
        print(f"Loaded graph with {data.x.size(0)} nodes and {data.edge_index.size(1) // 2} edges")
        
        run_training(args, data, device)
    elif args.mode == 'evaluate':
        # Load graph data
        data = load_graph_data(
            args.edge_file, 
            feature_dim=args.feature_dim, 
            max_nodes=args.max_nodes, 
            ensure_connected=True,
            use_word2vec=True
        )
        print(f"Loaded graph with {data.x.size(0)} nodes and {data.edge_index.size(1) // 2} edges")
        
        run_evaluation(args, data, device)

    else:  # traverse mode
        # Load graph data
        data = load_graph_data(
            args.edge_file, 
            feature_dim=args.feature_dim, 
            max_nodes=args.max_nodes, 
            ensure_connected=True,
            use_word2vec=True
        )
        print(f"Loaded graph with {data.x.size(0)} nodes and {data.edge_index.size(1) // 2} edges")
        
        run_traversal(args, data, device)

if __name__ == "__main__":
    main()