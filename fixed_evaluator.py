import torch
import numpy as np
import random
import os
import time
from collections import deque
from tqdm import tqdm

from models import WikiGraphSAGE, EnhancedWikiGraphSAGE, SmartGraphTraverser
from traversal.utils import bidirectional_bfs
from utils import (
    load_graph_data,
    visualize_comparison,
    analyze_by_path_difficulty,
    visualize_path_distances,
    visualize_performance_by_difficulty
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
        path_ids = []
        for idx in path:
            if idx in data.reverse_mapping:
                path_ids.append(data.reverse_mapping[idx])
        
        return path_ids, nodes_explored
    except Exception as e:
        print(f"Error in bidirectional_bfs_wrapper: {e}")
        return [], 0

def simple_bfs(data, source_id, target_id, max_steps=100):
    """Simple BFS implementation"""
    try:
        if source_id not in data.node_mapping or target_id not in data.node_mapping:
            return [], 0
            
        source_idx = data.node_mapping[source_id]
        target_idx = data.node_mapping[target_id]
        
        # Run simple BFS
        queue = deque([(source_idx, [source_idx])])
        visited = {source_idx}
        nodes_explored = 1
        
        while queue and nodes_explored < max_steps:
            current, path = queue.popleft()
            
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
        print(f"Error in simple_bfs: {e}")
        return [], 0

class OptimizedGraphSAGETraverser:
    """A simplified and optimized traverser for GraphSAGE models"""
    
    def __init__(self, model, data, device='cpu'):
        self.model = model
        self.data = data
        self.device = device
        self.embedding_cache = {}
        
        # Put model in eval mode
        self.model.eval()
        
    def get_embedding(self, node_idx):
        """Get embedding for a single node using a local subgraph"""
        if node_idx in self.embedding_cache:
            return self.embedding_cache[node_idx]
            
        with torch.no_grad():
            # Create a local subgraph for this node only
            neighbors = []
            for i in range(self.data.edge_index.size(1)):
                if self.data.edge_index[0, i].item() == node_idx:
                    neighbors.append(self.data.edge_index[1, i].item())
                elif self.data.edge_index[1, i].item() == node_idx:
                    neighbors.append(self.data.edge_index[0, i].item())
            
            # Include the node itself
            subgraph_nodes = [node_idx] + neighbors[:20]  # Limit to 20 neighbors for efficiency
            sub_nodes = torch.tensor(subgraph_nodes, dtype=torch.long)
            
            # Create a mini-batch with just this node
            sub_x = self.data.x[sub_nodes].to(self.device)
            
            # Get node features
            node_x = self.data.x[node_idx].unsqueeze(0).to(self.device)
            
            # Calculate embedding - use pre-trained model but avoid using edge_index directly
            batch = torch.zeros(1, dtype=torch.long).to(self.device)
            
            # Use a forward pass with just the features (no message passing)
            embedding = self.model.embedding(node_x)
            
            # Store in cache
            self.embedding_cache[node_idx] = embedding
            
            return embedding
    
    def score_neighbors(self, neighbors, target_emb, temperature=1.0):
        scores = []
        for neighbor in neighbors:
            neighbor_emb = self.get_node_embedding(neighbor)
            similarity = self.get_similarity(neighbor_emb, target_emb)
            # Apply temperature scaling
            scaled_similarity = similarity / temperature
            scores.append((neighbor, scaled_similarity))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)
        
    def traverse(self, source_id, target_id, max_steps=100):
        """Perform graph traversal using the trained GNN model"""
        if source_id not in self.data.node_mapping or target_id not in self.data.node_mapping:
            return [], 0
            
        source_idx = self.data.node_mapping[source_id]
        target_idx = self.data.node_mapping[target_id]
        
        # Initialize
        visited = {source_idx}
        queue = [(source_idx, [source_idx])]
        nodes_explored = 1
        
        while queue and nodes_explored < max_steps:
            current, path = queue.pop(0)
            
            if current == target_idx:
                # Convert back to node IDs
                path_ids = [self.data.reverse_mapping[idx] for idx in path]
                return path_ids, nodes_explored
            
            # Get and score neighbors
            neighbors, scores = self.score_neighbors(current, target_idx)
            
            # Sort neighbors by score (descending)
            if neighbors and scores:
                sorted_pairs = sorted(zip(neighbors, scores), key=lambda x: x[1], reverse=True)
                
                # Add sorted neighbors to queue
                for neighbor, _ in sorted_pairs:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        nodes_explored += 1
                        queue.append((neighbor, path + [neighbor]))
        
        return [], nodes_explored

def compare_algorithms(data, algorithms, test_pairs, max_steps=100):
    """Compare different path finding algorithms"""
    # Initialize results
    results = {
        'source': [],
        'target': [],
        'path_length': {name: [] for name in algorithms},
        'nodes_explored': {name: [] for name in algorithms},
        'success': {name: [] for name in algorithms},
        'time': {name: [] for name in algorithms}
    }
    
    # Test each pair with each algorithm
    for source, target in test_pairs:
        source_id = data.reverse_mapping[source]
        target_id = data.reverse_mapping[target]
        
        print(f"Testing path from node {source_id} to {target_id}")
        
        # Record source and target
        results['source'].append(source_id)
        results['target'].append(target_id)
        
        # Run each algorithm
        for name, algo_func in algorithms.items():
            try:
                start_time = time.time()
                
                # Run the algorithm
                path, nodes_explored = algo_func(source_id, target_id, max_steps)
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                # Record results
                results['path_length'][name].append(len(path) if path else 0)
                results['nodes_explored'][name].append(nodes_explored)
                results['success'][name].append(bool(path))
                results['time'][name].append(elapsed_time)
                
                print(f"{name}: Path length: {len(path) if path else 0}, "
                      f"Nodes explored: {nodes_explored}, Success: {bool(path)}, "
                      f"Time: {elapsed_time:.4f}s")
                
            except Exception as e:
                print(f"Error in {name}: {e}")
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
        'avg_time': {}
    }
    
    for name in algorithms:
        # Calculate averages
        successful_paths = [l for l, s in zip(results['path_length'][name], results['success'][name]) if s]
        summary['avg_path_length'][name] = np.mean(successful_paths) if successful_paths else 0
        summary['avg_nodes_explored'][name] = np.mean(results['nodes_explored'][name])
        summary['success_rate'][name] = np.mean(results['success'][name])
        summary['avg_time'][name] = np.mean(results['time'][name])
    
    # Calculate efficiency ratios
    baseline = "BidirectionalBFS"  # Use this as our standard
    summary['efficiency_ratio'] = {}
    
    for name in algorithms:
        if name != baseline and summary['avg_nodes_explored'][name] > 0:
            summary['efficiency_ratio'][name] = summary['avg_nodes_explored'][baseline] / summary['avg_nodes_explored'][name]
    
    return results, summary

def generate_test_pairs(data, num_pairs=30):
    """Generate test pairs with varied path lengths"""
    all_nodes = list(range(data.x.size(0)))
    short_pairs = []  # 2-3 hops
    medium_pairs = [] # 4-6 hops
    long_pairs = []   # 7+ hops
    
    attempts = 0
    max_attempts = 1000
    
    print("Generating test pairs with varied path lengths...")
    
    while (len(short_pairs) < num_pairs//3 or 
           len(medium_pairs) < num_pairs//3 or 
           len(long_pairs) < num_pairs//3) and attempts < max_attempts:
        
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
        
        # Categorize based on length
        if path_length <= 3 and len(short_pairs) < num_pairs//3:
            short_pairs.append((source, target))
        elif 4 <= path_length <= 6 and len(medium_pairs) < num_pairs//3:
            medium_pairs.append((source, target))
        elif path_length >= 7 and len(long_pairs) < num_pairs//3:
            long_pairs.append((source, target))
            
        attempts += 1
    
    # Combine all pairs
    all_pairs = short_pairs + medium_pairs + long_pairs
    random.shuffle(all_pairs)
    
    print(f"Generated {len(short_pairs)} short, {len(medium_pairs)} medium, and {len(long_pairs)} long paths")
    return all_pairs


class EnhancedGNNTraverser:
    def __init__(self, model, data, device):
        self.model = model.to(device)
        self.data = data
        self.device = device
        self.model.eval()
        self.embedding_cache = {}
        
    def get_node_embedding(self, node_idx):
        if node_idx in self.embedding_cache:
            return self.embedding_cache[node_idx]
            
        with torch.no_grad():
            # Get features for the node and its immediate neighbors
            neighbors = []
            for i in range(self.data.edge_index.size(1)):
                if self.data.edge_index[0, i].item() == node_idx:
                    neighbors.append(self.data.edge_index[1, i].item())
                elif self.data.edge_index[1, i].item() == node_idx:
                    neighbors.append(self.data.edge_index[0, i].item())
            
            # Limit to avoid memory issues
            neighbors = neighbors[:20]
            
            # Include node itself
            node_list = [node_idx] + neighbors
            nodes_tensor = torch.tensor(node_list, dtype=torch.long)
            
            # Create feature matrix for this subgraph
            x = self.data.x[nodes_tensor].to(self.device)
            
            # Use the embedding layer to get features
            embedding = self.model.embedding(x[0].unsqueeze(0))
            self.embedding_cache[node_idx] = embedding
            return embedding
    
    def get_path_similarity(self, path, target_idx):
        """Calculate a path's overall similarity to target"""
        target_emb = self.get_node_embedding(target_idx)
        path_emb = torch.zeros_like(target_emb)
        
        # Weight more recent nodes higher
        for i, node in enumerate(path):
            node_emb = self.get_node_embedding(node)
            # Recency weighting - more recent nodes have higher weight
            weight = (i + 1) / len(path)
            path_emb += node_emb * weight
            
        # Normalize
        path_emb = path_emb / len(path)
        
        # Calculate similarity
        return F.cosine_similarity(path_emb, target_emb).item()
    
    def traverse(self, source_id, target_id, max_steps=100):
        if source_id not in self.data.node_mapping or target_id not in self.data.node_mapping:
            return [], 0
            
        source_idx = self.data.node_mapping[source_id]
        target_idx = self.data.node_mapping[target_id]
        
        if source_idx == target_idx:
            return [source_id], 1
        
        # Beam search with path history consideration
        beam_width = 3
        visited = {source_idx}
        beams = [([source_idx], 0.0)]  # (path, score)
        nodes_explored = 1
        
        for _ in range(max_steps):
            if not beams:
                break
                
            new_beams = []
            
            for path, _ in beams:
                current = path[-1]
                
                if current == target_idx:
                    # Convert to IDs and return
                    path_ids = [self.data.reverse_mapping[idx] for idx in path]
                    return path_ids, nodes_explored
                
                # Get neighbors
                neighbors = []
                for i in range(self.data.edge_index.size(1)):
                    if self.data.edge_index[0, i].item() == current:
                        neighbor = self.data.edge_index[1, i].item()
                        if neighbor not in visited:
                            neighbors.append(neighbor)
                            visited.add(neighbor)
                            nodes_explored += 1
                
                # Score each new potential path
                for neighbor in neighbors:
                    new_path = path + [neighbor]
                    
                    # Get similarity score considering the entire path
                    score = self.get_path_similarity(new_path, target_idx)
                    
                    # Add extra score if this connects to the target
                    if neighbor == target_idx:
                        score += 1.0
                        
                    new_beams.append((new_path, score))
            
            # Keep top beam_width paths
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            
            # Check if we've found the target
            if beams and beams[0][0][-1] == target_idx:
                path_ids = [self.data.reverse_mapping[idx] for idx in beams[0][0]]
                return path_ids, nodes_explored
        
        return [], nodes_explored

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Create directories
    os.makedirs('plots', exist_ok=True)
    
    # Load graph data
    edge_file = "data/croc_edges.csv"
    max_nodes = 1000
    data = load_graph_data(edge_file, feature_dim=64, max_nodes=max_nodes, ensure_connected=True)
    print(f"Loaded graph with {data.x.size(0)} nodes and {data.edge_index.size(1) // 2} edges")
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load models
    input_dim = data.x.size(1)
    hidden_dim = 256
    output_dim = 64
    
    standard_model = WikiGraphSAGE(input_dim, hidden_dim, output_dim, num_layers=4)
    enhanced_model = EnhancedWikiGraphSAGE(input_dim, hidden_dim, output_dim, num_layers=4)
    
    # For testing, initialize with random weights
    # To use trained models, uncomment below
    
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
    
    
    # Force CPU for stability
    device = torch.device('cpu')
    standard_model = standard_model.to(device)
    enhanced_model = enhanced_model.to(device)

    # Create traversers
    standard_traverser = SmartGraphTraverser(standard_model, data, device)
    enhanced_traverser = SmartGraphTraverser(enhanced_model, data, device)

    # Define algorithms to compare
    algorithms = {
        "BidirectionalBFS": lambda s, t, max_steps: bidirectional_bfs_wrapper(data, s, t, max_steps),
        "StandardGNN": lambda s, t, max_steps: standard_traverser.traverse(s, t, max_steps),
        "EnhancedGNN": lambda s, t, max_steps: enhanced_traverser.traverse(s, t, max_steps)
    }

    test_pairs = generate_test_pairs(data, num_pairs=30)

    # Run comparison
    results, summary = compare_algorithms(data, algorithms, test_pairs, max_steps=100)
    
    # Print results
    print("\nEvaluation complete. Results:")
    
    print("\nSummary statistics:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Visualize results
    from utils.evaluation import visualize_comparison, analyze_by_path_difficulty
    visualize_comparison(results, summary)
    
    # Analyze by path difficulty
    difficulty_stats = analyze_by_path_difficulty(results, summary)
    
    # Additional visualizations
    visualize_path_distances(results['path_length'], list(algorithms.keys()))
    visualize_performance_by_difficulty(difficulty_stats, list(algorithms.keys()))

if __name__ == "__main__":
    main()