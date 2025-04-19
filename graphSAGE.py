
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import SAGEConv
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler

from utils import (neighbor_sampler, 
                   load_graph_data, 
                   plot_graph_sample, 
                   visualize_results, 
                   bidirectional_bfs)

class AttentionReadout(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionReadout, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, batch):
        # Compute attention scores
        attention_scores = self.attention(x)
        attention_weights = F.softmax(attention_scores, dim=0)
        
        # Apply attention weights
        weighted_x = x * attention_weights
        
        # Aggregate based on batch
        output = torch_geometric.nn.global_add_pool(weighted_x, batch)
        return output

class WikiGraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(WikiGraphSAGE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Input embedding layer
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        # Attention readout
        self.attention = AttentionReadout(hidden_dim)
        
        # Output layer
        self.output = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, edge_index, batch):
        # Initial embedding
        x = self.embedding(x)
        
        # GraphSAGE layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        
        # Attention readout
        x = self.attention(x, batch)
        
        # Output layer
        x = self.output(x)
        
        return x

class WikiTraverser:
    def __init__(self, model, data, device, num_neighbors=10, num_hops=2):
        self.model = model
        self.data = data
        self.device = device
        self.num_neighbors = num_neighbors
        self.num_hops = num_hops
        self.nodes_explored = 0
        
    def reset_exploration_counter(self):
        """Reset the counter for nodes explored during traversal"""
        self.nodes_explored = 0
        
    def traverse(self, start_node_id, target_node_id=None, max_steps=10):
        """
        Traverse the Wikipedia graph starting from start_node_id,
        optionally targeting a specific node.
        
        Returns:
            path: List of node IDs in the traversal path
            nodes_explored: Number of nodes explored during traversal
        """
        self.reset_exploration_counter()
        
        # Convert original IDs to internal indices
        if start_node_id not in self.data.node_mapping:
            raise ValueError(f"Start node ID {start_node_id} not found in the graph")
        
        start_idx = self.data.node_mapping[start_node_id]
        target_idx = self.data.node_mapping[target_node_id] if target_node_id in self.data.node_mapping else None
        
        # Initialize path
        path = [start_idx]
        visited = {start_idx}
        
        for _ in range(max_steps):
            current_idx = path[-1]
            
            # If we've reached the target, stop
            if target_idx is not None and current_idx == target_idx:
                break
            
            # Sample neighbors
            sampled_nodes = neighbor_sampler(
                torch.tensor([current_idx], dtype=torch.long),
                self.data.edge_index,
                num_hops=self.num_hops,
                num_neighbors=self.num_neighbors
            )
            
            # Update exploration counter
            self.nodes_explored += len(sampled_nodes)
            
            # Extract subgraph
            subgraph_x = self.data.x[sampled_nodes].to(self.device)
            
            # Get node mapping in the subgraph
            subgraph_mapping = {node.item(): i for i, node in enumerate(sampled_nodes)}
            current_subgraph_idx = subgraph_mapping[current_idx]
            
            # Prepare edges for the subgraph
            edge_index = self.data.edge_index.to(self.device)
            # Move sampled_nodes to device before comparison
            sampled_nodes_device = sampled_nodes.to(self.device)
            mask = (edge_index[0].unsqueeze(1) == sampled_nodes_device.unsqueeze(0)).any(dim=1) & \
                   (edge_index[1].unsqueeze(1) == sampled_nodes_device.unsqueeze(0)).any(dim=1)
            subgraph_edge_index = edge_index[:, mask]
            
            # Map edge indices to subgraph indices
            for i in range(2):
                for j in range(subgraph_edge_index.size(1)):
                    subgraph_edge_index[i, j] = subgraph_mapping[subgraph_edge_index[i, j].item()]
            
            # Forward pass
            with torch.no_grad():
                self.model.eval()
                batch = torch.zeros(subgraph_x.size(0), dtype=torch.long).to(self.device)
                node_scores = self.model(subgraph_x, subgraph_edge_index, batch)
            
            # If target is specified, use it for node scoring
            if target_idx is not None:
                target_embedding = self.data.x[target_idx].to(self.device)
                similarity_scores = F.cosine_similarity(subgraph_x, target_embedding.unsqueeze(0), dim=1)
                node_scores = similarity_scores.reshape(-1, 1)
            
            # Mask already visited nodes
            for node in visited:
                if node in subgraph_mapping:
                    node_scores[subgraph_mapping[node]] = float('-inf')
            
            # Select next node
            next_subgraph_idx = torch.argmax(node_scores).item()
            next_idx = sampled_nodes[next_subgraph_idx].item()
            
            # Add to path and mark as visited
            path.append(next_idx)
            visited.add(next_idx)
        
        # Convert internal indices back to original node IDs
        return [self.data.reverse_mapping[idx] for idx in path], self.nodes_explored

    def traverse_beam_astar(self,
                            start_node_id,
                            target_node_id,
                            max_steps=10,
                            beam_width=3,
                            alpha=0.5):
        """
        Beam + A* style search.  
        Returns the best path found and total nodes_explored.
        """
        self.reset_exploration_counter()

        # map to indices
        start_idx = self.data.node_mapping[start_node_id]
        target_idx = self.data.node_mapping[target_node_id]

        # get target embedding once
        target_emb = self.data.x[target_idx].to(self.device)

        # each beam item: (path_list, cumulative_score)
        beams = [([start_idx], 0.0)]
        visited_sets = [{start_idx}]

        for step in range(max_steps):
            all_candidates = []
            all_visited   = []

            for (path, score), visited in zip(beams, visited_sets):
                curr = path[-1]
                # sample neighbors
                nbrs = neighbor_sampler(
                    torch.tensor([curr], dtype=torch.long),
                    self.data.edge_index,
                    num_hops=self.num_hops,
                    num_neighbors=self.num_neighbors
                ).tolist()

                self.nodes_explored += len(nbrs)

                # prepare subgraph & model scores once per beam
                sub_x, sub_ei, mapping = build_subgraph(self.data, nbrs+[curr], self.device)
                # run model
                with torch.no_grad():
                    logits = self.model(sub_x, sub_ei,
                                        torch.zeros(sub_x.size(0), dtype=torch.long, device=self.device))
                # extract scores & normalize to log‑probs
                log_p = F.log_softmax(logits[:,0], dim=0)  # assume scalar output

                for nbr in nbrs:
                    if nbr in visited:
                        continue
                    idx_in_sub = mapping[nbr]
                    model_score = log_p[idx_in_sub].item()
                    # heuristic: cosine similarity
                    h = 1.0 - F.cosine_similarity(
                        sub_x[idx_in_sub].unsqueeze(0),
                        target_emb.unsqueeze(0),
                        dim=1
                    ).item()
                    combined = (1-alpha)*model_score - alpha*h

                    all_candidates.append((path+[nbr], score + combined))
                    all_visited.append(visited | {nbr})

            # select top‑B beams
            if not all_candidates:
                break
            # sort by descending score
            sorted_idx = sorted(range(len(all_candidates)),
                                key=lambda i: all_candidates[i][1],
                                reverse=True)[:beam_width]
            beams = [all_candidates[i] for i in sorted_idx]
            visited_sets = [all_visited[i]  for i in sorted_idx]

            # check for target
            for path, _ in beams:
                if path[-1] == target_idx:
                    return [self.data.reverse_mapping[i] for i in path], self.nodes_explored

        # no beam reached target — return highest‑scoring
        best_path, _ = beams[0]
        return [self.data.reverse_mapping[i] for i in best_path], self.nodes_explored


def build_subgraph(data, nodes, device):
    """
    Utility to extract x, edge_index, and a mapping {orig_idx:sub_idx}.
    """
    mapping = {n:i for i,n in enumerate(nodes)}
    # mask edges
    ei = data.edge_index.to(device)
    mask = ((ei[0].unsqueeze(1)==torch.tensor(nodes,device=device)) &
            (ei[1].unsqueeze(1)==torch.tensor(nodes,device=device))).any(0)
    sub_ei = ei[:,mask]
    # remap indices
    for d in (0,1):
        for j in range(sub_ei.size(1)):
            sub_ei[d,j] = mapping[int(sub_ei[d,j])]
    sub_x = data.x[nodes].to(device)
    return sub_x, sub_ei, mapping


def train_wiki_gnn(data, model, device, num_epochs=100, batch_size=32):
    """
    Train the GNN model using mini-batch training.
    For simplicity, we're using a self-supervised approach where the model
    predicts the features of neighboring nodes.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=5,
                                                gamma=0.5)
    
    # Create random train/val split
    num_nodes = data.x.size(0)
    train_idx = np.arange(num_nodes)
    val_idx = train_idx.copy()
    np.random.shuffle(train_idx)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        
        # Process in batches
        pbar = tqdm(range(0, train_idx.shape[0], batch_size), desc=f"Epoch {epoch+1}/{num_epochs}")
        total_loss = 0
        
        for i in pbar:
            # Get batch indices
            batch_idx = train_idx[i:i+batch_size]
            optimizer.zero_grad()
            
            # Sample neighbors for this batch
            sampled_nodes = neighbor_sampler(
                torch.tensor(batch_idx, dtype=torch.long),
                data.edge_index,
                num_hops=4,
                num_neighbors=10
            )
            
            # Create subgraph
            subgraph_x = data.x[sampled_nodes].to(device)
            
            # Create mapping from original indices to subgraph indices
            subgraph_mapping = {node.item(): i for i, node in enumerate(sampled_nodes)}
            
            # Map batch indices to subgraph indices
            batch_subgraph_idx = torch.tensor([subgraph_mapping[idx] for idx in batch_idx if idx in subgraph_mapping], 
                                             dtype=torch.long)
            
            # Prepare edges for the subgraph
            edge_index = data.edge_index.to(device)
            # Move sampled_nodes to device before comparison
            sampled_nodes_device = sampled_nodes.to(device)
            mask = (edge_index[0].unsqueeze(1) == sampled_nodes_device.unsqueeze(0)).any(dim=1) & \
                   (edge_index[1].unsqueeze(1) == sampled_nodes_device.unsqueeze(0)).any(dim=1)
            subgraph_edge_index = edge_index[:, mask]
            
            # Map edge indices to subgraph indices
            for i in range(2):
                for j in range(subgraph_edge_index.size(1)):
                    subgraph_edge_index[i, j] = subgraph_mapping[subgraph_edge_index[i, j].item()]
            
            # Self-supervised task: predict node features
            batch = torch.zeros(subgraph_x.size(0), dtype=torch.long).to(device)
            output = model(subgraph_x, subgraph_edge_index, batch)
            
            # Use the first n dimensions as prediction targets (simplified)
            target_dim = min(model.output_dim, subgraph_x.size(1))
            loss = F.mse_loss(output[:, :target_dim], subgraph_x[batch_subgraph_idx, :target_dim])
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': total_loss / (i // batch_size + 1)})
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for i in range(0, val_idx.shape[0], batch_size):
                batch_idx = val_idx[i:i+batch_size]
                
                # Sample neighbors
                sampled_nodes = neighbor_sampler(
                    torch.tensor(batch_idx, dtype=torch.long),
                    data.edge_index,
                    num_hops=2,
                    num_neighbors=10
                )
                
                # Create subgraph
                subgraph_x = data.x[sampled_nodes].to(device)
                
                # Create mapping
                subgraph_mapping = {node.item(): i for i, node in enumerate(sampled_nodes)}
                batch_subgraph_idx = torch.tensor([subgraph_mapping[idx] for idx in batch_idx if idx in subgraph_mapping], 
                                                 dtype=torch.long)
                
                # Prepare edges
                edge_index = data.edge_index.to(device)
                # Move sampled_nodes to device before comparison
                sampled_nodes_device = sampled_nodes.to(device)
                mask = (edge_index[0].unsqueeze(1) == sampled_nodes_device.unsqueeze(0)).any(dim=1) & \
                       (edge_index[1].unsqueeze(1) == sampled_nodes_device.unsqueeze(0)).any(dim=1)
                subgraph_edge_index = edge_index[:, mask]
                
                # Map edge indices
                for i in range(2):
                    for j in range(subgraph_edge_index.size(1)):
                        subgraph_edge_index[i, j] = subgraph_mapping[subgraph_edge_index[i, j].item()]
                
                # Forward pass
                batch = torch.zeros(subgraph_x.size(0), dtype=torch.long).to(device)
                output = model(subgraph_x, subgraph_edge_index, batch)
                
                # Compute loss
                target_dim = min(model.output_dim, subgraph_x.size(1))
                loss = F.mse_loss(output[:, :target_dim], subgraph_x[batch_subgraph_idx, :target_dim])
                val_loss += loss.item()
            
            avg_val_loss = val_loss / (val_idx.shape[0] // batch_size + 1)
            print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {avg_val_loss:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "best_wiki_gnn_model.pt")

def compare_with_bfs(data, model, device, num_test_pairs=10):
    """
    Compare GNN traversal with Bidirectional BFS.
    
    Args:
        data: Graph data object
        model: Trained GNN model
        device: Computation device
        num_test_pairs: Number of source-target pairs to test
        
    Returns:
        Dictionary with comparison results
    """
    traverser = WikiTraverser(model, data, device)
    
    results = {
        'source': [],
        'target': [],
        'gnn_path_length': [],
        'bfs_path_length': [],
        'gnn_nodes_explored': [],
        'bfs_nodes_explored': [],
        'gnn_success': [],
        'is_shortest_path': []
    }
    
    # Select random node pairs for testing
    num_nodes = data.x.size(0)
    all_nodes = list(range(num_nodes))
    
    # Create a set of connected node pairs using BFS exploration
    connected_pairs = []
    for _ in range(min(num_test_pairs * 2, num_nodes)):  # Try more pairs than needed
        if len(connected_pairs) >= num_test_pairs:
            break
            
        source = random.choice(all_nodes)
        target = random.choice(all_nodes)
        
        # Skip if same node or already tested
        if source == target or (source, target) in connected_pairs:
            continue
            
        # Check if there's a path between them
        bfs_path, _ = bidirectional_bfs(data, source, target)
        if bfs_path:  # If a path exists
            connected_pairs.append((source, target))
            
    # If we couldn't find enough connected pairs, add random pairs
    while len(connected_pairs) < num_test_pairs:
        source = random.choice(all_nodes)
        target = random.choice(all_nodes)
        if source != target and (source, target) not in connected_pairs:
            connected_pairs.append((source, target))
    
    # Test each pair
    for source, target in connected_pairs:
        source_id = data.reverse_mapping[source]
        target_id = data.reverse_mapping[target]
        
        print(f"Testing path from node {source_id} to {target_id}")
        
        # GNN traversal
        gnn_path, gnn_nodes_explored = traverser.traverse(source_id, target_id, max_steps=20)
        
        # Convert back to internal indices for comparison
        gnn_path_idx = [data.node_mapping[node_id] for node_id in gnn_path]
        
        # Check if target was reached
        gnn_success = gnn_path_idx[-1] == target if gnn_path else False
        
        # Bidirectional BFS
        bfs_path, bfs_nodes_explored = bidirectional_bfs(data, source, target)
        
        # Compare paths
        is_shortest = len(gnn_path_idx) == len(bfs_path) if gnn_success and bfs_path else False
        
        # Record results
        results['source'].append(source_id)
        results['target'].append(target_id)
        results['gnn_path_length'].append(len(gnn_path_idx) if gnn_path else 0)
        results['bfs_path_length'].append(len(bfs_path) if bfs_path else 0)
        results['gnn_nodes_explored'].append(gnn_nodes_explored)
        results['bfs_nodes_explored'].append(bfs_nodes_explored)
        results['gnn_success'].append(gnn_success)
        results['is_shortest_path'].append(is_shortest)
        
        print(f"GNN path length: {results['gnn_path_length'][-1]}, Nodes explored: {gnn_nodes_explored}")
        print(f"BFS path length: {results['bfs_path_length'][-1]}, Nodes explored: {bfs_nodes_explored}")
        print(f"GNN found target: {gnn_success}, Is shortest path: {is_shortest}")
        print("-" * 50)
    
    # Calculate summary statistics
    summary = {
        'avg_gnn_path_length': np.mean([l for l in results['gnn_path_length'] if l > 0]),
        'avg_bfs_path_length': np.mean([l for l in results['bfs_path_length'] if l > 0]),
        'avg_gnn_nodes_explored': np.mean(results['gnn_nodes_explored']),
        'avg_bfs_nodes_explored': np.mean(results['bfs_nodes_explored']),
        'gnn_success_rate': np.mean(results['gnn_success']),
        'shortest_path_rate': np.mean([r for r, s in zip(results['is_shortest_path'], results['gnn_success']) if s])
    }
    
    return results, summary

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load graph data with reduced number of nodes
    edge_file = "croc_edges.csv"
    # Use a reduced dataset (100 nodes) for testing
    max_nodes = 100  # Set to None to use the full dataset
    data = load_graph_data(edge_file, feature_dim=64, max_nodes=max_nodes, ensure_connected=True)
    print(f"Loaded graph with {data.x.size(0)} nodes and {data.edge_index.size(1) // 2} edges (bidirectional)")

    scaler = StandardScaler()
    # .numpy() → scale → back to torch.Tensor
    data.x = torch.from_numpy(
        scaler.fit_transform(data.x.numpy())
    ).float()

    print("Feature means:", data.x.mean(dim=0)[:5], "…")
    print("Feature stds:", data.x.std(dim=0)[:5], "…")

    # Visualize a sample of the graph
    plot_graph_sample(data)
    
    # Initialize model
    input_dim = data.x.size(1)  # Feature dimension
    hidden_dim = 256
    output_dim = 64
    model = WikiGraphSAGE(input_dim, hidden_dim, output_dim, num_layers=2)
    
    # Train the model
    print("Training the model...")
    train_wiki_gnn(data, model, device, num_epochs=10, batch_size=16)
    
    # Load the best model
    model.load_state_dict(torch.load("best_wiki_gnn_model.pt"))
    model = model.to(device)
    
    # Compare GNN traversal with BFS
    print("Comparing GNN traversal with Bidirectional BFS...")
    results, summary = compare_with_bfs(data, model, device, num_test_pairs=10)
    
    # Visualize results
    visualize_results(results, summary)
    
    # Display summary
    print("\nPerformance Summary:")
    print(f"GNN Success Rate: {summary['gnn_success_rate']*100:.2f}%")
    print(f"GNN Shortest Path Rate: {summary['shortest_path_rate']*100:.2f}%")
    print(f"Average GNN Path Length: {summary['avg_gnn_path_length']:.2f}")
    print(f"Average BFS Path Length: {summary['avg_bfs_path_length']:.2f}")
    print(f"Average GNN Nodes Explored: {summary['avg_gnn_nodes_explored']:.2f}")
    print(f"Average BFS Nodes Explored: {summary['avg_bfs_nodes_explored']:.2f}")
    print(f"Exploration Efficiency Ratio (BFS/GNN): {summary['avg_bfs_nodes_explored']/summary['avg_gnn_nodes_explored']:.2f}")

    # Example traversal
    traverser = WikiTraverser(model, data, device)
    
    # Select a random source-target pair
    num_nodes = data.x.size(0)
    source_idx = random.randint(0, num_nodes-1)
    target_idx = random.randint(0, num_nodes-1)
    while source_idx == target_idx:
        target_idx = random.randint(0, num_nodes-1)
    
    source_id = data.reverse_mapping[source_idx]
    target_id = data.reverse_mapping[target_idx]
    
    print(f"\nExample traversal from node {source_id} to node {target_id}:")
    path, nodes_explored = traverser.traverse(source_id, target_id, max_steps=20)
    
    print("GNN Path:")
    for i, node_id in enumerate(path):
        print(f"Step {i}: Node {node_id}")
    print(f"GNN explored {nodes_explored} nodes")
    
    print("\nBFS Path:")
    bfs_path, bfs_nodes = bidirectional_bfs(data, source_idx, target_idx)
    for i, node_idx in enumerate(bfs_path):
        node_id = data.reverse_mapping[node_idx]
        print(f"Step {i}: Node {node_id}")
    print(f"BFS explored {bfs_nodes} nodes")
    
    # Additional analysis: Plot distribution of node degrees
    degrees = []
    for i in range(data.x.size(0)):
        neighbors = data.adj_list.get(i, [])
        degrees.append(len(neighbors))
    
    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=20)
    plt.title("Distribution of Node Degrees")
    plt.xlabel("Degree")
    plt.ylabel("Number of Nodes")
    plt.savefig("plots/degree_distribution.png")
    plt.close()
    
    # Compare performance on nodes with different degrees
    high_degree_nodes = [i for i, d in enumerate(degrees) if d > np.median(degrees)]
    low_degree_nodes = [i for i, d in enumerate(degrees) if d <= np.median(degrees)]
    
    print("\nAnalyzing performance based on node degree:")
    
    # Test paths between high-degree nodes
    high_high_results = []
    for _ in range(5):
        src = random.choice(high_degree_nodes)
        tgt = random.choice(high_degree_nodes)
        while src == tgt:
            tgt = random.choice(high_degree_nodes)
            
        src_id = data.reverse_mapping[src]
        tgt_id = data.reverse_mapping[tgt]
        
        gnn_path, gnn_nodes = traverser.traverse(src_id, tgt_id, max_steps=20)
        bfs_path, bfs_nodes = bidirectional_bfs(data, src, tgt)
        
        high_high_results.append((len(gnn_path), len(bfs_path), gnn_nodes, bfs_nodes))
    
    # Test paths between low-degree nodes
    low_low_results = []
    for _ in range(5):
        src = random.choice(low_degree_nodes)
        tgt = random.choice(low_degree_nodes)
        while src == tgt:
            tgt = random.choice(low_degree_nodes)
            
        src_id = data.reverse_mapping[src]
        tgt_id = data.reverse_mapping[tgt]
        
        gnn_path, gnn_nodes = traverser.traverse(src_id, tgt_id, max_steps=20)
        bfs_path, bfs_nodes = bidirectional_bfs(data, src, tgt)
        
        low_low_results.append((len(gnn_path), len(bfs_path), gnn_nodes, bfs_nodes))
    
    # Print results
    print("\nHigh degree to high degree node paths:")
    print(f"Avg GNN path length: {np.mean([r[0] for r in high_high_results]):.2f}")
    print(f"Avg BFS path length: {np.mean([r[1] for r in high_high_results]):.2f}")
    print(f"Avg GNN nodes explored: {np.mean([r[2] for r in high_high_results]):.2f}")
    print(f"Avg BFS nodes explored: {np.mean([r[3] for r in high_high_results]):.2f}")
    
    print("\nLow degree to low degree node paths:")
    print(f"Avg GNN path length: {np.mean([r[0] for r in low_low_results]):.2f}")
    print(f"Avg BFS path length: {np.mean([r[1] for r in low_low_results]):.2f}")
    print(f"Avg GNN nodes explored: {np.mean([r[2] for r in low_low_results]):.2f}")
    print(f"Avg BFS nodes explored: {np.mean([r[3] for r in low_low_results]):.2f}")

if __name__ == "__main__":
    main()