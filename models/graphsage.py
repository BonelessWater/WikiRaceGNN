import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class WikiGraphSAGE(nn.Module):
    """
    GraphSAGE model for Wikipedia graph traversal.
    Combines GraphSAGE layers with specialized scoring mechanisms.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4, dropout=0.2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList([SAGEConv(hidden_dim, hidden_dim)
                                    for _ in range(num_layers)])
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Node scorer (for selecting next hops)
        self.scorer = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Output layer
        self.output = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass to produce node embeddings.
        
        Args:
            x: Node features [N, input_dim]
            edge_index: Edge indices [2, E]
            batch: Batch assignment for nodes [N] (optional)
            
        Returns:
            h: Node embeddings [N, hidden_dim]
        """
        # Initial embedding
        h = F.relu(self.embedding(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Apply GraphSAGE convolutions
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        return h
    
    def score_neighbors(self, sub_h, sub_nodes, target_emb):
        """
        Score neighbor nodes based on their similarity to the target.
        
        Args:
            sub_h: [M, hidden_dim] embeddings for subgraph nodes
            sub_nodes: list of original node indices for those embeddings
            target_emb: [hidden_dim] embedding of the target node
            
        Returns:
            logits: [M] scores for each subgraph node
        """
        # Replicate target_emb to shape [M, hidden_dim]
        tgt_rep = target_emb.unsqueeze(0).expand(sub_h.size(0), -1)
        feats = torch.cat([sub_h, tgt_rep], dim=1)   # [M, 2*hidden_dim]
        logits = self.scorer(feats).squeeze(1)       # [M]
        return logits


class EnhancedWikiGraphSAGE(WikiGraphSAGE):
    """
    Enhanced version of WikiGraphSAGE with additional capabilities for traversal.
    Includes specialized scoring mechanisms and embedding caching.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4, dropout=0.2):
        super().__init__(input_dim, hidden_dim, output_dim, num_layers, dropout)
        
        # Additional scoring mechanism for path finding
        self.path_scorer = nn.Sequential(
            nn.Linear(3*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Multi-hop embedding transformation
        self.multi_hop_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def score_path(self, curr_node_emb, next_node_emb, target_emb):
        """
        Score a potential path step based on current node, next node, and target.
        
        Args:
            curr_node_emb: [hidden_dim] embedding of current node
            next_node_emb: [hidden_dim] embedding of potential next node
            target_emb: [hidden_dim] embedding of target node
            
        Returns:
            score: Scalar score for the potential path step
        """
        combined = torch.cat([
            curr_node_emb.unsqueeze(0), 
            next_node_emb.unsqueeze(0), 
            target_emb.unsqueeze(0)
        ], dim=1)  # [1, 3*hidden_dim]
        
        return self.path_scorer(combined).item()
    
    def predict_multi_hop(self, node_emb, target_emb, num_hops=3):
        """
        Estimate a multi-hop direction vector toward the target.
        
        Args:
            node_emb: Current node embedding
            target_emb: Target node embedding
            num_hops: Number of hops to plan ahead
            
        Returns:
            direction_vector: Predicted direction in embedding space
        """
        # Transform the embeddings for multi-hop planning
        node_transformed = self.multi_hop_transform(node_emb)
        target_transformed = self.multi_hop_transform(target_emb)
        
        # Compute direction vector (normalized)
        direction = target_transformed - node_transformed
        direction_norm = torch.norm(direction, p=2)
        
        if direction_norm > 0:
            return direction / direction_norm
        else:
            return direction
        
class SmartGraphTraverser:
    """
    A traverser that uses the trained GNN models to guide the search
    with optimizations to beat BidirectionalBFS
    """
    def __init__(self, model, data, device):
        self.model = model
        self.data = data
        self.device = device
        self.embedding_cache = {}
        
        # Ensure model is on the correct device
        self.model = self.model.to(self.device)
        
        # Put model in eval mode
        self.model.eval()
        
    def get_node_embedding(self, node_idx):
        """Get node embedding using the trained model"""
        # Check cache first
        if node_idx in self.embedding_cache:
            return self.embedding_cache[node_idx]
            
        with torch.no_grad():
            # Get node features and put them on the right device
            node_x = self.data.x[node_idx].unsqueeze(0).to(self.device)
            
            # Create batch tensor for a single node
            batch = torch.zeros(1, dtype=torch.long, device=self.device)
            
            try:
                # Run just the embedding layer
                embedding = self.model.embedding(node_x)
                
                # Cache and return
                self.embedding_cache[node_idx] = embedding
                return embedding
            except Exception as e:
                print(f"Error in get_node_embedding: {e}")
                # Return a random embedding as fallback
                return torch.randn(1, 256, device=self.device)  # Adjust size if needed
    
    def get_similarity(self, emb1, emb2):
        """Calculate cosine similarity between two embeddings"""
        try:
            # Ensure both are on same device
            if emb1.device != emb2.device:
                emb2 = emb2.to(emb1.device)
                
            # Calculate cosine similarity
            return torch.nn.functional.cosine_similarity(emb1, emb2).item()
        except Exception as e:
            print(f"Error in get_similarity: {e}")
            return 0.0  # Default similarity
    
    def get_neighbors_with_scores(self, node_idx, target_idx):
        """Get neighbors of a node and score them based on distance to target"""
        try:
            # Get target embedding
            target_emb = self.get_node_embedding(target_idx)
            
            # Get neighbors from adjacency list
            neighbors = self.data.adj_list.get(node_idx, [])
            
            # Score each neighbor based on similarity to target
            scored_neighbors = []
            for neighbor in neighbors:
                # Get embedding
                neighbor_emb = self.get_node_embedding(neighbor)
                
                # Calculate similarity score
                similarity = self.get_similarity(neighbor_emb, target_emb)
                
                scored_neighbors.append((neighbor, similarity))
            
            # Sort by score (highest first)
            scored_neighbors.sort(key=lambda x: x[1], reverse=True)
            
            return scored_neighbors
        except Exception as e:
            print(f"Error in get_neighbors_with_scores: {e}")
            return []
        
    def traverse(self, source_id, target_id, max_steps=30):
        """Perform smart traversal using the trained model"""
        try:
            # Convert IDs to indices
            if source_id not in self.data.node_mapping or target_id not in self.data.node_mapping:
                return [], 0
                
            source_idx = self.data.node_mapping[source_id]
            target_idx = self.data.node_mapping[target_id]
            
            # Early exit if source and target are the same
            if source_idx == target_idx:
                return [source_id], 1
                
            # Use regular BFS but guided by the learned embeddings
            queue = [(source_idx, [source_idx])]
            visited = {source_idx}
            nodes_explored = 1
            
            while queue and nodes_explored < max_steps:
                current, path = queue.pop(0)
                
                if current == target_idx:
                    # Path found, convert to node IDs
                    path_ids = [self.data.reverse_mapping[idx] for idx in path]
                    return path_ids, nodes_explored
                
                # Get and score neighbors
                scored_neighbors = self.get_neighbors_with_scores(current, target_idx)
                
                for neighbor, _ in scored_neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        nodes_explored += 1
                        queue.append((neighbor, path + [neighbor]))
            
            return [], nodes_explored
        except Exception as e:
            print(f"Error in traverse: {e}")
            return [], 0



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
    
