import torch
import torch.nn.functional as F
from utils.data import neighbor_sampler

class BaseTraverser:
    """
    Base class for Wikipedia graph traversal.
    Provides common functionality for different traversal methods.
    """
    def __init__(self, model, data, device, num_neighbors=20, num_hops=2):
        self.model = model
        self.data = data
        self.device = device
        self.num_neighbors = num_neighbors
        self.num_hops = num_hops
        self.nodes_explored = 0
        
    def reset_exploration_counter(self):
        """Reset the counter for nodes explored during traversal"""
        self.nodes_explored = 0
        
    def get_node_embedding(self, node_idx):
        """
        Get embedding for a single node.
        
        Args:
            node_idx: Index of the node
            
        Returns:
            embedding: Node embedding vector
        """
        node_x = self.data.x[node_idx:node_idx+1].to(self.device)
        batch = torch.zeros(1, dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            self.model.eval()
            embedding = self.model(node_x, self.data.edge_index.to(self.device), batch)[0]
            
        return embedding
    
    def sample_neighbors(self, node_idx, num_neighbors=None, num_hops=None):
        """
        Sample neighbors for a node.
        
        Args:
            node_idx: Index of the node
            num_neighbors: Number of neighbors to sample (default: self.num_neighbors)
            num_hops: Number of hops to consider (default: self.num_hops)
            
        Returns:
            sampled_nodes: List of sampled neighbor node indices
        """
        if num_neighbors is None:
            num_neighbors = self.num_neighbors
        if num_hops is None:
            num_hops = self.num_hops
            
        edge_index_cpu = self.data.edge_index.cpu()
    
        sampled_nodes = neighbor_sampler(
            torch.tensor([node_idx], dtype=torch.long),
            edge_index_cpu,  # Use CPU tensor
            num_hops=num_hops,
            num_neighbors=num_neighbors
        )
        
        sampled_nodes = [n for n in sampled_nodes if n < self.data.x.size(0)]
    
        self.nodes_explored += len(sampled_nodes)
        return sampled_nodes
        
    def build_subgraph(self, nodes, include_extra_nodes=None):
        """
        Extract a subgraph for the given nodes.
        
        Args:
            nodes: List of node indices
            include_extra_nodes: Additional nodes to include (optional)
            
        Returns:
            sub_x: Subgraph node features
            sub_edge_index: Subgraph edge indices
            mapping: Mapping from original to subgraph indices
        """
        # Include extra nodes if specified
        if include_extra_nodes:
            node_set = set(nodes.tolist())
            for extra_node in include_extra_nodes:
                if extra_node not in node_set:
                    nodes = torch.cat([nodes, torch.tensor([extra_node], dtype=torch.long)])
        
        # Create mapping from original to subgraph indices
        mapping = {int(n): i for i, n in enumerate(nodes)}
        
        # Extract subgraph features
        sub_x = self.data.x[nodes].to(self.device)
        
        # Map edges
        edge_index = self.data.edge_index.to(self.device)
        sampled_nodes_device = nodes.to(self.device)
        
        # Create masks for source and target nodes
        src_mask = torch.isin(edge_index[0], sampled_nodes_device)
        dst_mask = torch.isin(edge_index[1], sampled_nodes_device)
        
        # Only keep edges where both source and target are in the subgraph
        edge_mask = src_mask & dst_mask
        sub_edge_index = edge_index[:, edge_mask]
        
        # Remap edge indices to subgraph indexing
        for d in (0, 1):
            for j in range(sub_edge_index.size(1)):
                node_idx = sub_edge_index[d, j].item()
                if node_idx in mapping:
                    sub_edge_index[d, j] = mapping[node_idx]
        
        return sub_x, sub_edge_index, mapping
    
    def score_subgraph_nodes(self, sub_x, sub_edge_index, sub_nodes, target_emb, exclude_nodes=None):
        """
        Score nodes in a subgraph based on their similarity to the target.
        
        Args:
            sub_x: Subgraph node features
            sub_edge_index: Subgraph edge indices
            sub_nodes: Original indices of subgraph nodes
            target_emb: Target node embedding
            exclude_nodes: Nodes to exclude from scoring (e.g., already visited)
            
        Returns:
            scores: Scores for each node
            node_embeddings: Embeddings for each node
        """
        sub_nodes = [n for n in sub_nodes if n < self.data.x.size(0)]
    
        # Check if we have any valid nodes
        if not sub_nodes:
            # Return dummy scores
            return torch.zeros(0, device=self.device), torch.zeros((0, self.model.hidden_dim), device=self.device)
        
        # Get node embeddings
        batch = torch.zeros(sub_x.size(0), dtype=torch.long).to(self.device)
        with torch.no_grad():
            self.model.eval()
            node_embeddings = self.model(sub_x, sub_edge_index, batch)
        
        # Score nodes based on target similarity
        scores = self.model.score_neighbors(node_embeddings, sub_nodes, target_emb)
        
        # Exclude specified nodes
        if exclude_nodes:
            for node in exclude_nodes:
                if node in sub_nodes:
                    idx = sub_nodes.index(node)
                    scores[idx] = float('-inf')
        
        return scores, node_embeddings
    
    def traverse(self, start_node_id, target_node_id=None, max_steps=10, method=None):
        """
        Base implementation for traversing from start to target.
        Simply follows the highest-scoring neighbor in each step.
        
        Args:
            start_node_id: ID of the starting node
            target_node_id: ID of the target node (optional)
            max_steps: Maximum number of steps to take
            method: Optional parameter (ignored in base traverser)
            
        Returns:
            path: List of node IDs in the traversal path
            nodes_explored: Number of nodes explored
        """
        self.reset_exploration_counter()
        
        # Convert original IDs to internal indices
        if start_node_id not in self.data.node_mapping:
            raise ValueError(f"Start node ID {start_node_id} not found in the graph")
        
        start_idx = self.data.node_mapping[start_node_id]
        
        # If target is specified, convert it too
        if target_node_id is not None:
            if target_node_id not in self.data.node_mapping:
                raise ValueError(f"Target node ID {target_node_id} not found in the graph")
            target_idx = self.data.node_mapping[target_node_id]
            target_emb = self.get_node_embedding(target_idx)
        else:
            target_idx = None
            target_emb = None
        
        # Initialize path
        path = [start_idx]
        visited = {start_idx}
        
        # Traverse
        for step in range(max_steps):
            current_idx = path[-1]
            
            # If we've reached the target, stop
            if target_idx is not None and current_idx == target_idx:
                break
            
            # Sample neighbors
            sampled_nodes = self.sample_neighbors(current_idx)
            
            # Filter out already visited nodes
            valid_nodes = [n.item() for n in sampled_nodes if n.item() not in visited]
            
            if not valid_nodes:
                break
            
            # Build subgraph for valid nodes
            sub_x, sub_edge_index, mapping = self.build_subgraph(
                torch.tensor(valid_nodes, dtype=torch.long)
            )
            
            # Score nodes
            scores, _ = self.score_subgraph_nodes(
                sub_x, sub_edge_index, valid_nodes, target_emb
            )
            
            # Select best node
            best_idx = torch.argmax(scores).item()
            next_idx = valid_nodes[best_idx]
            
            # Add to path and mark as visited
            path.append(next_idx)
            visited.add(next_idx)
        
        # Convert internal indices back to original node IDs
        return [self.data.reverse_mapping[idx] for idx in path], self.nodes_explored
    
    def id_to_idx(self, node_id):
        """Convert a node ID to its internal index"""
        if node_id not in self.data.node_mapping:
            raise ValueError(f"Node ID {node_id} not found in the graph")
        return self.data.node_mapping[node_id]
    
    def idx_to_id(self, node_idx):
        """Convert a node internal index to its ID"""
        return self.data.reverse_mapping[node_idx]
    
    def path_to_ids(self, path):
        """Convert a path of internal indices to node IDs"""
        return [self.idx_to_id(idx) for idx in path]