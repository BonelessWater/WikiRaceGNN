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
        