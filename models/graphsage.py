import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class WikiGraphSAGE(torch.nn.Module):
    """
    GraphSAGE model for Wikipedia graph traversal with Word2Vec embeddings.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4, dropout=0.2, word2vec_model=None):
        super().__init__()
        from torch_geometric.nn import SAGEConv
        
        self.embedding = torch.nn.Linear(input_dim, hidden_dim)
        self.convs = torch.nn.ModuleList([SAGEConv(hidden_dim, hidden_dim)
                                    for _ in range(num_layers)])
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Store Word2Vec model for additional support
        self.word2vec_model = word2vec_model
        
        # Node scorer (for selecting next hops)
        self.scorer = torch.nn.Sequential(
            torch.nn.Linear(2*hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1)
        )
        
        # Output layer
        self.output = torch.nn.Linear(hidden_dim, output_dim)
    
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
        import torch.nn.functional as F
        
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
        Enhanced with Word2Vec similarity when available.
        
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

class EnhancedWikiGraphSAGE(torch.nn.Module):
    """
    Enhanced GraphSAGE with multi-scale representations and attention mechanisms
    for more effective path planning.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4, dropout=0.2,
                 use_attention=True, use_layer_norm=True, use_skip=True, word2vec_model=None):
        super().__init__()
        from torch_geometric.nn import SAGEConv
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_attention = use_attention
        self.use_layer_norm = use_layer_norm
        self.use_skip = use_skip
        
        # Store Word2Vec model for additional support
        self.word2vec_model = word2vec_model
        
        # Initial embedding
        self.embedding = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norms = torch.nn.ModuleList([
                torch.nn.LayerNorm(hidden_dim)
                for _ in range(num_layers)
            ])
        
        # GraphSAGE convolutions with attention
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        # Attention mechanism for node scoring
        if use_attention:
            self.attention = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim * 2, hidden_dim // 2),  # Input is 2x hidden_dim because we concatenate embeddings
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim // 2, 1)
            )
        
        # Path scoring with context awareness
        self.path_scorer = torch.nn.Sequential(
            torch.nn.Linear(3*hidden_dim, hidden_dim),
            torch.nn.ReLU(), 
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, 1)
        )
        
        # Multi-scale fusion layer
        self.multi_scale_fusion = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * (num_layers + 1), hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output layer
        self.output = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass with multi-scale fusion and skip connections.
        """
        import torch.nn.functional as F
        
        # Initial embedding
        h = self.embedding(x)
        
        # Store intermediate representations for multi-scale fusion
        all_reps = [h]
        
        # Apply GraphSAGE convolutions with skip connections
        for i, conv in enumerate(self.convs):
            # Apply convolution
            h_new = conv(h, edge_index)
            
            # Apply layer norm if enabled
            if self.use_layer_norm:
                h_new = self.layer_norms[i](h_new)
            
            # Apply skip connection if enabled
            if self.use_skip and i > 0:
                h_new = h_new + h
            
            # Apply activation and dropout
            h = F.relu(h_new)
            h = F.dropout(h, p=self.dropout, training=self.training)
            
            # Store for multi-scale fusion
            all_reps.append(h)
        
        # Multi-scale fusion if more than one layer
        if self.num_layers > 0:
            # Concatenate representations from different scales
            multi_scale = torch.cat(all_reps, dim=1)
            h = self.multi_scale_fusion(multi_scale)
        
        return h
    
    def score_neighbors(self, sub_h, sub_nodes, target_emb):
        """
        Score neighbor nodes with attention mechanism.
        """
        # Calculate basic similarity
        similarities = F.cosine_similarity(sub_h, target_emb.unsqueeze(0), dim=1)
        
        if self.use_attention:
            # Calculate attention scores - fix the dimension issue
            # First, ensure target_emb is properly expanded
            expanded_target = target_emb.expand(sub_h.size(0), -1)
            
            # Debug: Print shapes
            # print(f"sub_h shape: {sub_h.shape}, target_emb shape: {target_emb.shape}, expanded: {expanded_target.shape}")
            
            # Concatenate along the feature dimension (dim=1)
            attention_input = torch.cat([sub_h, expanded_target], dim=1)
            
            # Make sure attention layers match the input size
            # The first layer of self.attention expects input of size [batch, sub_h.size(1) + target_emb.size(0)]
            attention_scores = self.attention(attention_input).squeeze(-1)
            
            # Combine similarity with attention
            combined_scores = similarities + F.sigmoid(attention_scores)
            return combined_scores
        else:
            return similarities
    
    def predict_multi_hop(self, node_emb, target_emb, num_hops=3):
        """
        Estimate a multi-hop direction vector toward the target.
        """
        # Concatenate embeddings for context-aware prediction
        combined = torch.cat([
            node_emb,
            target_emb,
            torch.abs(target_emb - node_emb)  # Distance features
        ], dim=1)
        
        # Predict direction vector
        direction = self.path_scorer(combined)
        
        # Normalize
        direction_norm = torch.norm(direction, p=2)
        if direction_norm > 0:
            return direction / direction_norm
        else:
            return direction
        
          