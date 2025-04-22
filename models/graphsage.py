import torch
import torch.nn.functional as F

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
     
          