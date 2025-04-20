import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geo_nn

class AttentionReadout(nn.Module):
    """
    Attention-based readout layer that applies attention weights to node embeddings.
    This helps to focus on more important nodes when aggregating node embeddings.
    """
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
        output = geo_nn.global_add_pool(weighted_x, batch)
        return output


class MultiheadAttentionLayer(nn.Module):
    """
    Multi-head attention layer for more sophisticated attention mechanisms.
    Can be used in place of simpler attention mechanisms for better performance.
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super(MultiheadAttentionLayer, self).__init__()
        assert hidden_dim % num_heads == 0, "Hidden dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, batch=None):
        batch_size = x.size(0) if batch is None else batch.max().item() + 1
        seq_len = x.size(0) // batch_size
        
        # Reshape input for multi-head attention
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape for output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(attn_output).view(batch_size * seq_len, self.hidden_dim)
        
        return output