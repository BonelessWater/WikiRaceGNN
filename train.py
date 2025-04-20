import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from torch.cuda.amp import autocast, GradScaler

from models import WikiGraphSAGE, EnhancedWikiGraphSAGE
from utils import (
    load_graph_data, 
    neighbor_sampler, 
    plot_graph_sample, 
    visualize_embeddings,
)
from traversal.utils import bidirectional_bfs

def train_supervised(data, model, device, num_epochs=20, batch_size=16, num_pairs=500, 
                     learning_rate=0.001, weight_decay=0.0001, validation_split=0.1):
    """
    Train the GNN model in a supervised manner using paths as supervision.
    
    Args:
        data: Graph data object
        model: GNN model to train
        device: Computation device
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        num_pairs: Number of node pairs to generate for training
        learning_rate: Initial learning rate
        weight_decay: Weight decay for regularization
        validation_split: Fraction of data to use for validation
        
    Returns:
        model: Trained model
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = GradScaler()
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True,
        min_lr=1e-6
    )
    loss_fn = nn.CrossEntropyLoss()
    
    # Create output directory for models
    os.makedirs('models', exist_ok=True)
    
    # Precompute global embeddings for faster training
    print("Precomputing node embeddings...")
    with torch.no_grad():
        model.eval()
        full_h = model(
            data.x.to(device),
            data.edge_index.to(device),
            batch=torch.zeros(data.x.size(0), dtype=torch.long, device=device)
        )
    
    # Build (src, tgt, true_nbr) triplets
    print("Generating training triplets...")
    triplets = []
    all_idx = list(range(data.x.size(0)))
    
    with tqdm(total=num_pairs) as pbar:
        while len(triplets) < num_pairs:
            src = random.choice(all_idx)
            tgt = random.choice(all_idx)
            if src == tgt: 
                continue
                
            path, _ = bidirectional_bfs(data, src, tgt)
            if len(path) < 2: 
                continue
                
            triplets.append((src, tgt, path[1]))
            pbar.update(1)
    
    # Split into training and validation sets
    val_size = int(len(triplets) * validation_split)
    random.shuffle(triplets)
    val_triplets = triplets[:val_size]
    train_triplets = triplets[val_size:]
    
    print(f"Generated {len(train_triplets)} training and {len(val_triplets)} validation triplets")
    
    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    patience = 5
    patience_counter = 0
    
    print("Starting training...")
    for epoch in range(1, num_epochs+1):
        random.shuffle(train_triplets)
        model.train()
        total_loss = 0
        pbar = tqdm(range(0, len(train_triplets), batch_size),
                    desc=f"Epoch {epoch}/{num_epochs}")
        
        for i in pbar:
            batch = train_triplets[i : i+batch_size]
            optimizer.zero_grad()
            
            batch_loss = 0
            for src, tgt, true_nbr in batch:
                # Sample & build subgraph (with src & true_nbr forced in)
                sampled = neighbor_sampler(
                    torch.tensor([src], dtype=torch.long),
                    data.edge_index,
                    num_hops=2, 
                    num_neighbors=20
                ).tolist()
                
                if src not in sampled: 
                    sampled.append(src)
                if true_nbr not in sampled: 
                    sampled.append(true_nbr)
                
                # Convert to tensor
                sampled_tensor = torch.tensor(sampled, dtype=torch.long)
                
                # Build subgraph
                sub_x = data.x[sampled_tensor].to(device)
                
                # Create mapping from original indices to subgraph indices
                mapping = {n: i for i, n in enumerate(sampled)}
                
                # Prepare edges for the subgraph
                edge_index = data.edge_index.to(device)
                
                # Create masks for source and target nodes
                sampled_tensor_device = sampled_tensor.to(device)
                src_mask = torch.isin(edge_index[0], sampled_tensor_device)
                dst_mask = torch.isin(edge_index[1], sampled_tensor_device)
                
                # Only keep edges where both source and target are in the subgraph
                edge_mask = src_mask & dst_mask
                sub_edge_index = edge_index[:, edge_mask]
                
                # Remap edge indices to subgraph indexing
                for d in (0, 1):
                    for j in range(sub_edge_index.size(1)):
                        node_idx = sub_edge_index[d, j].item()
                        if node_idx in mapping:
                            sub_edge_index[d, j] = mapping[node_idx]
                
                # Compute subgraph embeddings
                with autocast():
                    batch_tensor = torch.zeros(sub_x.size(0), dtype=torch.long).to(device)
                    sub_h = model(sub_x, sub_edge_index, batch_tensor)
                    
                    # Use the precomputed embedding for the target
                    tgt_emb = full_h[tgt]
                    
                    # Score neighbors and compute loss
                    logits = model.score_neighbors(sub_h, sampled, tgt_emb)
                    true_pos = mapping[true_nbr]
                    
                    loss = loss_fn(
                        logits.unsqueeze(0),  # [1, M]
                        torch.tensor([true_pos], device=device)
                    )
                    
                    batch_loss += loss
            
            # Normalize batch loss
            batch_loss = batch_loss / len(batch)
            
            # Backpropagate
            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += batch_loss.item()
            pbar.set_postfix({'loss': total_loss / ((i//batch_size)+1)})
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt, true_nbr in val_triplets:
                # Similar to training, but for validation
                sampled = neighbor_sampler(
                    torch.tensor([src], dtype=torch.long),
                    data.edge_index,
                    num_hops=2, 
                    num_neighbors=20
                ).tolist()
                
                if src not in sampled: 
                    sampled.append(src)
                if true_nbr not in sampled: 
                    sampled.append(true_nbr)
                
                sampled_tensor = torch.tensor(sampled, dtype=torch.long)
                sub_x, sub_edge_index, mapping = build_subgraph(data, sampled_tensor, device)
                
                batch_tensor = torch.zeros(sub_x.size(0), dtype=torch.long).to(device)
                sub_h = model(sub_x, sub_edge_index, batch_tensor)
                
                tgt_emb = full_h[tgt]
                logits = model.score_neighbors(sub_h, sampled, tgt_emb)
                true_pos = mapping[true_nbr]
                
                loss = loss_fn(
                    logits.unsqueeze(0),  # [1, M]
                    torch.tensor([true_pos], device=device)
                )
                
                val_loss += loss.item()
        
        val_loss /= len(val_triplets)
        print(f"Epoch {epoch}, Train Loss: {total_loss/len(pbar):.4f}, Val Loss: {val_loss:.4f}")
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), "models/best_model.pt")
            print(f"Saved new best model with validation loss: {val_loss:.4f}")
        else:
            patience_counter += 1
        
        # Save periodic checkpoint
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch
            }, f"models/checkpoint_epoch_{epoch}.pt")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}. Best epoch was {best_epoch} with val_loss {best_val_loss:.4f}")
            break
    
    # Load the best model
    model.load_state_dict(torch.load("models/best_model.pt"))
    return model


def build_subgraph(data, nodes, device):
    """
    Utility to extract x, edge_index, and a mapping {orig_idx:sub_idx}.
    
    Args:
        data: Graph data object
        nodes: List of node indices to include in the subgraph
        device: Computation device
        
    Returns:
        sub_x: Node features for the subgraph
        sub_ei: Edge indices for the subgraph
        mapping: Dictionary mapping original indices to subgraph indices
    """
    mapping = {int(n): i for i, n in enumerate(nodes)}
    
    # Extract subgraph features
    sub_x = data.x[nodes].to(device)
    
    # Filter edges
    edge_index = data.edge_index.to(device)
    nodes_tensor = nodes.to(device)
    
    # Create masks for source and target nodes
    src_mask = torch.isin(edge_index[0], nodes_tensor)
    dst_mask = torch.isin(edge_index[1], nodes_tensor)
    
    # Only keep edges where both source and target are in the subgraph
    edge_mask = src_mask & dst_mask
    sub_ei = edge_index[:, edge_mask]
    
    # Remap edge indices to subgraph indexing
    for d in (0, 1):
        for j in range(sub_ei.size(1)):
            node_idx = sub_ei[d, j].item()
            if node_idx in mapping:
                sub_ei[d, j] = mapping[node_idx]
    
    return sub_x, sub_ei, mapping


def main():
    """Main training function"""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs('plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Load graph data
    edge_file = "data/wiki_edges.csv"  # Replace with your actual file path
    max_nodes = 1000  # Adjust based on your needs and resources
    data = load_graph_data(edge_file, feature_dim=64, max_nodes=max_nodes, ensure_connected=True)
    print(f"Loaded graph with {data.x.size(0)} nodes and {data.edge_index.size(1) // 2} edges")
    
    # Visualize sample of the graph
    plot_graph_sample(data)
    
    # Feature normalization
    scaler = StandardScaler()
    data.x = torch.from_numpy(
        scaler.fit_transform(data.x.numpy())
    ).float()
    
    # Initialize model
    input_dim = data.x.size(1)
    hidden_dim = 256
    output_dim = 64
    
    # Create both standard and enhanced models
    standard_model = WikiGraphSAGE(input_dim, hidden_dim, output_dim, num_layers=4)
    enhanced_model = EnhancedWikiGraphSAGE(input_dim, hidden_dim, output_dim, num_layers=4)
    
    # Train standard model
    print("\nTraining standard model...")
    standard_model = train_supervised(
        data, 
        standard_model, 
        device, 
        num_epochs=20, 
        batch_size=32, 
        num_pairs=1000
    )
    
    # Save final model
    torch.save(standard_model.state_dict(), "models/standard_model_final.pt")
    
    # Train enhanced model
    print("\nTraining enhanced model...")
    enhanced_model = train_supervised(
        data, 
        enhanced_model, 
        device, 
        num_epochs=20, 
        batch_size=32, 
        num_pairs=1000
    )
    
    # Save final model
    torch.save(enhanced_model.state_dict(), "models/enhanced_model_final.pt")
    
    # Visualize embeddings
    visualize_embeddings(standard_model, data, device)
    visualize_embeddings(enhanced_model, data, device, sample_size=500)
    
    print("Training complete. Models saved to 'models/' directory.")


if __name__ == "__main__":
    main()