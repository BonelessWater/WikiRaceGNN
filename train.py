import torch
import torch.nn as nn
import numpy as np
import os
import random
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from torch.cuda.amp import autocast, GradScaler
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

from models import WikiGraphSAGE
from utils import (
    load_graph_data, 
    neighbor_sampler, 
    initialize_word2vec_model
)
from traversal.utils import bidirectional_bfs
from traversal import Word2VecEnhancedTraverser

def train_path_predictor(data, model, device, num_epochs=50, batch_size=32, 
                         num_train_pairs=1000, learning_rate=0.001, 
                         weight_decay=0.0001, validation_split=0.2):
    """
    Train the GNN model with a focus on path prediction and navigation.
    Includes more diverse training pairs and better loss functions.
    
    Args:
        data: Graph data object
        model: GNN model to train
        device: Computation device
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        num_train_pairs: Number of node pairs to generate for training
        learning_rate: Initial learning rate
        weight_decay: Weight decay for regularization
        validation_split: Fraction of data to use for validation
        
    Returns:
        model: Trained model
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Only use GradScaler with CUDA
    use_cuda = device.type == 'cuda'
    if use_cuda:
        scaler = GradScaler()
    
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )
    loss_fn = nn.CrossEntropyLoss()
    
    # Create output directory for models
    os.makedirs('models', exist_ok=True)
    
    # Precompute node embeddings for faster training
    print("Precomputing node embeddings...")
    with torch.no_grad():
        model.eval()
        x_device = data.x.to(device)
        edge_index_device = data.edge_index.to(device)
        batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
        full_h = model(x_device, edge_index_device, batch)
    
    # Generate diverse training pairs with varying path lengths
    print("Generating training pairs with diverse path lengths...")
    train_pairs = []
    all_nodes = list(range(data.x.size(0)))
    
    # Try to get equal numbers of short, medium, and long paths
    short_paths = []  # 1-2 hops
    medium_paths = []  # 3-5 hops
    long_paths = []   # 6+ hops
    
    with tqdm(total=num_train_pairs) as pbar:
        attempts = 0
        max_attempts = num_train_pairs * 10
        
        while (len(short_paths) + len(medium_paths) + len(long_paths) < num_train_pairs and 
               attempts < max_attempts):
            
            src = random.choice(all_nodes)
            tgt = random.choice(all_nodes)
            
            if src == tgt:
                attempts += 1
                continue
                
            path, _ = bidirectional_bfs(data, src, tgt)
            
            if not path:
                attempts += 1
                continue
                
            path_length = len(path)
            path_info = (src, tgt, path)
            
            # Categorize based on path length
            if path_length <= 3:
                if len(short_paths) < num_train_pairs // 3:
                    short_paths.append(path_info)
                    pbar.update(1)
            elif 4 <= path_length <= 6:
                if len(medium_paths) < num_train_pairs // 3:
                    medium_paths.append(path_info)
                    pbar.update(1)
            elif path_length >= 7:
                if len(long_paths) < num_train_pairs // 3:
                    long_paths.append(path_info)
                    pbar.update(1)
                    
            attempts += 1
            
            # Update progress
            pbar.set_postfix({
                'short': len(short_paths), 
                'medium': len(medium_paths), 
                'long': len(long_paths),
                'attempts': attempts
            })
    
    # Combine all path types
    train_pairs = short_paths + medium_paths + long_paths
    random.shuffle(train_pairs)
    
    print(f"Generated {len(short_paths)} short, {len(medium_paths)} medium, "
          f"and {len(long_paths)} long paths for training")
    
    # Split into training and validation sets
    val_size = int(len(train_pairs) * validation_split)
    val_pairs = train_pairs[:val_size]
    train_pairs = train_pairs[val_size:]
    
    print(f"Using {len(train_pairs)} pairs for training and {len(val_pairs)} pairs for validation")
    
    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    patience = 10
    patience_counter = 0
    
    print("Starting training...")
    for epoch in range(1, num_epochs+1):
        random.shuffle(train_pairs)
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        pbar = tqdm(range(0, len(train_pairs), batch_size),
                    desc=f"Epoch {epoch}/{num_epochs}")
        
        for i in pbar:
            batch_pairs = train_pairs[i : i+batch_size]
            optimizer.zero_grad()
            
            batch_loss = 0
            batch_correct = 0
            batch_total = 0
            
            for src, tgt, path in batch_pairs:
                # For short paths, train on the direct next hop
                if len(path) > 1:
                    next_node = path[1]  # The next node in the path from source
                    
                    # Sample a neighborhood subgraph
                    sampled = neighbor_sampler(
                        torch.tensor([src], dtype=torch.long),
                        data.edge_index,
                        num_hops=1, 
                        num_neighbors=30
                    ).tolist()
                    
                    # Ensure the next node is in the subgraph
                    if next_node not in sampled:
                        sampled.append(next_node)
                    
                    # Convert to tensor and build subgraph
                    sampled_tensor = torch.tensor(sampled, dtype=torch.long)
                    sub_x = data.x[sampled_tensor].to(device)
                    
                    # Create mapping from original indices to subgraph indices
                    mapping = {n: i for i, n in enumerate(sampled)}
                    
                    # Build edge index for the subgraph
                    edge_list = []
                    for e in range(data.edge_index.size(1)):
                        source = data.edge_index[0, e].item()
                        target = data.edge_index[1, e].item()
                        if source in mapping and target in mapping:
                            edge_list.append([mapping[source], mapping[target]])
                    
                    if edge_list:
                        sub_edge_index = torch.tensor(edge_list, dtype=torch.long).t().to(device)
                        
                        # Compute node embeddings
                        if use_cuda:
                            with autocast():
                                batch_tensor = torch.zeros(sub_x.size(0), dtype=torch.long).to(device)
                                sub_h = model(sub_x, sub_edge_index, batch_tensor)
                                
                                # Get target embedding
                                tgt_emb = full_h[tgt].detach()
                                
                                # Score neighbors and compute loss
                                logits = model.score_neighbors(sub_h, sampled, tgt_emb)
                                true_pos = mapping[next_node]
                                
                                loss = loss_fn(
                                    logits.unsqueeze(0),  # [1, M]
                                    torch.tensor([true_pos], device=device)
                                )
                        else:
                            # Normal forward pass without autocast for CPU
                            batch_tensor = torch.zeros(sub_x.size(0), dtype=torch.long).to(device)
                            sub_h = model(sub_x, sub_edge_index, batch_tensor)
                            
                            # Get target embedding
                            tgt_emb = full_h[tgt].detach()
                            
                            # Score neighbors and compute loss
                            logits = model.score_neighbors(sub_h, sampled, tgt_emb)
                            true_pos = mapping[next_node]
                            
                            loss = loss_fn(
                                logits.unsqueeze(0),  # [1, M]
                                torch.tensor([true_pos], device=device)
                            )
                            
                        # Calculate accuracy
                        pred = torch.argmax(logits).item()
                        if pred == true_pos:
                            batch_correct += 1
                        batch_total += 1
                        
                        batch_loss += loss
            
            # Skip empty batches
            if batch_total == 0:
                continue
                
            # Normalize batch loss
            batch_loss = batch_loss / batch_total
            
            # Backpropagate (with or without scaler based on device)
            if use_cuda:
                scaler.scale(batch_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                batch_loss.backward()
                optimizer.step()
            
            # Update metrics
            total_loss += batch_loss.item()
            correct_predictions += batch_correct
            total_predictions += batch_total
            
            accuracy = batch_correct / batch_total if batch_total > 0 else 0
            pbar.set_postfix({
                'loss': batch_loss.item(), 
                'accuracy': accuracy
            })
        
        # Calculate epoch metrics
        epoch_loss = total_loss / len(pbar) if len(pbar) > 0 else 0
        epoch_acc = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        print(f"Epoch {epoch}: Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for src, tgt, path in val_pairs:
                if len(path) > 1:
                    next_node = path[1]
                    
                    # Sample neighborhood
                    sampled = neighbor_sampler(
                        torch.tensor([src], dtype=torch.long),
                        data.edge_index,
                        num_hops=1, 
                        num_neighbors=30
                    ).tolist()
                    
                    if next_node not in sampled:
                        sampled.append(next_node)
                    
                    # Build subgraph
                    sampled_tensor = torch.tensor(sampled, dtype=torch.long)
                    sub_x = data.x[sampled_tensor].to(device)
                    
                    # Create mapping
                    mapping = {n: i for i, n in enumerate(sampled)}
                    
                    # Build edge index
                    edge_list = []
                    for e in range(data.edge_index.size(1)):
                        source = data.edge_index[0, e].item()
                        target = data.edge_index[1, e].item()
                        if source in mapping and target in mapping:
                            edge_list.append([mapping[source], mapping[target]])
                    
                    if edge_list:
                        sub_edge_index = torch.tensor(edge_list, dtype=torch.long).t().to(device)
                        
                        # Compute embeddings
                        batch_tensor = torch.zeros(sub_x.size(0), dtype=torch.long).to(device)
                        sub_h = model(sub_x, sub_edge_index, batch_tensor)
                        
                        # Score neighbors
                        tgt_emb = full_h[tgt].detach()
                        logits = model.score_neighbors(sub_h, sampled, tgt_emb)
                        true_pos = mapping[next_node]
                        
                        # Compute loss
                        loss = loss_fn(
                            logits.unsqueeze(0),
                            torch.tensor([true_pos], device=device)
                        )
                        val_loss += loss.item()
                        
                        # Calculate accuracy
                        pred = torch.argmax(logits).item()
                        if pred == true_pos:
                            val_correct += 1
                        val_total += 1
        
        # Calculate validation metrics
        val_loss = val_loss / val_total if val_total > 0 else float('inf')
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        print(f"Validation: Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        
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
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch
            }, f"models/checkpoint_epoch_{epoch}.pt")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}. Best epoch was {best_epoch} with val_loss {best_val_loss:.4f}")
            break
    
    # Load the best model
    model.load_state_dict(torch.load("models/best_model.pt", map_location=device))
    
    # Save final model
    torch.save(model.state_dict(), "models/enhanced_model_final.pt")
    
    return model

def main():
    """Main training function"""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Load graph data with Word2Vec embeddings
    edge_file = "data/wiki_edges.csv"  # Your generated edge file
    
    print("Loading graph data with Word2Vec embeddings...")
    data = load_graph_data(
        edge_file, 
        feature_dim=64, 
        ensure_connected=True,
        use_word2vec=True  # Enable Word2Vec embeddings
    )
    print(f"Loaded graph with {data.x.size(0)} nodes and {data.edge_index.size(1) // 2} edges")
    
    # Check if we can load a Word2Vec model
    word2vec_model = None
    try:
        # Try to load existing model
        model_path = os.path.join("data", "word2vec_model")
        if os.path.exists(model_path):
            print(f"Loading Word2Vec model from {model_path}")
            word2vec_model = Word2Vec.load(model_path)
            print(f"Loaded Word2Vec model with {len(word2vec_model.wv)} word vectors")
        else:
            print("Word2Vec model not found. Only GNN embeddings will be used.")
    except Exception as e:
        print(f"Error loading Word2Vec model: {e}")
    
    # Initialize GNN model
    input_dim = data.x.size(1)
    hidden_dim = 256
    output_dim = 64
    
    print("Initializing GNN model...")
    model = WikiGraphSAGE(
        input_dim, hidden_dim, output_dim,
        num_layers=4,
        word2vec_model=word2vec_model
    )
    
    # Load a pre-trained model if available
    model_path = "models/enhanced_model_final.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded pre-trained model from {model_path}")
    else:
        print("Pre-trained model not found. Using untrained model.")
    
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    # Create our Word2Vec-enhanced traverser
    print("Creating Word2Vec-enhanced traverser...")
    traverser = Word2VecEnhancedTraverser(
        model, data, device,
        beam_width=5,
        heuristic_weight=1.5,
        num_neighbors=20,
        num_hops=2
    )
    
    # Define a set of test cases
    num_test_cases = 5
    test_cases = []
    
    # Create test cases with different path lengths
    all_nodes = list(range(data.x.size(0)))
    
    print(f"Generating {num_test_cases} test cases...")
    for _ in range(num_test_cases):
        # Choose random source
        source_idx = random.choice(all_nodes)
        source_id = data.reverse_mapping[source_idx]
        
        # Find a target with a valid path
        found_target = False
        attempts = 0
        max_attempts = 100
        
        while not found_target and attempts < max_attempts:
            target_idx = random.choice(all_nodes)
            target_id = data.reverse_mapping[target_idx]
            
            # Skip if same node
            if target_idx == source_idx:
                attempts += 1
                continue
            
            # Check if there's a path
            path, _ = bidirectional_bfs(data, source_idx, target_idx)
            if path and 3 <= len(path) <= 10:  # Ensure reasonable path length
                test_cases.append((source_id, target_id, len(path)))
                found_target = True
            
            attempts += 1
    
    print(f"Generated {len(test_cases)} test cases")
    
    # Run each test case with different methods
    methods = ["bidirectional", "beam", "hybrid", "auto"]
    results = {}
    
    print("\nRunning traversal with different methods...")
    for i, (source_id, target_id, optimal_length) in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {source_id} → {target_id} (Optimal length: {optimal_length})")
        
        # Get the BFS baseline
        source_idx = data.node_mapping[source_id]
        target_idx = data.node_mapping[target_id]
        bfs_path, bfs_nodes = bidirectional_bfs(data, source_idx, target_idx)
        bfs_path_ids = [data.reverse_mapping[idx] for idx in bfs_path] if bfs_path else []
        
        print(f"BFS: {len(bfs_path_ids)} nodes, explored {bfs_nodes} nodes")
        
        # Run with different methods
        for method in methods:
            try:
                print(f"Running {method} traversal...")
                path, nodes_explored = traverser.traverse(
                    source_id, target_id, max_steps=30, method=method
                )
                
                # Record results
                if i not in results:
                    results[i] = {'bfs': (bfs_path_ids, bfs_nodes)}
                
                results[i][method] = (path, nodes_explored)
                
                # Print stats
                success = len(path) > 0 and path[-1] == target_id
                success_mark = "✓" if success else "✗"
                
                print(f"{method.capitalize()}: {success_mark} {len(path)} nodes, explored {nodes_explored} nodes")
                
                # Calculate improvement over BFS if successful
                if success and bfs_nodes > 0:
                    nodes_reduction = (bfs_nodes - nodes_explored) / bfs_nodes * 100
                    print(f"  {nodes_reduction:.2f}% fewer nodes explored than BFS")
                
            except Exception as e:
                print(f"Error in {method} traversal: {e}")
    
    # Create visualizations if possible
    try:
        from utils.visualization_manager import compare_paths_visualization
        
        # Choose a test case to visualize
        test_case = results[0]  # First test case
        source_id, target_id = test_cases[0][0], test_cases[0][1]
        
        # Prepare path data for visualization
        paths = {}
        paths["BFS"] = [data.node_mapping[node_id] for node_id in test_case['bfs'][0] if node_id in data.node_mapping]
        
        for method in methods:
            if method in test_case:
                path_ids = test_case[method][0]
                if path_ids:
                    paths[method.capitalize()] = [data.node_mapping[node_id] for node_id in path_ids if node_id in data.node_mapping]
        
        # Create visualization
        compare_paths_visualization(
            data, paths, 
            f"Path Comparison with Word2Vec Enhancement: {source_id} → {target_id}"
        )
        
        print("\nVisualization saved to 'plots/path_comparison.png'")
    except Exception as e:
        print(f"Error creating visualization: {e}")
    
    # Compute overall statistics
    print("\nOverall Statistics:")
    print("-" * 50)
    
    avg_reduction = {}
    success_rate = {}
    avg_path_length = {}
    
    for method in methods:
        # Success rate
        successes = 0
        path_lengths = []
        reduction_values = []
        
        for i in results:
            if method in results[i]:
                path, nodes = results[i][method]
                bfs_path, bfs_nodes = results[i]['bfs']
                
                if path and path[-1] == test_cases[i][1]:  # Successful traversal
                    successes += 1
                    path_lengths.append(len(path))
                    
                    if bfs_nodes > 0:
                        reduction = (bfs_nodes - nodes) / bfs_nodes * 100
                        reduction_values.append(reduction)
        
        # Calculate stats
        success_rate[method] = successes / len(results) if results else 0
        avg_path_length[method] = np.mean(path_lengths) if path_lengths else 0
        avg_reduction[method] = np.mean(reduction_values) if reduction_values else 0
        
        print(f"{method.capitalize()}:")
        print(f"  Success Rate: {success_rate[method]:.2%}")
        print(f"  Avg Path Length: {avg_path_length[method]:.2f}")
        print(f"  Avg Reduction in Nodes Explored: {avg_reduction[method]:.2f}%")
    
    return results

if __name__ == "__main__":
    main()