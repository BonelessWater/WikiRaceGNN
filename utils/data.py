import torch
import pandas as pd
import numpy as np
import csv
import random
import os
from torch_geometric.data import Data
from traversal.utils import bidirectional_bfs
from tqdm import tqdm
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from collections import defaultdict



def load_graph_data_no_pandas(
    edge_file,
    feature_dim=64,
    max_nodes=None,
    ensure_connected=True,
    use_word2vec=False,      # you can re‑enable your gensim path later
):
    # 1) slurp in the raw edges
    edges = []
    with open(edge_file, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)      # skip the header
        for row in reader:
            if len(row) != 2:
                continue
            src, dst = row
            edges.append((src.strip(), dst.strip()))

    # 2) build a node→index map
    nodes = set(u for u, v in edges) | set(v for u, v in edges)
    if max_nodes is not None:
        # (optional) you can sample/prune here to max_nodes
        nodes = set(list(nodes)[:max_nodes])
        edges = [(u, v) for u, v in edges if u in nodes and v in nodes]

    node_list = list(nodes)
    idx = {n:i for i,n in enumerate(node_list)}

    # 3) build the edge_index (and make it undirected)
    row, col = [], []
    for u, v in edges:
        i, j = idx[u], idx[v]
        row += [i, j]
        col += [j, i]
    edge_index = torch.tensor([row, col], dtype=torch.long)

    # 4) dummy features (random).  Re‑wire your Word2Vec bit after this if you want.
    x = torch.randn((len(node_list), feature_dim), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    data.node_mapping = idx
    data.reverse_mapping = {i:n for n,i in idx.items()}
    # 5) build an adjacency list for fast neighbor lookups
    adj_list = defaultdict(list)
    # edge_index is [2 x num_edges], and is already undirected
    for src, dst in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        adj_list[src].append(dst)

    data.adj_list = adj_list
    return data


def load_graph_data(edge_file, feature_dim=64, max_nodes=None, ensure_connected=True, use_word2vec=True):
    """
    Load graph data from edge list file with option to reduce to a smaller connected graph.
    Use Word2Vec for node embeddings instead of random embeddings.
    
    Args:
        edge_file: Path to the CSV edge list
        feature_dim: Dimension of node features to generate
        max_nodes: Maximum number of nodes to include (if None, use all)
        ensure_connected: Ensure the sampled graph is connected
        use_word2vec: Whether to use Word2Vec for embeddings
        
    Returns:
        Data object with graph information
    """
    try:
        # Try to read edge list
        df = pd.read_csv(edge_file, dtype=str)
    except FileNotFoundError:
        # If file not found, create a simple test graph
        print(f"Edge file {edge_file} not found. Creating a test graph.")
        # Create a simple grid graph
        import networkx as nx
        G = nx.grid_2d_graph(10, 10)
        # Relabel nodes to integers
        G = nx.convert_node_labels_to_integers(G)
        # Create edge list
        edges = list(G.edges())
        df = pd.DataFrame(edges, columns=['id1', 'id2'])
    
    # Check if node metadata exists (for titles/urls)
    base_dir = os.path.dirname(edge_file)
    node_meta_path = os.path.join(base_dir, "wiki_nodes.json")
    has_node_metadata = os.path.exists(node_meta_path)
    
    if has_node_metadata:
        import json
        with open(node_meta_path, 'r', encoding='utf-8') as f:
            node_metadata = json.load(f)
        print(f"Loaded node metadata for {len(node_metadata)} nodes")
    
    # Load or check for Word2Vec model
    word2vec_model = None
    embedding_file = os.path.join(base_dir, "wiki_embeddings.npy")
    embeddings = None
    
    if use_word2vec:
        w2v_model_path = os.path.join(base_dir, "word2vec_model")
        if os.path.exists(w2v_model_path):
            try:
                print(f"Loading Word2Vec model from {w2v_model_path}")
                word2vec_model = Word2Vec.load(w2v_model_path)
                print(f"Loaded Word2Vec model with {len(word2vec_model.wv)} word vectors")
            except Exception as e:
                print(f"Error loading Word2Vec model: {e}")
                word2vec_model = None
        
        # Load pre-computed embeddings if available
        if os.path.exists(embedding_file):
            try:
                print(f"Loading pre-computed embeddings from {embedding_file}")
                embeddings = np.load(embedding_file)
                print(f"Loaded embeddings of shape {embeddings.shape}")
            except Exception as e:
                print(f"Error loading embeddings: {e}")
                embeddings = None
    
    if max_nodes is not None:
        # Create networkx graph for connectivity analysis
        import networkx as nx
        G = nx.Graph()
        for _, row in df.iterrows():
            G.add_edge(row['id1'], row['id2'])
        
        if ensure_connected:
            # Find largest connected component if needed
            largest_cc = max(nx.connected_components(G), key=len)
            
            # If the largest component is smaller than max_nodes, use the whole component
            if len(largest_cc) < max_nodes:
                print(f"Warning: Largest connected component has only {len(largest_cc)} nodes")
                print(f"Using all {len(largest_cc)} nodes from largest component")
                selected_nodes = list(largest_cc)
            else:
                # Start with a random node from the largest component
                import random
                start_node = random.choice(list(largest_cc))
                selected_nodes = [start_node]
                frontier = list(nx.neighbors(G, start_node))
                
                # Breadth-first selection to ensure connectivity
                while len(selected_nodes) < max_nodes and frontier:
                    next_node = frontier.pop(0)
                    if next_node not in selected_nodes:
                        selected_nodes.append(next_node)
                        # Add its neighbors to the frontier
                        for neighbor in nx.neighbors(G, next_node):
                            if neighbor not in selected_nodes and neighbor not in frontier:
                                frontier.append(neighbor)
        else:
            # Simply select random nodes
            import random
            selected_nodes = random.sample(list(G.nodes()), min(max_nodes, len(G.nodes())))
        
        # Filter edges to only include selected nodes
        filtered_edges = []
        for _, row in df.iterrows():
            if row['id1'] in selected_nodes and row['id2'] in selected_nodes:
                filtered_edges.append((row['id1'], row['id2']))
                
        # Create a new dataframe with the filtered edges
        df = pd.DataFrame(filtered_edges, columns=['id1', 'id2'])
        
        print(f"Reduced graph to {len(selected_nodes)} nodes and {len(filtered_edges)} edges")

    # Get unique node IDs
    df['id1'] = df['id1'].astype(str)
    df['id2'] = df['id2'].astype(str)

    # Then proceed with the unique call
    node_ids = np.unique(df[['id1', 'id2']].values.flatten())
    num_nodes = len(node_ids)
    
    # Create mapping from original IDs to consecutive indices
    node_mapping = {node_id: idx for idx, node_id in enumerate(node_ids)}
    
    # Convert edges to use consecutive indices
    edge_index = torch.tensor([
        [node_mapping[id1] for id1 in df['id1'].values],
        [node_mapping[id2] for id2 in df['id2'].values]
    ], dtype=torch.long)
    
    # Make the graph undirected
    from torch_geometric.utils import to_undirected
    edge_index = to_undirected(edge_index)
    
    # Generate node features based on the approach selected
    if embeddings is not None and len(embeddings) >= num_nodes:
        # Use pre-computed embeddings
        print("Using pre-computed embeddings")
        x = torch.tensor(embeddings[:num_nodes], dtype=torch.float)
    elif word2vec_model is not None and has_node_metadata:
        # Generate embeddings using Word2Vec model
        print("Generating embeddings using Word2Vec model")
        x = torch.zeros((num_nodes, feature_dim), dtype=torch.float)
        
        # Preprocess titles and generate embeddings
        for node_id, idx in node_mapping.items():
            if str(node_id) in node_metadata:
                title = node_metadata[str(node_id)]['title']
                # Preprocess title
                tokens = simple_preprocess(title.replace("_", " "), deacc=True)
                
                if tokens:
                    # Get and average word vectors
                    vectors = []
                    for token in tokens:
                        if token in word2vec_model.wv:
                            vectors.append(word2vec_model.wv[token])
                    
                    if vectors:
                        # Average vectors to get node embedding
                        x[idx] = torch.tensor(np.mean(vectors, axis=0), dtype=torch.float)
                    else:
                        # Random embedding if no tokens in vocabulary
                        x[idx] = torch.randn(feature_dim)
                else:
                    # Random embedding if no tokens
                    x[idx] = torch.randn(feature_dim)
            else:
                # Random embedding if no metadata
                x[idx] = torch.randn(feature_dim)
    else:
        # Generate random node features if no Word2Vec model available
        print("Using random embeddings")
        x = torch.randn((num_nodes, feature_dim))
    
    # Create a PyG Data object
    data = Data(x=x, edge_index=edge_index)
    
    # Store the mapping for reference
    data.node_mapping = node_mapping
    data.reverse_mapping = {idx: node_id for node_id, idx in node_mapping.items()}
    
    # Create an adjacency list for faster BFS and neighbor lookup
    adj_list = {}
    for i in range(edge_index.size(1)):
        source = edge_index[0, i].item()
        target = edge_index[1, i].item()
        
        if source not in adj_list:
            adj_list[source] = []
        if target not in adj_list:
            adj_list[target] = []
            
        adj_list[source].append(target)
    
    data.adj_list = adj_list
    
    # Add node titles and URLs if available
    if has_node_metadata:
        node_titles = {}
        node_urls = {}
        
        for node_id, idx in node_mapping.items():
            if str(node_id) in node_metadata:
                node_titles[idx] = node_metadata[str(node_id)]['title']
                node_urls[idx] = node_metadata[str(node_id)]['url']
        
        data.node_titles = node_titles
        data.node_urls = node_urls
    
    return data
   
def neighbor_sampler(nodes, edge_index, num_hops=2, num_neighbors=10):
    """
    Sample neighbors for a batch of nodes.
    This is a simplified version of neighbor sampling with control over breadth and depth.
    
    Args:
        nodes: Tensor of node indices
        edge_index: Edge indices tensor [2, E]
        num_hops: Number of hops to expand
        num_neighbors: Max number of neighbors to sample per node
        
    Returns:
        sampled_nodes: Tensor of sampled node indices
    """
    sampled_nodes = set(nodes.tolist())
    current_nodes = set(nodes.tolist())
    
    for _ in range(num_hops):
        next_nodes = set()
        for node in current_nodes:
            # Find neighbors of the current node
            neighbors = edge_index[1][edge_index[0] == node].tolist()
            
            # Sample a subset if there are too many neighbors
            if len(neighbors) > num_neighbors:
                neighbors = random.sample(neighbors, num_neighbors)
            
            next_nodes.update(neighbors)
            sampled_nodes.update(neighbors)
        
        current_nodes = next_nodes
    
    return torch.tensor(list(sampled_nodes), dtype=torch.long)

def generate_test_pairs(data, num_pairs=30):
    """
    Generate test pairs with balanced path lengths: short, medium, and long.
    This ensures comprehensive evaluation across different path difficulties.
    """
    all_nodes = list(range(data.x.size(0)))
    short_pairs = []  # 2-3 hops
    medium_pairs = []  # 4-6 hops
    long_pairs = []   # 7+ hops
    
    attempts = 0
    max_attempts = 5000
    target_per_category = num_pairs // 3
    
    print("Generating test pairs with varied path lengths...")
    
    with tqdm(total=num_pairs) as pbar:
        while (len(short_pairs) < target_per_category or 
              len(medium_pairs) < target_per_category or 
              len(long_pairs) < target_per_category) and attempts < max_attempts:
            
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
            
            # Categorize based on path length
            if path_length <= 3 and len(short_pairs) < target_per_category:
                if (source, target) not in short_pairs:
                    short_pairs.append((source, target))
                    pbar.update(1)
            elif 4 <= path_length <= 6 and len(medium_pairs) < target_per_category:
                if (source, target) not in medium_pairs:
                    medium_pairs.append((source, target))
                    pbar.update(1)
            elif path_length >= 7 and len(long_pairs) < target_per_category:
                if (source, target) not in long_pairs:
                    long_pairs.append((source, target))
                    pbar.update(1)
                
            attempts += 1
            
            # Update progress
            pbar.set_postfix({
                'short': len(short_pairs), 
                'medium': len(medium_pairs), 
                'long': len(long_pairs),
                'attempts': attempts
            })
    
    # Combine all pairs
    all_pairs = short_pairs + medium_pairs + long_pairs
    random.shuffle(all_pairs)
    
    if len(short_pairs) < target_per_category:
        print(f"Warning: Could only find {len(short_pairs)} short paths")
    if len(medium_pairs) < target_per_category:
        print(f"Warning: Could only find {len(medium_pairs)} medium paths")
    if len(long_pairs) < target_per_category:
        print(f"Warning: Could only find {len(long_pairs)} long paths")
        
    print(f"Generated {len(short_pairs)} short, {len(medium_pairs)} medium, and {len(long_pairs)} long paths")
    return all_pairs, {'short': short_pairs, 'medium': medium_pairs, 'long': long_pairs}
