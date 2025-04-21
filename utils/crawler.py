import os
import torch
import pandas as pd
import json
from torch_geometric.data import Data
from utils.wikibuilder import create_wiki_edge_list

def convert_wiki_to_wikiracegnn_format(data_dir="../data"):
    """
    Convert the Wiki graph data to a format compatible with WikiRaceGNN.
    
    Args:
        data_dir: Directory containing the Wiki graph data
        
    Returns:
        tuple: (data, edge_list_path, id_mapping_path)
    """
    # Check if data directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Check if PyTorch Geometric data already exists
    pt_path = os.path.join(data_dir, "wiki_graph.pt")
    if os.path.exists(pt_path):
        print(f"Loading existing PyTorch Geometric data from {pt_path}")
        data = torch.load(pt_path)
        edge_list_path = os.path.join(data_dir, "wiki_edges.csv")
        id_mapping_path = os.path.join(data_dir, "wiki_id_mapping.csv")
        return data, edge_list_path, id_mapping_path
    
    # Check if required files exist
    edge_list_path = os.path.join(data_dir, "wiki_edges.csv")
    node_meta_path = os.path.join(data_dir, "wiki_nodes.json")
    
    if not os.path.exists(edge_list_path):
        raise FileNotFoundError(f"Edge list file not found: {edge_list_path}")
    
    if not os.path.exists(node_meta_path):
        raise FileNotFoundError(f"Node metadata file not found: {node_meta_path}")
    
    # Load edge list
    print(f"Loading edge list from {edge_list_path}")
    edges_df = pd.read_csv(edge_list_path)
    
    # Load node metadata
    print(f"Loading node metadata from {node_meta_path}")
    with open(node_meta_path, 'r', encoding='utf-8') as f:
        nodes = json.load(f)
    
    # Create node mapping
    print("Creating node mappings")
    node_titles = list(nodes.keys())
    node_mapping = {title: i for i, title in enumerate(node_titles)}
    reverse_mapping = {i: title for i, title in enumerate(node_titles)}
    
    # Save ID mapping
    id_mapping_path = os.path.join(data_dir, "wiki_id_mapping.csv")
    id_mapping_df = pd.DataFrame({
        'id': range(len(node_titles)),
        'title': node_titles,
        'url': [nodes[title]['url'] for title in node_titles]
    })
    id_mapping_df.to_csv(id_mapping_path, index=False)
    
    # Create edge index
    print("Creating edge index")
    edge_index = []
    for _, row in edges_df.iterrows():
        source = row['id1']
        target = row['id2']
        if source in node_mapping and target in node_mapping:
            edge_index.append([node_mapping[source], node_mapping[target]])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    
    # Create node features
    print("Creating node feature matrix")
    x = torch.tensor([nodes[title]['embedding'] for title in node_titles], dtype=torch.float)
    
    # Create PyTorch Geometric Data object
    print("Creating PyTorch Geometric Data object")
    data = Data(x=x, edge_index=edge_index)
    
    # Add mappings for reference
    data.node_mapping = node_mapping
    data.reverse_mapping = reverse_mapping
    
    # Create adjacency list for faster access
    print("Creating adjacency list")
    adjacency_list = {}
    for i in range(edge_index.size(1)):
        source = edge_index[0, i].item()
        target = edge_index[1, i].item()
        
        if source not in adjacency_list:
            adjacency_list[source] = []
        
        adjacency_list[source].append(target)
    
    data.adj_list = adjacency_list
    
    # Save the data
    print(f"Saving PyTorch Geometric data to {pt_path}")
    torch.save(data, pt_path)
    
    print(f"Conversion complete! Created graph with {data.num_nodes} nodes and {data.num_edges} edges")
    return data, edge_list_path, id_mapping_path

def run_wikiracegnn_with_wiki_data(data_dir="../data", mode="traverse", visualize=True):
    """
    Run WikiRaceGNN using the Wiki graph data.
    
    Args:
        data_dir: Directory containing the Wiki graph data
        mode: Mode to run WikiRaceGNN in ('train', 'evaluate', or 'traverse')
        visualize: Whether to visualize the results
    """
    # Convert data to WikiRaceGNN format
    data, edge_list_path, _ = convert_wiki_to_wikiracegnn_format(data_dir)
    
    # Import main function from WikiRaceGNN
    try:
        # Try to import the main function
        from main import main
        
        # Set up command line arguments
        import sys
        sys.argv = ['main.py']
        sys.argv.extend(['--edge_file', edge_list_path])
        sys.argv.extend(['--mode', mode])
        
        if visualize:
            sys.argv.append('--visualize')
        
        # Run WikiRaceGNN
        print(f"Running WikiRaceGNN in {mode} mode with Wiki graph data")
        main()
        
    except ImportError:
        print("Could not import WikiRaceGNN. Make sure the main.py file is accessible.")
    except Exception as e:
        print(f"Error running WikiRaceGNN: {e}")

def select_random_traversal_pair(data_dir="../data", min_path_length=3, max_path_length=10):
    """
    Select a random pair of nodes for traversal.
    
    Args:
        data_dir: Directory containing the Wiki graph data
        min_path_length: Minimum path length between the nodes
        max_path_length: Maximum path length between the nodes
        
    Returns:
        tuple: (source_id, target_id) or None if no suitable pair is found
    """
    # Convert data to WikiRaceGNN format
    data, _, _ = convert_wiki_to_wikiracegnn_format(data_dir)
    
    # Import bidirectional BFS from WikiRaceGNN
    try:
        from traversal.utils import bidirectional_bfs
    except ImportError:
        print("Could not import bidirectional_bfs from WikiRaceGNN.")
        return None
    
    # Sample node pairs until a valid pair is found
    import random
    max_attempts = 100
    
    for _ in range(max_attempts):
        # Select random source and target
        source_idx = random.randint(0, data.num_nodes - 1)
        target_idx = random.randint(0, data.num_nodes - 1)
        
        # Skip if same node
        if source_idx == target_idx:
            continue
        
        # Check if there's a path between them
        path, _ = bidirectional_bfs(data, source_idx, target_idx)
        
        # Check if path length is within the desired range
        if path and min_path_length <= len(path) <= max_path_length:
            source_id = data.reverse_mapping[source_idx]
            target_id = data.reverse_mapping[target_idx]
            
            print(f"Selected traversal pair: {source_id} → {target_id}")
            print(f"Path length: {len(path)}")
            return source_id, target_id
    
    print("Could not find a suitable traversal pair within the maximum number of attempts.")
    return None

def visualize_wiki_graph(data_dir="../data", sample_size=100):
    """
    Visualize the Wiki graph using NetworkX.
    
    Args:
        data_dir: Directory containing the Wiki graph data
        sample_size: Number of nodes to show in the visualization
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
    except ImportError:
        print("NetworkX or Matplotlib not installed. Install with: pip install networkx matplotlib")
        return
    
    # Load edge list
    edge_list_path = os.path.join(data_dir, "wiki_edges.csv")
    if not os.path.exists(edge_list_path):
        print(f"Edge list file not found: {edge_list_path}")
        return
    
    # Load ID mapping for node labels
    id_mapping_path = os.path.join(data_dir, "wiki_id_mapping.csv")
    if os.path.exists(id_mapping_path):
        id_mapping = pd.read_csv(id_mapping_path)
        node_labels = {row['id']: row['title'] for _, row in id_mapping.iterrows()}
    else:
        node_labels = None
    
    # Load edges
    edges_df = pd.read_csv(edge_list_path)
    
    # Create graph
    G = nx.from_pandas_edgelist(edges_df, 'id1', 'id2', create_using=nx.DiGraph())

    # Sample a connected subgraph if the graph is too large
    if len(G) > sample_size:
        # Try to get a connected component
        try:
            components = list(nx.weakly_connected_components(G))
            if components:
                largest_cc = max(components, key=len)
                if len(largest_cc) > sample_size:
                    # Choose a random starting node
                    import random
                    start_node = random.choice(list(largest_cc))
                    
                    # Get a BFS tree from that node
                    subgraph_nodes = list(nx.bfs_tree(G, start_node, depth_limit=int(sample_size/10)))[:sample_size]
                    G = G.subgraph(subgraph_nodes)
                else:
                    G = G.subgraph(largest_cc)
        except Exception as e:
            print(f"Error creating connected subgraph: {e}")
            # Fall back to random sampling
            sample_nodes = list(G.nodes())[:sample_size]
            G = G.subgraph(sample_nodes)
    
    # Create plot
    plt.figure(figsize=(12, 12))
    
    # Use spring layout for visualization
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes and edges
    nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True)
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color='lightblue')
    
    # Draw labels if available and not too many nodes
    if node_labels and len(G) <= 50:
        labels = {n: node_labels.get(n, str(n)) for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    
    plt.title(f'Wikipedia Graph Sample ({len(G)} nodes)')
    plt.axis('off')
    
    # Save and show
    output_path = os.path.join(data_dir, "graph_visualization.png")
    plt.savefig(output_path, dpi=300)
    print(f"Graph visualization saved to {output_path}")
    plt.close()

def find_path_between_pages(source_title, target_title, data_dir="../data"):
    """
    Find a path between two Wikipedia pages using WikiRaceGNN's traversal algorithms.
    
    Args:
        source_title: Title of the source Wikipedia page
        target_title: Title of the target Wikipedia page
        data_dir: Directory containing the Wiki graph data
        
    Returns:
        List of page titles forming the path
    """
    # Convert data to WikiRaceGNN format
    data, _, _ = convert_wiki_to_wikiracegnn_format(data_dir)
    
    # Check if pages exist in the graph
    if source_title not in data.node_mapping:
        raise ValueError(f"Source page '{source_title}' not found in the graph")
    
    if target_title not in data.node_mapping:
        raise ValueError(f"Target page '{target_title}' not found in the graph")
    
    # Get node indices
    source_idx = data.node_mapping[source_title]
    target_idx = data.node_mapping[target_title]
    
    # Import bidirectional BFS from WikiRaceGNN
    try:
        from traversal.utils import bidirectional_bfs
    except ImportError:
        print("Could not import bidirectional_bfs from WikiRaceGNN. Make sure the traversal module is accessible.")
        return None
    
    # Find path
    print(f"Finding path from '{source_title}' to '{target_title}'...")
    path, nodes_explored = bidirectional_bfs(data, source_idx, target_idx)
    
    # Convert path indices to page titles
    if path:
        path_titles = [data.reverse_mapping[idx] for idx in path]
        
        print(f"Found path in {len(path)} steps, exploring {nodes_explored} nodes:")
        for i, title in enumerate(path_titles):
            print(f"  {i+1}. {title}")
        
        return path_titles
    else:
        print(f"No path found from '{source_title}' to '{target_title}'")
        return None

def main():
    """
    Main function to demonstrate the Wiki Graph Builder integration with WikiRaceGNN.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Wiki Graph Builder and Integration with WikiRaceGNN")
    
    parser.add_argument("--max_nodes", type=int, default=1000, help="Maximum number of nodes to include in the graph")
    parser.add_argument("--visualize", action="store_true", help="Visualize the graph")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory to save/load data")
    
    args = parser.parse_args()
    
    # Build the graph if requested
    create_wiki_edge_list(output_dir=args.data_dir, max_nodes=args.max_nodes, use_word2vec=True)
    
    # Visualize the graph if requested
    if args.visualize:
        visualize_wiki_graph(data_dir=args.data_dir)
    
if __name__ == "__main__":
    main()