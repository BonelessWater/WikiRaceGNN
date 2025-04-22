import os
import torch
import pandas as pd
import json
import random
from torch_geometric.data import Data
from utils.wikibuilder import create_wiki_edge_list

def convert_wiki_to_graph_format(data_dir="data"):
    """Convert Wiki data to PyG format or load cached version"""
    # Check for cached data
    pt_path = os.path.join(data_dir, "wiki_graph.pt")
    if os.path.exists(pt_path):
        print(f"Loading existing PyTorch Geometric data from {pt_path}")
        data = torch.load(pt_path)
        return data, os.path.join(data_dir, "wiki_edges.csv"), os.path.join(data_dir, "wiki_id_mapping.csv")
    
    # Check for required files
    edge_list_path = os.path.join(data_dir, "wiki_edges.csv")
    node_meta_path = os.path.join(data_dir, "wiki_nodes.json")
    
    if not os.path.exists(edge_list_path) or not os.path.exists(node_meta_path):
        raise FileNotFoundError(f"Required files not found in {data_dir}")
    
    # Load data
    print(f"Loading edge list from {edge_list_path}")
    edges_df = pd.read_csv(edge_list_path)
    
    print(f"Loading node metadata from {node_meta_path}")
    with open(node_meta_path, 'r', encoding='utf-8') as f:
        nodes = json.load(f)
    
    # Create mappings
    print("Creating node mappings")
    node_titles = list(nodes.keys())
    node_mapping = {title: i for i, title in enumerate(node_titles)}
    reverse_mapping = {i: title for i, title in enumerate(node_titles)}
    
    # Save ID mapping
    id_mapping_path = os.path.join(data_dir, "wiki_id_mapping.csv")
    pd.DataFrame({
        'id': range(len(node_titles)),
        'title': node_titles,
        'url': [nodes[title]['url'] for title in node_titles]
    }).to_csv(id_mapping_path, index=False)
    
    # Create edge index
    print("Creating edge index")
    edge_index = []
    for _, row in edges_df.iterrows():
        source, target = row['id1'], row['id2']
        if source in node_mapping and target in node_mapping:
            edge_index.append([node_mapping[source], node_mapping[target]])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    
    # Create node features
    print("Creating node feature matrix")
    x = torch.tensor([nodes[title]['embedding'] for title in node_titles], dtype=torch.float)
    
    # Create PyG data object with adjacency list
    print("Creating PyTorch Geometric Data object")
    data = Data(x=x, edge_index=edge_index)
    data.node_mapping = node_mapping
    data.reverse_mapping = reverse_mapping
    
    # Create adjacency list for faster access
    print("Creating adjacency list")
    adj_list = {}
    for i in range(edge_index.size(1)):
        source = edge_index[0, i].item()
        target = edge_index[1, i].item()
        if source not in adj_list:
            adj_list[source] = []
        adj_list[source].append(target)
    
    data.adj_list = adj_list
    
    # Cache the data
    print(f"Saving PyTorch Geometric data to {pt_path}")
    torch.save(data, pt_path)
    
    print(f"Conversion complete! Created graph with {data.num_nodes} nodes and {data.num_edges} edges")
    return data, edge_list_path, id_mapping_path

def select_traversal_pair(data_dir="data", min_path_length=3, max_path_length=10):
    """Select a random node pair with a path in the specified length range"""
    # Convert data to graph format
    data, _, _ = convert_wiki_to_graph_format(data_dir)
    
    # Import bidirectional BFS
    try:
        from traversal.utils import bidirectional_bfs
    except ImportError:
        print("Could not import bidirectional_bfs from traversal module.")
        return None
    
    # Sample node pairs until a valid pair is found
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

def find_path(source_title, target_title, data_dir="data"):
    """Find a path between two Wikipedia pages"""
    # Convert data to graph format
    data, _, _ = convert_wiki_to_graph_format(data_dir)
    
    # Check if pages exist in the graph
    if source_title not in data.node_mapping:
        print(f"Source page '{source_title}' not found in the graph")
        return None
    
    if target_title not in data.node_mapping:
        print(f"Target page '{target_title}' not found in the graph")
        return None
    
    # Get node indices
    source_idx = data.node_mapping[source_title]
    target_idx = data.node_mapping[target_title]
    
    # Import bidirectional BFS
    try:
        from traversal.utils import bidirectional_bfs
    except ImportError:
        print("Could not import bidirectional_bfs. Make sure the traversal module is accessible.")
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

def run_with_wiki_data(data_dir="data", mode="traverse", visualize=False):
    """Run code using the Wiki graph data"""
    # Convert data to graph format
    data, edge_list_path, _ = convert_wiki_to_graph_format(data_dir)
    
    # Import main function
    try:
        from main import main
        
        # Set up command line arguments
        import sys
        sys.argv = ['main.py']
        sys.argv.extend(['--edge_file', edge_list_path])
        sys.argv.extend(['--mode', mode])
        
        if visualize:
            sys.argv.append('--visualize')
        
        # Run main
        print(f"Running in {mode} mode with Wiki graph data")
        main()
        
    except ImportError:
        print("Could not import main. Make sure the main.py file is accessible.")
    except Exception as e:
        print(f"Error running with Wiki graph data: {e}")

def prepare_graph_data(max_nodes=1000, data_dir="data", use_word2vec=True):
    """Create graph data if needed or use existing data"""
    os.makedirs(data_dir, exist_ok=True)
    
    edge_file = os.path.join(data_dir, "wiki_edges.csv")
    if not os.path.exists(edge_file):
        create_wiki_edge_list(output_dir=data_dir, max_nodes=max_nodes, use_word2vec=use_word2vec)
    
    return convert_wiki_to_graph_format(data_dir)

def main():
    """
    Main function to demonstrate Wiki Graph Builder integration.
    Builds the graph, finds random traversal pairs, and runs traversal.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Wiki Graph Builder and Integration")
    
    parser.add_argument("--max_nodes", type=int, default=1000, 
                        help="Maximum number of nodes to include in the graph")
    parser.add_argument("--visualize", action="store_true", 
                        help="Enable visualization (if supported)")
    parser.add_argument("--data_dir", type=str, default="data", 
                        help="Directory to save/load data")
    parser.add_argument("--mode", type=str, choices=['data', 'traverse', 'train', 'evaluate'],
                        default='data', help="Operation mode")
    parser.add_argument("--find_path", action="store_true",
                        help="Find a random path after building the graph")
    
    args = parser.parse_args()
    
    # Create data directory
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Build the graph if requested or run with existing data
    if args.mode == 'data':
        # Build graph data
        print(f"Building graph with max {args.max_nodes} nodes")
        prepare_graph_data(args.max_nodes, args.data_dir)
        
        # Find a random traversal pair if requested
        if args.find_path:
            pair = select_traversal_pair(args.data_dir)
            if pair:
                source_id, target_id = pair
                find_path(source_id, target_id, args.data_dir)
    else:
        # Run with existing data
        run_with_wiki_data(args.data_dir, args.mode, args.visualize)
    
if __name__ == "__main__":
    main()