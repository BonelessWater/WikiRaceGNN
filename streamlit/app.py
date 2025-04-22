import os
import torch
import numpy as np
import random
import json
import time
import networkx as nx
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from werkzeug.utils import secure_filename
from collections import deque

# Import project modules
from models import WikiGraphSAGE
from traversal import GraphTraverser
from utils import load_graph_data
from traversal.utils import bidirectional_bfs

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['SECRET_KEY'] = 'enhanced_wikipedia_graph_traversal'  # For session management
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables to store model and data
graph_data = None
model = None
traverser = None
graph_nx = None
graph_layout = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Default configuration
config = {
    'edge_file': 'data/wiki_edges.csv',
    'max_nodes': 1000,
    'feature_dim': 64,
    'model_path': 'models/enhanced_model_final.pt',
    'beam_width': 5,
    'heuristic_weight': 1.5,
    'max_steps': 30,
    'use_word2vec': True,
    'layout_iterations': 50,
    'visualize_exploration': True
}

def init_model():
    """Initialize the model, graph data, and compute graph layout"""
    global graph_data, model, traverser, graph_nx, graph_layout
    
    print("Initializing model and graph data...")
    
    try:
        # Load graph data
        graph_data = load_graph_data(
            config['edge_file'],
            feature_dim=config['feature_dim'],
            max_nodes=config['max_nodes'],
            ensure_connected=True,
            use_word2vec=config['use_word2vec']
        )
        
        print(f"Loaded graph with {graph_data.x.size(0)} nodes and {graph_data.edge_index.size(1) // 2} edges")
        
        # Create NetworkX graph for visualization
        print("Building NetworkX graph for visualization...")
        graph_nx = nx.Graph()
        num_nodes = graph_data.x.size(0)
        graph_nx.add_nodes_from(range(num_nodes))
        
        # Add edges
        edge_index_np = graph_data.edge_index.cpu().numpy().astype(int)
        for i in range(edge_index_np.shape[1]):
            source = int(edge_index_np[0, i])
            target = int(edge_index_np[1, i])
            graph_nx.add_edge(source, target)
        
        print(f"Built NetworkX graph with {graph_nx.number_of_nodes()} nodes and {graph_nx.number_of_edges()} edges")
        
        # Pre-compute graph layout for visualization
        print(f"Computing graph layout (iterations: {config['layout_iterations']})...")
        graph_layout = nx.spring_layout(
            graph_nx, 
            seed=42, 
            iterations=config['layout_iterations']
        )
        print("Layout computation complete")
        
        # Initialize model
        input_dim = graph_data.x.size(1)
        hidden_dim = 256
        output_dim = 64
        
        model = WikiGraphSAGE(input_dim, hidden_dim, output_dim, num_layers=4)
        
        # Load model weights if available
        if os.path.exists(config['model_path']):
            model_state = torch.load(config['model_path'], map_location=device)
            
            # Handle potential checkpoint format differences
            if isinstance(model_state, dict) and ('state_dict' in model_state or 'model_state_dict' in model_state):
                model_state = model_state.get('state_dict', model_state.get('model_state_dict'))
                model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
            
            model.load_state_dict(model_state)
            print(f"Loaded model from {config['model_path']}")
        else:
            print(f"Warning: Model not found at {config['model_path']}")
        
        model = model.to(device)
        model.eval()
        
        # Initialize traverser
        traverser = GraphTraverser(
            model,
            graph_data,
            device,
            beam_width=config['beam_width'],
            heuristic_weight=config['heuristic_weight']
        )
        
        print("Model and graph data initialized successfully")
        return True
    except Exception as e:
        import traceback
        print(f"Error initializing model: {e}")
        print(traceback.format_exc())
        return False

@app.route('/')
def index():
    """Render the main page"""
    global graph_data, graph_nx
    
    # Initialize model if not already done
    if graph_data is None:
        success = init_model()
        if not success:
            return render_template('error.html', error="Failed to initialize model and graph data.")
    
    # Get node information for dropdown selection
    nodes = []
    if hasattr(graph_data, 'node_titles') and graph_data.node_titles:
        # Use titles if available
        for idx, title in graph_data.node_titles.items():
            nodes.append({
                'id': graph_data.reverse_mapping[idx],
                'title': title
            })
    else:
        # Use IDs if titles not available
        for idx in range(min(100, graph_data.x.size(0))):  # Limit to 100 nodes for performance
            nodes.append({
                'id': graph_data.reverse_mapping[idx],
                'title': f"Node {graph_data.reverse_mapping[idx]}"
            })
    
    # Sort nodes by title for easier selection
    nodes.sort(key=lambda x: x['title'])
    
    # Get graph statistics
    graph_stats = {
        'num_nodes': graph_data.x.size(0),
        'num_edges': graph_data.edge_index.size(1) // 2,
        'avg_degree': graph_data.edge_index.size(1) / graph_data.x.size(0)
    }
    
    # Get additional stats from NetworkX if available
    if graph_nx is not None:
        try:
            # Calculate some basic graph properties
            graph_stats['density'] = nx.density(graph_nx)
            
            # Use largest connected component for further analysis
            largest_cc = max(nx.connected_components(graph_nx), key=len)
            largest_cc_graph = graph_nx.subgraph(largest_cc).copy()
            
            # Calculate diameter with a timeout to avoid long computation
            try:
                import signal
                
                def handler(signum, frame):
                    raise TimeoutError("Diameter calculation timed out")
                
                # Set timeout for 2 seconds
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(2)
                
                try:
                    diameter = nx.diameter(largest_cc_graph)
                    graph_stats['diameter'] = diameter
                except TimeoutError:
                    graph_stats['diameter'] = 'N/A (timeout)'
                except Exception as e:
                    graph_stats['diameter'] = 'N/A'
                
                # Reset the alarm
                signal.alarm(0)
            except:
                # Fallback if the signal module is not available
                graph_stats['diameter'] = 'N/A'
            
            # Calculate other metrics
            graph_stats['clustering'] = round(nx.average_clustering(graph_nx), 4)
            graph_stats['connected_components'] = nx.number_connected_components(graph_nx)
            graph_stats['largest_component_size'] = len(largest_cc)
            graph_stats['largest_component_percentage'] = round(len(largest_cc) / graph_nx.number_of_nodes() * 100, 2)
            
        except Exception as e:
            print(f"Error calculating graph stats: {e}")
    
    return render_template('index.html', nodes=nodes, config=config, graph_stats=graph_stats)

def reconstruct_bidir_bfs_path(start_node_idx, end_node_idx, meeting_node_idx, parent_f, parent_b):
    """Reconstruct a bidirectional BFS path for visualization"""
    path_f = deque([meeting_node_idx])
    curr = meeting_node_idx
    while curr != start_node_idx:
        if curr not in parent_f:
            return []
        curr = parent_f[curr]
        path_f.appendleft(curr)
    
    path_b_rev = []
    curr = meeting_node_idx
    while curr != end_node_idx:
        if curr not in parent_b:
            return []
        curr = parent_b[curr]
        path_b_rev.append(curr)
    
    return list(path_f) + path_b_rev

def run_bidirectional_bfs(data, src_idx, tgt_idx, max_steps=30, track_exploration=True):
    """
    Enhanced bidirectional BFS that also captures traversal history
    for visualization purposes
    """
    # Return already if source and target are the same
    if src_idx == tgt_idx:
        return [src_idx], 1, [(set([src_idx]), set([tgt_idx]), set(), 0)]

    # Initialize queues, visited sets, and traversal history
    q_f = deque([src_idx])
    q_b = deque([tgt_idx])
    visited_f = {src_idx: 0}
    visited_b = {tgt_idx: 0}
    parent_f = {src_idx: None}
    parent_b = {tgt_idx: None}
    meeting_node_idx = -1
    min_dist = float('inf')
    
    current_frontier_f = {src_idx}
    current_frontier_b = {tgt_idx}
    traversed_edges_cumulative = set()
    traversal_history = [(current_frontier_f.copy(), current_frontier_b.copy(), traversed_edges_cumulative.copy(), 0)]
    
    step = 0
    while q_f and q_b and step < max_steps:
        step += 1
        
        # Forward search step
        next_frontier_f = set()
        count = len(q_f)
        for _ in range(count):
            u = q_f.popleft()
            if visited_f[u] + 1 > min_dist:
                continue
            
            for v in data.adj_list.get(u, []):
                edge = tuple(sorted((u, v)))
                
                if v not in visited_f:
                    visited_f[v] = visited_f[u] + 1
                    parent_f[v] = u
                    next_frontier_f.add(v)
                    q_f.append(v)
                    traversed_edges_cumulative.add(edge)
                    
                    if v in visited_b:
                        dist = visited_f[v] + visited_b[v]
                        if dist < min_dist:
                            min_dist = dist
                            meeting_node_idx = v
                
                elif v in visited_b:
                    dist = visited_f[u] + 1 + visited_b[v]
                    if dist < min_dist:
                        min_dist = dist
                        meeting_node_idx = v
                    traversed_edges_cumulative.add(edge)
        
        current_frontier_f = next_frontier_f
        
        # Backward search step
        next_frontier_b = set()
        count = len(q_b)
        for _ in range(count):
            u = q_b.popleft()
            if visited_b[u] + 1 > min_dist:
                continue
            
            for v in data.adj_list.get(u, []):
                edge = tuple(sorted((u, v)))
                
                if v not in visited_b:
                    visited_b[v] = visited_b[u] + 1
                    parent_b[v] = u
                    next_frontier_b.add(v)
                    q_b.append(v)
                    traversed_edges_cumulative.add(edge)
                    
                    if v in visited_f:
                        dist = visited_f[v] + visited_b[v]
                        if dist < min_dist:
                            min_dist = dist
                            meeting_node_idx = v
                
                elif v in visited_f:
                    dist = visited_f[v] + visited_b[u] + 1
                    if dist < min_dist:
                        min_dist = dist
                        meeting_node_idx = v
                    traversed_edges_cumulative.add(edge)
        
        current_frontier_b = next_frontier_b
        
        # Record history if tracking is enabled
        if track_exploration:
            traversal_history.append((
                current_frontier_f.copy(), 
                current_frontier_b.copy(), 
                traversed_edges_cumulative.copy(),
                step
            ))
    
    # Reconstruct path if meeting point was found
    nodes_explored = len(visited_f) + len(visited_b)
    if meeting_node_idx != -1:
        path = reconstruct_bidir_bfs_path(src_idx, tgt_idx, meeting_node_idx, parent_f, parent_b)
        return path, nodes_explored, traversal_history
    
    # No path found
    return [], nodes_explored, traversal_history

@app.route('/traverse', methods=['POST'])
def traverse():
    """Perform traversal between two nodes and return results with visualization data"""
    global traverser, graph_data
    
    # Check if model is initialized
    if traverser is None or graph_data is None:
        return jsonify({'error': 'Model not initialized'}), 500
    
    # Get request parameters
    data = request.json
    source_id = data.get('source_id')
    target_id = data.get('target_id')
    method = data.get('method', 'auto')
    max_steps = int(data.get('max_steps', config['max_steps']))
    
    # Validate IDs
    if source_id not in graph_data.node_mapping or target_id not in graph_data.node_mapping:
        return jsonify({'error': 'Invalid node IDs'}), 400
    
    # Get indices
    source_idx = graph_data.node_mapping[source_id]
    target_idx = graph_data.node_mapping[target_id]
    
    # Get BFS baseline for comparison with exploration history
    start_time_bfs = time.time()
    bfs_path, bfs_nodes_explored, bfs_history = run_bidirectional_bfs(
        graph_data, source_idx, target_idx, max_steps=max_steps, 
        track_exploration=config['visualize_exploration']
    )
    end_time_bfs = time.time()
    bfs_time = end_time_bfs - start_time_bfs
    
    # Convert BFS path to original IDs
    bfs_path_ids = [graph_data.reverse_mapping[idx] for idx in bfs_path] if bfs_path else []
    
    # Run GNN traversal
    try:
        # Enable traversal history collection if configured
        if hasattr(traverser, 'enable_history_collection'):
            traverser.enable_history_collection(config['visualize_exploration'])
        
        # Start timing
        start_time_gnn = time.time()
        
        # Perform traversal
        path, nodes_explored = traverser.traverse(
            source_id, target_id, max_steps=max_steps, method=method
        )
        
        # End timing
        end_time_gnn = time.time()
        gnn_time = end_time_gnn - start_time_gnn
        
        # Get traversal history for visualization if available
        gnn_history = []
        if hasattr(traverser, 'get_traversal_history'):
            try:
                raw_history = traverser.get_traversal_history()
                
                # Format history to match the BFS history format
                for item in raw_history:
                    if len(item) == 3:  # Should have fwd, bwd, step
                        fwd, bwd, step_num = item
                        gnn_history.append((
                            set(map(int, fwd)) if fwd else set(),
                            set(map(int, bwd)) if bwd else set(),
                            set(),  # No edge info available
                            step_num
                        ))
            except Exception as e:
                print(f"Error getting GNN traversal history: {e}")
        
        # Prepare node information for visualization
        path_info = []
        for node_id in path:
            idx = graph_data.node_mapping[node_id]
            title = graph_data.node_titles[idx] if hasattr(graph_data, 'node_titles') and idx in graph_data.node_titles else f"Node {node_id}"
            url = graph_data.node_urls[idx] if hasattr(graph_data, 'node_urls') and idx in graph_data.node_urls else None
            
            path_info.append({
                'id': node_id,
                'title': title,
                'url': url
            })
        
        # Prepare BFS path info
        bfs_path_info = []
        for node_id in bfs_path_ids:
            idx = graph_data.node_mapping[node_id]
            title = graph_data.node_titles[idx] if hasattr(graph_data, 'node_titles') and idx in graph_data.node_titles else f"Node {node_id}"
            url = graph_data.node_urls[idx] if hasattr(graph_data, 'node_urls') and idx in graph_data.node_urls else None
            
            bfs_path_info.append({
                'id': node_id,
                'title': title,
                'url': url
            })
        
        # Calculate statistics
        success = len(path) > 0 and path[-1] == target_id
        bfs_success = len(bfs_path_ids) > 0 and bfs_path_ids[-1] == target_id
        
        # Efficiency ratio (BFS nodes / GNN nodes)
        if nodes_explored > 0 and bfs_nodes_explored > 0:
            efficiency_ratio = bfs_nodes_explored / nodes_explored
        else:
            efficiency_ratio = 0
        
        # Prepare layout data for visualization if needed
        layout_data = {}
        if graph_layout is not None:
            for node_idx, pos in graph_layout.items():
                node_id = graph_data.reverse_mapping[node_idx]
                layout_data[node_id] = {
                    'x': float(pos[0]),
                    'y': float(pos[1])
                }
        
        # Return comprehensive results
        result = {
            'success': success,
            'path': path_info,
            'nodes_explored': nodes_explored,
            'bfs_path': bfs_path_info,
            'bfs_nodes_explored': bfs_nodes_explored,
            'efficiency_ratio': efficiency_ratio,
            'path_length': len(path),
            'bfs_path_length': len(bfs_path_ids),
            'method': method,
            'time_taken': gnn_time,
            'bfs_time_taken': bfs_time,
            'layout': layout_data
        }
        
        # Add visualization data if requested
        if config['visualize_exploration']:
            result['bfs_exploration_history'] = [
                {
                    'forward_frontier': list(fwd),
                    'backward_frontier': list(bwd),
                    'traversed_edges': [list(edge) for edge in edges],
                    'step': step
                } for fwd, bwd, edges, step in bfs_history
            ]
            
            result['gnn_exploration_history'] = [
                {
                    'forward_frontier': list(fwd),
                    'backward_frontier': list(bwd),
                    'traversed_edges': [],  # No edge info
                    'step': step
                } for fwd, bwd, _, step in gnn_history
            ]
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        print(f"Error during traversal: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/config', methods=['POST'])
def update_config():
    """Update configuration"""
    global config
    
    data = request.form
    
    # Update configuration
    config['max_nodes'] = int(data.get('max_nodes', config['max_nodes']))
    config['beam_width'] = int(data.get('beam_width', config['beam_width']))
    config['heuristic_weight'] = float(data.get('heuristic_weight', config['heuristic_weight']))
    config['max_steps'] = int(data.get('max_steps', config['max_steps']))
    
    # Handle edge file upload
    if 'edge_file' in request.files:
        file = request.files['edge_file']
        if file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            config['edge_file'] = filepath
    
    # Handle model file upload
    if 'model_path' in request.files:
        file = request.files['model_path']
        if file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            config['model_path'] = filepath
    
    # Reinitialize model with new configuration
    success = init_model()
    
    if success:
        return redirect(url_for('index'))
    else:
        return render_template('error.html', error="Failed to initialize model with new configuration.")

(50):  # Try several times
        source_idx = random.choice(all_nodes)
        target_idx = random.choice(all_nodes)
        
        if source_idx != target_idx:
            # Check if there's a path
            path, _ = bidirectional_bfs(graph_data, source_idx, target_idx)
            if path and 3 <= len(path) <= 8:  # Ensure reasonable path length
                source_id = graph_data.reverse_mapping[source_idx]
                target_id = graph_data.reverse_mapping[target_idx]
                
                # Get titles if available
                source_title = graph_data.node_titles[source_idx] if hasattr(graph_data, 'node_titles') and source_idx in graph_data.node_titles else f"Node {source_id}"
                target_title = graph_data.node_titles[target_idx] if hasattr(graph_data, 'node_titles') and target_idx in graph_data.node_titles else f"Node {target_id}"
                
                return jsonify({
                    'source_id': source_id,
                    'source_title': source_title,
                    'target_id': target_id,
                    'target_title': target_title,
                    'expected_path_length': len(path)
                })
    
    return jsonify({'error': 'Could not find suitable node pair'}), 404

@app.route('/node_info')
def node_info():
    """Get information about nodes"""
    global graph_data
    
    if graph_data is None:
        return jsonify({'error': 'Model not initialized'}), 500
    
    # Get node ID from query parameter
    node_id = request.args.get('id')
    if node_id is None:
        return jsonify({'error': 'No node ID provided'}), 400
    
    # Validate node ID
    if node_id not in graph_data.node_mapping:
        return jsonify({'error': 'Invalid node ID'}), 400
    
    # Get node index
    node_idx = graph_data.node_mapping[node_id]
    
    # Get node information
    title = graph_data.node_titles[node_idx] if hasattr(graph_data, 'node_titles') and node_idx in graph_data.node_titles else f"Node {node_id}"
    url = graph_data.node_urls[node_idx] if hasattr(graph_data, 'node_urls') and node_idx in graph_data.node_urls else None
    
    # Get neighbors
    neighbors = []
    for neighbor_idx in graph_data.adj_list.get(node_idx, []):
        neighbor_id = graph_data.reverse_mapping[neighbor_idx]
        neighbor_title = graph_data.node_titles[neighbor_idx] if hasattr(graph_data, 'node_titles') and neighbor_idx in graph_data.node_titles else f"Node {neighbor_id}"
        
        neighbors.append({
            'id': neighbor_id,
            'title': neighbor_title
        })
    
    return jsonify({
        'id': node_id,
        'title': title,
        'url': url,
        'neighbors': neighbors[:10]  # Limit to 10 neighbors
    })

if __name__ == '__main__':
    # Initialize model
    init_model()
    
    # Run the app
    app.run(debug=True, port=5000)