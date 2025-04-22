import os
import torch
import numpy as np
import random
import json
import time
import sys
import networkx as nx
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from werkzeug.utils import secure_filename
from collections import deque
import traceback # Import traceback for better error logging
import signal # For timeouts (optional but good for long calculations)

# --- Import project modules ---
# Ensure these paths are correct relative to where you run app.py
try:
    from models.graphsage import WikiGraphSAGE
    from traversal.unified import GraphTraverser
    from utils.data import load_graph_data_no_pandas as load_graph_data
    # Local BFS implementation handles history needed for visualization
except ImportError as e:
    print(f"--- ImportError ---")
    print(f"Failed to import project modules: {e}")
    print("Current working directory:", os.getcwd())
    print("Python path:", sys.path)
    print("Please ensure 'models', 'traversal', and 'utils' directories are accessible.")
    print("-------------------")
    sys.exit(1)

# --- Flask App Setup ---
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'default_secret_key_change_me_!@#') # Use env var or default
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Global Variables ---
graph_data = None       # PyG Data object
model = None            # PyTorch model
traverser = None        # GraphTraverser instance
graph_nx = None         # NetworkX graph (using integer indices for internal calcs)
graph_layout = None     # NetworkX layout {index: [x, y]}
device = None           # PyTorch device (CPU or CUDA)

# --- Default Configuration ---
config = {
    'edge_file': os.path.join('data', 'wiki_edges.csv'),
    'max_nodes': 1000,
    'feature_dim': 64,
    'model_path': os.path.join('models', 'enhanced_model_final.pt'),
    'beam_width': 5,
    'heuristic_weight': 1.5,
    'max_steps': 30,
    'use_word2vec': True, # Flag for data loading
    'layout_iterations': 50,
    'visualize_exploration': True # Default UI state / Request flag
}

# --- Helper Functions ---
def get_data_attribute(data_obj, attr_name, default=None):
    """Safely get an attribute from a PyG Data object or its internal store."""
    default_val = default if default is not None else {} # Default for mappings/lists is {}
    if data_obj is None: return default_val
    val = getattr(data_obj, attr_name, None)
    if val is not None:
        if isinstance(default_val, (dict, list)) and val is None and default is None: return default_val
        return val
    if hasattr(data_obj, '_store') and data_obj._store is not None and attr_name in data_obj._store:
         val = data_obj._store[attr_name]
         if isinstance(default_val, (dict, list)) and val is None and default is None: return default_val
         return val
    return default_val

def set_device():
    """Sets the global PyTorch device."""
    global device
    # (Keep the set_device function from the previous correct version)
    try:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"CUDA device detected: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
             try:
                 torch.ones(1, device='mps'); device = torch.device('mps')
                 print("MPS device detected and available (Apple Silicon GPU).")
             except Exception as mps_err: print(f"MPS detected but unavailable ({mps_err}), falling back to CPU."); device = torch.device('cpu')
        else: device = torch.device('cpu'); print("CUDA/MPS not available, using CPU.")
    except Exception as e: print(f"Error detecting device, defaulting to CPU: {e}"); device = torch.device('cpu')


# --- Initialization ---
def init_model():
    """Initialize the model, graph data, NetworkX graph, and layout."""
    global graph_data, model, traverser, graph_nx, graph_layout, device, config
    set_device(); print(f"--- Initializing Application ---\nUsing device: {device}\nUsing configuration: {config}")
    try:
        # --- Load Graph Data ---
        print(f"Loading graph from: {config['edge_file']}")
        if not os.path.exists(config['edge_file']): raise FileNotFoundError(f"Edge file not found: {config['edge_file']}")
        graph_data = load_graph_data(config['edge_file'], feature_dim=config['feature_dim'], max_nodes=config['max_nodes'], ensure_connected=True, use_word2vec=config['use_word2vec'])
        if graph_data is None: raise ValueError("load_graph_data returned None.")
        if not hasattr(graph_data, 'x') or graph_data.x is None: raise ValueError("Graph data lacks node features ('x').")
        node_mapping = get_data_attribute(graph_data, 'node_mapping', {}); reverse_mapping = get_data_attribute(graph_data, 'reverse_mapping', {}); adj_list = get_data_attribute(graph_data, 'adj_list', {})
        if not node_mapping or not reverse_mapping or not adj_list: missing = [k for k, v in {'node_mapping': node_mapping, 'reverse_mapping': reverse_mapping, 'adj_list': adj_list}.items() if not v]; raise ValueError(f"Essential data missing: {', '.join(missing)}. Check load_graph_data.")
        num_nodes_loaded = graph_data.x.size(0); num_edges_loaded = get_data_attribute(graph_data, 'edge_index', torch.empty(2,0)).size(1) // 2
        print(f"Loaded PyG graph with {num_nodes_loaded} nodes and {num_edges_loaded} edges. Mappings: {len(node_mapping)}, AdjList: {len(adj_list)}")

        # --- Create NetworkX Graph (using integer indices 0 to N-1) ---
        print("Building NetworkX graph (using indices)...")
        graph_nx = nx.Graph(); graph_nx.add_nodes_from(range(num_nodes_loaded))
        edge_index = get_data_attribute(graph_data, 'edge_index')
        if edge_index is not None and edge_index.numel() > 0:
            edge_index_np = edge_index.cpu().numpy()
            edges_to_add = [(int(u), int(v)) for u, v in zip(edge_index_np[0], edge_index_np[1]) if u < v and 0 <= u < num_nodes_loaded and 0 <= v < num_nodes_loaded]
            graph_nx.add_edges_from(edges_to_add)
            print(f"Built NetworkX graph: {graph_nx.number_of_nodes()} nodes, {graph_nx.number_of_edges()} edges.")
        else: print("Warning: No 'edge_index'. NetworkX graph has no edges.")

        # --- Compute Graph Layout (using integer indices) ---
        graph_layout = None
        if graph_nx.number_of_nodes() > 0:
            print(f"Computing graph layout (iterations: {config['layout_iterations']})..."); start_layout_time = time.time()
            try: graph_layout = nx.spring_layout(graph_nx, seed=42, iterations=config['layout_iterations']); print(f"Layout done in {time.time() - start_layout_time:.2f}s.")
            except Exception as e_layout: print(f"Error computing graph layout: {e_layout}. Visualization might fail."); graph_layout = None
        else: print("Skipping layout (no nodes).")

        # --- Initialize GNN Model ---
        input_dim = get_data_attribute(graph_data, 'x').size(1)
        hidden_dim = 256; output_dim = 64; num_layers = 4 # *** MATCH SAVED MODEL ***
        print(f"Initializing WikiGraphSAGE: In={input_dim}, Hidden={hidden_dim}, Out={output_dim}, Layers={num_layers}")
        model = WikiGraphSAGE(input_dim, hidden_dim, output_dim, num_layers=num_layers)
        model_path = config['model_path']
        if os.path.exists(model_path):
            print(f"Loading model weights from {model_path}...");
            try:
                model_state = torch.load(model_path, map_location=device)
                if isinstance(model_state, dict): state_dict_key = 'state_dict' if 'state_dict' in model_state else 'model_state_dict'; model_state = model_state.get(state_dict_key, model_state)
                model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
                missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False) # *** USE strict=False ***
                if unexpected_keys: print(f"Warning: Unexpected keys found while loading model: {unexpected_keys}")
                if missing_keys: raise RuntimeError(f"Missing essential keys in model state_dict: {missing_keys}")
                print(f"Successfully loaded model weights (strict=False).")
            except Exception as e: print(f"Error loading model weights: {e}\nProceeding with random weights.")
        else: print(f"Warning: Model weights file not found: {model_path}. Using random weights.")
        model = model.to(device); model.eval()

        # --- Initialize Traverser ---
        print("Initializing GraphTraverser...")
        traverser = GraphTraverser(model=model, data=graph_data, device=device, beam_width=config['beam_width'], heuristic_weight=config['heuristic_weight'])
        print("--- Application Initialization Complete ---")
        return True
    except FileNotFoundError as e: print(f"----- INIT ERROR: File Not Found: {e} -----"); graph_data = model = traverser = graph_nx = graph_layout = None; return False
    except Exception as e: print(f"----- INIT ERROR: {e} -----\n{traceback.format_exc()}\n-------------------------"); graph_data = model = traverser = graph_nx = graph_layout = None; return False

# --- Flask Routes ---
@app.route('/')
def index():
    """Render the main page, initializing model if necessary."""
    global graph_data, config
    if graph_data is None:
        print("State Check: graph_data is None in '/', attempting init_model()...")
        if not init_model(): return render_template('error.html', error="Initialization Error. Check server logs.")
        if graph_data is None: return render_template('error.html', error="Initialization failed. Graph data still None. Check server logs.")
    nodes = []; node_titles = get_data_attribute(graph_data, 'node_titles', {}); reverse_mapping = get_data_attribute(graph_data, 'reverse_mapping', {})
    if not reverse_mapping: print("Error in /: Reverse mapping missing.")
    else:
        max_dropdown_nodes = 5000; node_count = 0
        for idx, original_id in reverse_mapping.items():
            if node_count >= max_dropdown_nodes:
                if node_count == max_dropdown_nodes: print(f"Warning: Limiting node dropdown to {max_dropdown_nodes}.")
                node_count += 1; continue
            nodes.append({'id': str(original_id), 'title': node_titles.get(idx, f"Node {original_id}")})
            node_count += 1
        try: nodes.sort(key=lambda x: x['title'])
        except Exception as e: print(f"Warning: Could not sort nodes: {e}")
    graph_stats = {};
    try: # Calculate stats safely
        num_nodes = get_data_attribute(graph_data, 'num_nodes', 0); num_edges = get_data_attribute(graph_data, 'num_edges', 0)
        graph_stats['num_nodes'] = num_nodes; graph_stats['num_edges'] = num_edges
        graph_stats['avg_degree'] = (2 * num_edges) / num_nodes if num_nodes > 0 else 0
        if graph_nx is not None and graph_nx.number_of_nodes() > 0:
            graph_stats['density'] = nx.density(graph_nx)
            if num_nodes < 10000: # Limit complex calcs
                 try: is_connected = nx.is_connected(graph_nx)
                 except Exception: is_connected = False
                 if is_connected: graph_stats['connected_components'] = 1; graph_stats['largest_component_size'] = num_nodes; graph_stats['largest_component_percentage'] = 100.0
                 else: ccs = list(nx.connected_components(graph_nx)); graph_stats['connected_components'] = len(ccs); largest_cc_nodes = max(ccs, key=len) if ccs else []; graph_stats['largest_component_size'] = len(largest_cc_nodes); graph_stats['largest_component_percentage'] = round(len(largest_cc_nodes)/num_nodes*100,1) if num_nodes > 0 else 0.0
                 if num_nodes < 5000: graph_stats['clustering'] = round(nx.average_clustering(graph_nx), 4)
            else: print("Skipping some metrics due to graph size.")
    except Exception as e: print(f"Error calculating graph stats: {e}")
    return render_template('index.html', nodes=nodes, config=config, graph_stats=graph_stats)


# --- Enhanced Bidirectional BFS (Local Version) ---
# (Keep the run_bidirectional_bfs and reconstruct_bidir_bfs_path functions from the previous corrected version)
def reconstruct_bidir_bfs_path(start_node_idx, end_node_idx, meeting_node_idx, parent_f, parent_b):
    path_f = deque([meeting_node_idx]); curr = meeting_node_idx; count = 0; limit = config.get('max_nodes', 1000) * 2
    while curr != start_node_idx and count < limit:
        if curr not in parent_f or parent_f[curr] is None: return None
        curr = parent_f[curr]; path_f.appendleft(curr); count += 1
    if curr != start_node_idx: return None
    path_b_rev = []; curr = meeting_node_idx; count = 0
    while curr != end_node_idx and count < limit:
        if curr not in parent_b or parent_b[curr] is None:
            if curr == end_node_idx: break
            return None
        curr = parent_b[curr]; path_b_rev.append(curr); count += 1
    if curr != end_node_idx and meeting_node_idx != end_node_idx: return None
    return list(path_f) + path_b_rev

def run_bidirectional_bfs(data_obj, src_idx, tgt_idx, max_steps=30, track_exploration=True):
    """Enhanced bidirectional BFS capturing traversal history."""
    adj_list = get_data_attribute(data_obj, 'adj_list', {})
    if not adj_list: return [], 0, []
    if src_idx == tgt_idx: return [src_idx], 1, [({src_idx}, {tgt_idx}, set(), 0)] if track_exploration else []

    q_f, q_b = deque([src_idx]), deque([tgt_idx])
    visited_f, visited_b = {src_idx: 0}, {tgt_idx: 0} # Store node_idx: distance
    parent_f, parent_b = {src_idx: None}, {tgt_idx: None}
    meeting_node_idx, min_dist = -1, float('inf')
    curr_f, curr_b = {src_idx}, {tgt_idx}
    edges_cum = set()
    history = []
    if track_exploration: history.append((curr_f.copy(), curr_b.copy(), set(), 0)) # Initial state

    step, nodes_explored, max_total_steps = 0, 2, max_steps * 2

    while q_f and q_b and step < max_total_steps:
        step += 1
        is_fwd = len(q_f) <= len(q_b) # Expand smaller queue

        # Select queue, visited sets, parent dict based on direction
        if is_fwd:
            q, visited, other_visited, parent = q_f, visited_f, visited_b, parent_f
        else:
            q, visited, other_visited, parent = q_b, visited_b, visited_f, parent_b

        count = len(q)
        next_frontier = set()

        for _ in range(count):
            u = q.popleft()
            current_dist = visited[u]

            # Pruning
            if current_dist >= min_dist: continue # Path already longer than best found
            if current_dist >= max_steps: continue # Exceeded max depth for this direction

            # Process neighbors
            for v in adj_list.get(u, []):
                edge = tuple(sorted((u, v)))

                # --- Check Intersection FIRST if neighbor is already visited by the other search ---
                if v in other_visited:
                    # Calculate potential distance through this neighbor
                    # dist = current_dist (dist to u) + 1 (edge u->v) + other_visited[v] (dist from other end to v)
                    dist_through_v = current_dist + 1 + other_visited[v]
                    if dist_through_v < min_dist:
                        min_dist = dist_through_v
                        # Meeting node is tricky: it's the one *closer* to its respective start
                        # For simplicity, let's assign 'v' here, path reconstruction handles it.
                        meeting_node_idx = v
                        # print(f"  Intersection Check 1 at {v}, new min_dist = {min_dist}")

                # --- Process neighbor if not visited by *this* search direction ---
                if v not in visited:
                    new_dist_v = current_dist + 1
                    # Check pruning again before adding
                    if new_dist_v >= min_dist: continue

                    # Add to visited, parent, queue, frontier
                    if v not in visited_f and v not in visited_b: nodes_explored += 1 # Count truly new node
                    visited[v] = new_dist_v
                    parent[v] = u
                    next_frontier.add(v)
                    q.append(v)
                    if track_exploration: edges_cum.add(edge)

                    # Re-check intersection after adding (neighbor might now be in other_visited)
                    if v in other_visited:
                        dist_now = visited[v] + other_visited[v] # Calculate dist using newly added v
                        if dist_now < min_dist:
                            min_dist = dist_now
                            meeting_node_idx = v
                            # print(f"  Intersection Check 2 at {v}, new min_dist = {min_dist}")

                # --- Add edge for history even if v was already visited by this direction ---
                elif track_exploration:
                    edges_cum.add(edge)

        # Update the correct overall frontier set after processing all nodes at this level
        if is_fwd: curr_f = next_frontier
        else: curr_b = next_frontier

        # Record history after each *full* step (or adjust if needed)
        if track_exploration and step % 2 == 0: # Record every two half-steps
             history.append((curr_f.copy(), curr_b.copy(), edges_cum.copy(), step // 2))

    # Path Reconstruction
    path = []
    if meeting_node_idx != -1:
        path = reconstruct_bidir_bfs_path(src_idx, tgt_idx, meeting_node_idx, parent_f, parent_b) or []

    # Add final state to history if tracking
    if track_exploration:
        last_step = history[-1][-1] if history else -1
        final_step_num = (step + 1) // 2 # Ensure final step number is correct
        if last_step != final_step_num: # Add final frame if loop ended or step is odd
            history.append((curr_f.copy(), curr_b.copy(), edges_cum.copy(), final_step_num))

    return path, nodes_explored, history

# --- Main Traversal Route ---
@app.route('/traverse', methods=['POST'])
def traverse():
    """Perform traversal, return results including layout, edges, and history."""
    global traverser, graph_data, graph_nx, graph_layout, config
    start_req_time = time.time()

    # --- Pre-checks ---
    if traverser is None or graph_data is None or graph_layout is None: return jsonify({'error': 'Application components not ready.'}), 503
    node_mapping = get_data_attribute(graph_data, 'node_mapping', {}); reverse_mapping = get_data_attribute(graph_data, 'reverse_mapping', {})
    node_titles = get_data_attribute(graph_data, 'node_titles', {}); node_urls = get_data_attribute(graph_data, 'node_urls', {})
    if not node_mapping or not reverse_mapping: return jsonify({'error': 'Essential node mappings missing.'}), 500

    # --- Get Request Data & Validate ---
    try:
        data = request.json; source_id_str = data.get('source_id'); target_id_str = data.get('target_id')
        method = data.get('method', 'auto'); max_steps = int(data.get('max_steps', config['max_steps']))
        track_exploration = data.get('visualize_exploration', config['visualize_exploration'])
        if not source_id_str or not target_id_str: return jsonify({'error': 'Source/Target ID missing.'}), 400
        if source_id_str not in node_mapping: return jsonify({'error': f'Source ID "{source_id_str}" not found.'}), 404
        if target_id_str not in node_mapping: return jsonify({'error': f'Target ID "{target_id_str}" not found.'}), 404
        source_idx, target_idx = node_mapping[source_id_str], node_mapping[target_id_str]
    except Exception as e: return jsonify({'error': f'Invalid request format: {e}'}), 400

    # --- Run BFS Baseline ---
    print(f"Running BFS: {source_id_str} -> {target_id_str} (Track={track_exploration})")
    start_time_bfs = time.time()
    bfs_path_indices, bfs_nodes_explored, bfs_history_raw = run_bidirectional_bfs(graph_data, source_idx, target_idx, max_steps=max_steps, track_exploration=track_exploration)
    bfs_time = time.time() - start_time_bfs; print(f"BFS done ({bfs_time:.3f}s). Path len: {len(bfs_path_indices)}, Explored: {bfs_nodes_explored}")

    # --- Run GNN Traversal ---
    print(f"Running GNN ({method}): {source_id_str} -> {target_id_str} (Track={track_exploration})")
    start_time_gnn = time.time(); gnn_path_ids, gnn_nodes_explored, method_used, gnn_history_raw = [], 0, method, []
    try:
        if hasattr(traverser, 'enable_history_collection'): traverser.enable_history_collection(track_exploration)
        # *** Assume traverser.traverse RETURNS 3 values ***
        gnn_result = traverser.traverse(source_id_str, target_id_str, max_steps=max_steps, method=method)
        # *** Robust Unpacking ***
        if isinstance(gnn_result, tuple) and len(gnn_result) == 3: gnn_path_ids, gnn_nodes_explored, method_used = gnn_result; gnn_path_ids = gnn_path_ids if isinstance(gnn_path_ids, list) else []
        elif isinstance(gnn_result, tuple) and len(gnn_result) == 2: gnn_path_ids, gnn_nodes_explored = gnn_result; gnn_path_ids = gnn_path_ids if isinstance(gnn_path_ids, list) else []; method_used = method; print("Warning: traverser.traverse returned only 2 values.")
        elif isinstance(gnn_result, list): gnn_path_ids = gnn_result; gnn_nodes_explored = getattr(traverser, 'nodes_explored', 0); method_used = method; print("Warning: traverser.traverse returned only a list (path).")
        else: print(f"Error: Unexpected return type from traverser.traverse: {type(gnn_result)}."); gnn_path_ids, gnn_nodes_explored = [], 0; method_used = method
        # *** Get History ***
        if track_exploration and hasattr(traverser, 'get_traversal_history'): gnn_history_raw = traverser.get_traversal_history() or []
    except Exception as e: print(f"--- ERROR DURING GNN TRAVERSAL ---\n{traceback.format_exc()}"); gnn_path_ids, gnn_nodes_explored = [], 0
    gnn_time = time.time() - start_time_gnn; print(f"GNN ({method_used}) done ({gnn_time:.3f}s). Path len: {len(gnn_path_ids)}, Explored: {gnn_nodes_explored}")

    # --- Prepare Response Data ---
    try: # Wrap response prep in try-except
        def get_path_node_info(node_id_list_str):
            info = []; [info.append({'id': id_str, 'title': node_titles.get(node_mapping.get(id_str), f"Node {id_str}"), 'url': node_urls.get(node_mapping.get(id_str))}) for id_str in node_id_list_str if id_str in node_mapping]; return info
        gnn_path_info = get_path_node_info(gnn_path_ids)
        bfs_path_indices_str = [str(reverse_mapping.get(idx)) for idx in bfs_path_indices if reverse_mapping.get(idx) is not None]
        bfs_path_info = get_path_node_info(bfs_path_indices_str)
        gnn_success = bool(gnn_path_info) and gnn_path_info[-1]['id'] == target_id_str if gnn_path_info else False
        efficiency_ratio = bfs_nodes_explored / gnn_nodes_explored if gnn_nodes_explored > 0 else 0
        layout_data = {str(rid): {'x': float(pos[0]), 'y': float(pos[1])} for idx, pos in graph_layout.items() if (rid := reverse_mapping.get(idx)) is not None} if graph_layout else {}
        all_edges_list = [[str(reverse_mapping.get(u)), str(reverse_mapping.get(v))] for u, v in graph_nx.edges() if reverse_mapping.get(u) is not None and reverse_mapping.get(v) is not None] if graph_nx else []

        bfs_viz_history, gnn_viz_history = [], []
        if track_exploration:
            # BFS History Processing (with safety)
            for fwd_set, bwd_set, edges_set, step in bfs_history_raw:
                fwd_set = fwd_set or set(); bwd_set = bwd_set or set(); edges_set = edges_set or set()
                fwd_ids = {str(reverse_mapping.get(i)) for i in fwd_set if reverse_mapping.get(i) is not None}
                bwd_ids = {str(reverse_mapping.get(i)) for i in bwd_set if reverse_mapping.get(i) is not None}
                edge_ids = [[str(reverse_mapping.get(u)), str(reverse_mapping.get(v))] for u, v in edges_set if reverse_mapping.get(u) is not None and reverse_mapping.get(v) is not None]
                bfs_viz_history.append({'forward_frontier': list(fwd_ids), 'backward_frontier': list(bwd_ids), 'traversed_edges': edge_ids, 'step': step})
            # GNN History Processing (with safety)
            for history_item in gnn_history_raw:
                if isinstance(history_item, tuple) and len(history_item) == 3:
                    fwd_set, bwd_set, step = history_item
                    fwd_set = fwd_set or set(); bwd_set = bwd_set or set() # *** Safety Check ***
                    fwd_ids = {str(reverse_mapping.get(i)) for i in fwd_set if reverse_mapping.get(i) is not None}
                    bwd_ids = {str(reverse_mapping.get(i)) for i in bwd_set if reverse_mapping.get(i) is not None}
                    gnn_viz_history.append({'forward_frontier': list(fwd_ids), 'backward_frontier': list(bwd_ids), 'traversed_edges': [], 'step': step})
                else: print(f"Warning: Skipping malformed GNN history item: {history_item}")

        result = {
            'success': gnn_success, 'path': gnn_path_info, 'nodes_explored': gnn_nodes_explored, 'bfs_path': bfs_path_info,
            'bfs_nodes_explored': bfs_nodes_explored, 'efficiency_ratio': efficiency_ratio, 'path_length': len(gnn_path_info),
            'bfs_path_length': len(bfs_path_info), 'method_used': method_used, 'time_taken': gnn_time, 'bfs_time_taken': bfs_time,
            'layout': layout_data, 'all_edges': all_edges_list, 'visualize_exploration': track_exploration,
            'bfs_exploration_history': bfs_viz_history if track_exploration else None, 'gnn_exploration_history': gnn_viz_history if track_exploration else None,
        }
        total_req_time = time.time() - start_req_time; print(f"Total /traverse request time: {total_req_time:.3f}s")
        # *** Explicit return of the response ***
        return jsonify(result)

    except Exception as e_final:
        print(f"--- ERROR DURING FINAL RESPONSE PREPARATION in /traverse ---")
        print(traceback.format_exc())
        return jsonify({'error': f'Failed to prepare response: {str(e_final)}'}), 500


# --- Other Routes (/config, /random_nodes, /node_info - Using safe access) ---
@app.route('/config', methods=['POST'])
def update_config():
    global config, graph_data, model, traverser, graph_nx, graph_layout
    print("--- Received Configuration Update Request ---")
    old_config = config.copy(); new_config = config.copy()
    try:
        data = request.form; files = request.files
        new_config['max_nodes'] = int(data.get('max_nodes', old_config['max_nodes']))
        new_config['beam_width'] = int(data.get('beam_width', old_config['beam_width']))
        new_config['heuristic_weight'] = float(data.get('heuristic_weight', old_config['heuristic_weight']))
        new_config['max_steps'] = int(data.get('max_steps', old_config['max_steps']))
        new_config['layout_iterations'] = int(data.get('layout_iterations', old_config['layout_iterations']))
        new_config['visualize_exploration'] = 'visualize_exploration' in data
        for key in ['edge_file', 'model_path']:
            if key in files and files[key].filename:
                file = files[key]; filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                try: file.save(filepath); new_config[key] = filepath; print(f"Saved '{filename}' for '{key}'.")
                except Exception as e: print(f"Error saving file for {key}: {e}. Using previous."); new_config[key] = old_config[key]
        print(f"Applying new configuration and re-initializing: {new_config}")
        config = new_config
        success = init_model()
        if success: print("Re-initialization successful."); session.clear(); return redirect(url_for('index'))
        else: print("Re-initialization failed. Reverting configuration."); config = old_config; init_model(); return render_template('error.html', error="Init failed with new config. Settings reverted.")
    except Exception as e: print(f"Error processing config update: {e}\n{traceback.format_exc()}"); config = old_config; init_model(); return render_template('error.html', error=f"Error updating config: {e}")

@app.route('/random_nodes', methods=['GET'])
def get_random_nodes():
    global graph_data
    if graph_data is None: return jsonify({'error': 'Graph data not ready.'}), 503
    reverse_mapping = get_data_attribute(graph_data, 'reverse_mapping', {})
    node_titles = get_data_attribute(graph_data, 'node_titles', {})
    if not reverse_mapping: return jsonify({'error': 'Node mappings missing.'}), 500
    all_indices = list(reverse_mapping.keys())
    if len(all_indices) < 2: return jsonify({'error': 'Not enough nodes.'}), 400
    max_attempts, min_len, max_len = 100, 3, 10
    for _ in range(max_attempts):
        source_idx, target_idx = random.sample(all_indices, 2)
        path_indices, _, _ = run_bidirectional_bfs(graph_data, source_idx, target_idx, max_steps=max_len + 2, track_exploration=False)
        path_len = len(path_indices)
        if min_len <= path_len <= max_len:
            source_id_str = str(reverse_mapping.get(source_idx))
            target_id_str = str(reverse_mapping.get(target_idx))
            source_title = node_titles.get(source_idx, f"Node {source_id_str}")
            target_title = node_titles.get(target_idx, f"Node {target_id_str}")
            print(f"Found random pair: {source_title} -> {target_title} (BFS Len: {path_len})")
            return jsonify({'source_id': source_id_str, 'target_id': target_id_str, 'expected_path_length': path_len})
    print(f"Could not find suitable random pair after {max_attempts} attempts.")
    source_idx, target_idx = random.sample(all_indices, 2) # Fallback
    source_id_str = str(reverse_mapping.get(source_idx))
    target_id_str = str(reverse_mapping.get(target_idx))
    source_title = node_titles.get(source_idx, f"Node {source_id_str}")
    target_title = node_titles.get(target_idx, f"Node {target_id_str}")
    return jsonify({'source_id': source_id_str, 'target_id': target_id_str, 'expected_path_length': 'Unknown'})


@app.route('/node_info')
def node_info():
    global graph_data, graph_nx
    if graph_data is None: return jsonify({'error': 'Graph data not ready.'}), 503
    node_mapping = get_data_attribute(graph_data, 'node_mapping', {}); reverse_mapping = get_data_attribute(graph_data, 'reverse_mapping', {})
    node_titles = get_data_attribute(graph_data, 'node_titles', {}); node_urls = get_data_attribute(graph_data, 'node_urls', {})
    adj_list = get_data_attribute(graph_data, 'adj_list', {})
    if not node_mapping: return jsonify({'error': 'Node mapping missing.'}), 500
    node_id_str = request.args.get('id');
    if not node_id_str: return jsonify({'error': 'No node ID provided.'}), 400
    if node_id_str not in node_mapping: return jsonify({'error': f'Node ID "{node_id_str}" not found.'}), 404
    node_idx = node_mapping[node_id_str]
    title = node_titles.get(node_idx, f"Node {node_id_str}"); url = node_urls.get(node_idx)
    neighbors_info = []; neighbor_indices = adj_list.get(node_idx, []); neighbor_count = len(neighbor_indices)
    for neighbor_idx in neighbor_indices[:50]:
        neighbor_id = reverse_mapping.get(neighbor_idx)
        if neighbor_id is not None: neighbors_info.append({'id': str(neighbor_id), 'title': node_titles.get(neighbor_idx, f"Node {str(neighbor_id)}")})
    neighbors_info.sort(key=lambda x: x['title'])
    centrality_metrics = {}
    if graph_nx is not None and node_idx in graph_nx:
        try: centrality_metrics['degree'] = graph_nx.degree(node_idx)
        except Exception as e: print(f"Warning: Could not get degree for node {node_idx}: {e}")
    return jsonify({'id': node_id_str, 'title': title, 'url': url, 'neighbor_count': neighbor_count, 'neighbors': neighbors_info, 'centrality': centrality_metrics})


# --- Main Execution ---
if __name__ == '__main__':
    print("Starting Flask application initialization...")
    if not init_model(): sys.exit(1) # Exit if init fails
    print("--- Starting Flask Development Server ---")
    # Make accessible on local network by default
    app.run(host='0.0.0.0', port=5000, debug=True) # Turn debug=False for production