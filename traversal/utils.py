import torch
import numpy as np
from collections import deque

def bidirectional_bfs(data, start_idx, target_idx):
    """
    Perform bidirectional BFS to find the shortest path between two nodes.
    Searches from both start and target simultaneously.
    
    Args:
        data: Graph data object
        start_idx: Starting node index
        target_idx: Target node index
        
    Returns:
        path: List of nodes in the path (or empty list if no path exists)
        nodes_explored: Number of nodes explored during search
    """
    if start_idx == target_idx:
        return [start_idx], 1
    
    # Initialize forward and backward searches
    forward_queue = deque([(start_idx, [start_idx])])
    backward_queue = deque([(target_idx, [target_idx])])
    
    forward_visited = {start_idx: [start_idx]}
    backward_visited = {target_idx: [target_idx]}
    
    nodes_explored = 1  # Start node
    
    while forward_queue and backward_queue:
        # Forward search
        curr_node, curr_path = forward_queue.popleft()
        
        for neighbor in data.adj_list.get(curr_node, []):
            nodes_explored += 1
            if neighbor in backward_visited:
                # Found intersection, construct the path
                forward_path = curr_path
                backward_path = backward_visited[neighbor]
                
                # Combine paths (reverse the backward path)
                full_path = forward_path + backward_path[::-1][1:]
                return full_path, nodes_explored
            
            if neighbor not in forward_visited:
                new_path = curr_path + [neighbor]
                forward_visited[neighbor] = new_path
                forward_queue.append((neighbor, new_path))
        
        # Backward search
        curr_node, curr_path = backward_queue.popleft()
        
        for neighbor in data.adj_list.get(curr_node, []):
            nodes_explored += 1
            if neighbor in forward_visited:
                # Found intersection, construct the path
                forward_path = forward_visited[neighbor]
                backward_path = curr_path
                
                # Combine paths (reverse the backward path)
                full_path = forward_path + backward_path[::-1][1:]
                return full_path, nodes_explored
            
            if neighbor not in backward_visited:
                new_path = curr_path + [neighbor]
                backward_visited[neighbor] = new_path
                backward_queue.append((neighbor, new_path))
    
    # No path found
    return [], nodes_explored