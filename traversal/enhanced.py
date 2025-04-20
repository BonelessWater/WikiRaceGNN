import torch
import torch.nn.functional as F
import heapq
from collections import defaultdict, deque
from traversal.base import BaseTraverser
from utils.data import neighbor_sampler

class EnhancedWikiTraverser(BaseTraverser):
    """
    Enhanced Wikipedia graph traversal with sophisticated search algorithms.
    Implements multiple search strategies optimized for different scenarios.
    """
    def __init__(self, model, data, device, beam_width=5, heuristic_weight=1.5, 
                 max_memory_nodes=1000, num_neighbors=20, num_hops=2):
        super().__init__(model, data, device, num_neighbors, num_hops)
        self.beam_width = beam_width
        self.heuristic_weight = heuristic_weight
        self.max_memory_nodes = max_memory_nodes
        self.cache = {}  # Cache for node embeddings
        
    def reset(self):
        """Reset the traverser's state"""
        self.reset_exploration_counter()
        self.cache = {}
        
    def get_cached_embedding(self, node_idx):
        """Get a node embedding with caching for efficiency"""
        if node_idx in self.cache:
            return self.cache[node_idx]
        
        embedding = self.get_node_embedding(node_idx)
        
        # Cache the result if we have space
        if len(self.cache) < self.max_memory_nodes:
            self.cache[node_idx] = embedding
            
        return embedding
    
    def score_node(self, node_idx, target_idx, path_cost):
        """
        A* scoring function: f = g + h
        g = path cost so far
        h = heuristic estimate (embedding similarity to target)
        """
        node_embedding = self.get_cached_embedding(node_idx)
        target_embedding = self.get_cached_embedding(target_idx)
        
        # Heuristic: negative cosine similarity (higher means more similar)
        similarity = F.cosine_similarity(node_embedding.unsqueeze(0), target_embedding.unsqueeze(0))
        # Convert to a distance-like metric (lower is better for A*)
        heuristic = -similarity.item()
        
        # A* formula
        return path_cost + self.heuristic_weight * heuristic
    
    def parallel_beam_search(self, start_idx, target_idx, max_steps=30):
        """
        Parallel beam search with GNN guidance.
        Maintains multiple candidate paths and expands them in parallel.
        
        Args:
            start_idx: Starting node index
            target_idx: Target node index
            max_steps: Maximum number of steps to take
            
        Returns:
            path: Path from start to target
            nodes_explored: Number of nodes explored
        """
        if start_idx == target_idx:
            return [start_idx], 1
        
        # Initialize
        self.reset()
        target_emb = self.get_cached_embedding(target_idx)
        
        visited = {start_idx}
        paths = {start_idx: [start_idx]}
        current_beams = [(0, start_idx)]  # (score, node_idx)
        
        for step in range(max_steps):
            if not current_beams:
                break
                
            # Extract current beam nodes
            current_nodes = [node for _, node in current_beams]
            
            # Check if target is in the current beam
            if target_idx in current_nodes:
                return paths[target_idx], self.nodes_explored
            
            # Process all current beams in parallel
            all_candidates = []
            
            for node_idx in current_nodes:
                # Sample neighbors
                sampled_nodes = self.sample_neighbors(node_idx, num_hops=1)
                
                # Filter out already visited nodes
                valid_nodes = [n.item() for n in sampled_nodes if n.item() not in visited]
                
                if not valid_nodes:
                    continue
                    
                # Get embeddings and score all at once
                nodes_tensor = torch.tensor(valid_nodes, dtype=torch.long)
                sub_x, sub_edge_index, mapping = self.build_subgraph(nodes_tensor)
                
                batch = torch.zeros(len(valid_nodes), dtype=torch.long).to(self.device)
                with torch.no_grad():
                    self.model.eval()
                    nodes_embeddings = self.model(sub_x, sub_edge_index, batch)
                
                # Score all nodes
                scores = self.model.score_neighbors(nodes_embeddings, valid_nodes, target_emb)
                
                # Add candidates with their scores and paths
                for i, next_node in enumerate(valid_nodes):
                    if next_node not in visited:
                        score = scores[i].item()
                        new_path = paths[node_idx] + [next_node]
                        all_candidates.append((score, next_node, new_path))
            
            # No valid candidates found
            if not all_candidates:
                break
                
            # Sort and keep top-k candidates
            all_candidates.sort(key=lambda x: x[0], reverse=True)
            new_beams = []
            
            for score, next_node, new_path in all_candidates[:self.beam_width]:
                if next_node not in visited:
                    new_beams.append((score, next_node))
                    visited.add(next_node)
                    paths[next_node] = new_path
                    
                    # Early stopping if we found the target
                    if next_node == target_idx:
                        return new_path, self.nodes_explored
            
            current_beams = new_beams
        
        # If we didn't find a path, return the best path we have
        if current_beams:
            best_node = max(current_beams, key=lambda x: x[0])[1]
            return paths[best_node], self.nodes_explored
        
        # No path found
        return [], self.nodes_explored
    
    def bidirectional_guided_search(self, start_idx, target_idx, max_steps=30):
        """
        Bidirectional A* search with GNN guidance.
        Searches from both start and target, meeting in the middle.
        
        Args:
            start_idx: Starting node index
            target_idx: Target node index
            max_steps: Maximum number of steps to take
            
        Returns:
            path: Path from start to target
            nodes_explored: Number of nodes explored
        """
        if start_idx == target_idx:
            return [start_idx], 1
        
        # Reset state
        self.reset()
        
        # Initialize embeddings
        start_emb = self.get_cached_embedding(start_idx)
        target_emb = self.get_cached_embedding(target_idx)
        
        # Priority queues for forward and backward search
        # Format: (priority, cost_so_far, node, path)
        forward_queue = [(0, 0, start_idx, [start_idx])]
        backward_queue = [(0, 0, target_idx, [target_idx])]
        
        # Visited sets with path costs
        forward_visited = {start_idx: 0}  # node -> cost
        backward_visited = {target_idx: 0}
        
        # Best paths for each node
        forward_paths = {start_idx: [start_idx]}
        backward_paths = {target_idx: [target_idx]}
        
        # For tracking the best meeting point
        best_meeting_node = None
        best_meeting_cost = float('inf')
        
        for step in range(max_steps):
            # Decide which direction to expand
            if len(forward_queue) <= len(backward_queue) and forward_queue:
                direction = "forward"
                queue = forward_queue
                visited = forward_visited
                paths = forward_paths
                other_visited = backward_visited
                other_paths = backward_paths
                target_embedding = target_emb
            elif backward_queue:
                direction = "backward"
                queue = backward_queue
                visited = backward_visited
                paths = backward_paths
                other_visited = forward_visited
                other_paths = forward_paths
                target_embedding = start_emb
            else:
                # Both queues empty
                break
            
            # Pop the best candidate
            _, cost, node, path = heapq.heappop(queue)
            
            # Check if this node has been visited from the other direction
            if node in other_visited:
                total_cost = cost + other_visited[node]
                if total_cost < best_meeting_cost:
                    best_meeting_node = node
                    best_meeting_cost = total_cost
            
            # Sample and score neighbors
            sampled_nodes = self.sample_neighbors(node, num_hops=1)
            
            # Filter nodes that we haven't visited or have found a better path to
            valid_nodes = [n.item() for n in sampled_nodes if n.item() not in visited or 
                          cost + 1 < visited[n.item()]]
            
            if valid_nodes:
                # Get embeddings for all valid nodes
                sub_x, sub_edge_index, mapping = self.build_subgraph(torch.tensor(valid_nodes, dtype=torch.long))
                
                batch = torch.zeros(len(valid_nodes), dtype=torch.long).to(self.device)
                with torch.no_grad():
                    self.model.eval()
                    nodes_embeddings = self.model(sub_x, sub_edge_index, batch)
                
                # Score neighbors using the appropriate embedding (target or start)
                scores = self.model.score_neighbors(nodes_embeddings, valid_nodes, target_embedding)
                
                # Add neighbors to queue
                for i, next_node in enumerate(valid_nodes):
                    new_cost = cost + 1
                    
                    # Only add if we found a better path
                    if next_node not in visited or new_cost < visited[next_node]:
                        # Calculate priority using A* formula
                        heuristic = -scores[i].item()
                        priority = new_cost + self.heuristic_weight * heuristic
                        
                        # Update path
                        new_path = path + [next_node]
                        visited[next_node] = new_cost
                        paths[next_node] = new_path
                        
                        # Add to priority queue
                        if direction == "forward":
                            heapq.heappush(forward_queue, (priority, new_cost, next_node, new_path))
                        else:
                            heapq.heappush(backward_queue, (priority, new_cost, next_node, new_path))
                        
                        # Check if this is a meeting point
                        if next_node in other_visited:
                            total_cost = new_cost + other_visited[next_node]
                            if total_cost < best_meeting_cost:
                                best_meeting_node = next_node
                                best_meeting_cost = total_cost
            
            # Early termination if we've found a good meeting point
            # and all remaining candidates are worse
            if best_meeting_node is not None:
                # Check if we can terminate early
                if ((not forward_queue or forward_queue[0][0] >= best_meeting_cost) and
                    (not backward_queue or backward_queue[0][0] >= best_meeting_cost)):
                    break
        
        # Reconstruct path if meeting point was found
        if best_meeting_node is not None:
            forward_path = forward_paths.get(best_meeting_node, [start_idx])
            backward_path = backward_paths.get(best_meeting_node, [target_idx])
            
            # Combine paths (remove duplicate meeting point and reverse backward path)
            full_path = forward_path + backward_path[::-1][1:]
            return full_path, self.nodes_explored
        
        # If no path found, return empty path
        return [], self.nodes_explored
    
    def traverse(self, start_node_id, target_node_id=None, max_steps=30, method="auto"):
        """
        Main traversal method that selects the best algorithm based on the graph structure
        and traversal distance.
        
        Args:
            start_node_id: Starting node ID
            target_node_id: Target node ID (if None, will explore without target)
            max_steps: Maximum steps to take
            method: Search method to use ("auto", "parallel_beam", "bidirectional_guided", or "hybrid")
            
        Returns:
            path: List of node IDs in the traversal path
            nodes_explored: Number of nodes explored during traversal
        """
        self.reset()
        
        # Convert original IDs to internal indices
        if start_node_id not in self.data.node_mapping:
            raise ValueError(f"Start node ID {start_node_id} not found in the graph")
        
        start_idx = self.data.node_mapping[start_node_id]
        
        # If target is specified, convert it too
        if target_node_id is not None:
            if target_node_id not in self.data.node_mapping:
                raise ValueError(f"Target node ID {target_node_id} not found in the graph")
            target_idx = self.data.node_mapping[target_node_id]
        else:
            # If no target, we'll just explore (not implemented here)
            raise ValueError("Exploration without target is not implemented yet")
        
        # Choose search method
        if method == "auto":
            # Get embeddings to estimate distance
            start_emb = self.get_cached_embedding(start_idx)
            target_emb = self.get_cached_embedding(target_idx)
            
            # Calculate similarity as a proxy for distance
            similarity = F.cosine_similarity(start_emb.unsqueeze(0), target_emb.unsqueeze(0)).item()
            
            # Choose method based on estimated distance
            if similarity > 0.5:  # Nodes are close
                method = "parallel_beam"
            else:  # Nodes are far apart
                method = "bidirectional_guided"
        
        # Execute the selected method
        if method == "parallel_beam":
            path, nodes_explored = self.parallel_beam_search(
                start_idx, target_idx, max_steps=max_steps
            )
        elif method == "bidirectional_guided":
            path, nodes_explored = self.bidirectional_guided_search(
                start_idx, target_idx, max_steps=max_steps
            )
        elif method == "hybrid":
            # Try parallel beam search first with fewer steps
            path, nodes_explored1 = self.parallel_beam_search(
                start_idx, target_idx, max_steps=max_steps // 2
            )
            
            if not path or path[-1] != target_idx:
                # If not successful, try bidirectional search
                path, nodes_explored2 = self.bidirectional_guided_search(
                    start_idx, target_idx, max_steps=max_steps
                )
                nodes_explored = nodes_explored1 + nodes_explored2
            else:
                nodes_explored = nodes_explored1
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Convert internal indices back to original node IDs
        return [self.data.reverse_mapping[idx] for idx in path], nodes_explored