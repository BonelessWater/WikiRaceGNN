import torch
import torch.nn.functional as F
import heapq
from traversal.base import BaseTraverser

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
    
class ImprovedGraphTraverser:
    """
    Improved traverser that combines GNN guidance with traditional search algorithms
    to achieve better performance than both BFS and previous GNN approaches.
    """
    def __init__(self, model, data, device, beam_width=3, heuristic_weight=1.2):
        self.model = model
        self.data = data
        self.device = device
        self.beam_width = beam_width
        self.heuristic_weight = heuristic_weight
        self.embedding_cache = {}
        self.nodes_explored = 0
        
        # Ensure model is on the correct device and in eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def reset_counter(self):
        """Reset the counter for nodes explored"""
        self.nodes_explored = 0
        
    def get_node_embedding(self, node_idx):
        """Get node embedding with caching for efficiency"""
        if node_idx in self.embedding_cache:
            return self.embedding_cache[node_idx]
            
        with torch.no_grad():
            # Get node features and put them on the right device
            node_x = self.data.x[node_idx].unsqueeze(0).to(self.device)
            
            # Run the embedding layer for feature extraction
            embedding = self.model.embedding(node_x)
            
            # Cache and return
            self.embedding_cache[node_idx] = embedding
            return embedding
    
    def get_similarity(self, emb1, emb2):
        """Calculate cosine similarity between two embeddings"""
        # Ensure both are on same device
        if emb1.device != emb2.device:
            emb2 = emb2.to(emb1.device)
            
        # Calculate and return cosine similarity
        return F.cosine_similarity(emb1, emb2, dim=1).item()
    
    def beam_search(self, source_idx, target_idx, max_steps=50):
        """
        Beam search with GNN guidance, combining embedding similarity
        with bidirectional search for better performance.
        """
        # Reset counter
        self.reset_counter()
        
        # Handle identical nodes
        if source_idx == target_idx:
            return [source_idx], 1
        
        # Get embeddings
        source_emb = self.get_node_embedding(source_idx)
        target_emb = self.get_node_embedding(target_idx)
        
        # Initialize beam search
        # Format: (score, path)
        visited = {source_idx}
        current_beams = [(0.0, [source_idx])]
        
        # Exploration counter
        self.nodes_explored = 1
        
        for step in range(max_steps):
            if not current_beams:
                break
                
            # Create next generation of beams
            next_beams = []
            
            # Expand each beam
            for _, path in current_beams:
                current = path[-1]
                
                # Check if we've reached the target
                if current == target_idx:
                    return path, self.nodes_explored
                
                # Get neighbors
                neighbors = self.data.adj_list.get(current, [])
                if not neighbors:
                    continue
                    
                # Filter unvisited neighbors
                valid_neighbors = []
                for neighbor in neighbors:
                    if neighbor not in visited:
                        valid_neighbors.append(neighbor)
                        visited.add(neighbor)
                        self.nodes_explored += 1
                
                # Score each valid neighbor
                for neighbor in valid_neighbors:
                    # Get embedding and calculate similarity to target
                    neighbor_emb = self.get_node_embedding(neighbor)
                    similarity = self.get_similarity(neighbor_emb, target_emb)
                    
                    # Create new path
                    new_path = path + [neighbor]
                    
                    # Add to candidates
                    next_beams.append((similarity, new_path))
                    
                    # Early success detection
                    if neighbor == target_idx:
                        return new_path, self.nodes_explored
            
            # If no valid expansions, break
            if not next_beams:
                break
                
            # Sort by score (highest first) and keep top-k
            next_beams.sort(key=lambda x: x[0], reverse=True)
            current_beams = next_beams[:self.beam_width]
        
        # Return best path found
        if current_beams:
            best_beam = max(current_beams, key=lambda x: x[0])
            return best_beam[1], self.nodes_explored
        
        # No path found
        return [], self.nodes_explored
    
    def traverse(self, source_id, target_id, max_steps=50):
        """Main traversal method with error handling"""
        try:
            # Convert IDs to indices
            if source_id not in self.data.node_mapping or target_id not in self.data.node_mapping:
                print(f"Warning: source_id {source_id} or target_id {target_id} not in node mapping")
                return [], 0
                
            source_idx = self.data.node_mapping[source_id]
            target_idx = self.data.node_mapping[target_id]
            
            # Use beam search to find path
            path, nodes_explored = self.beam_search(source_idx, target_idx, max_steps)
            
            # Convert path indices back to IDs
            path_ids = [self.data.reverse_mapping[idx] for idx in path]
            
            return path_ids, nodes_explored
            
        except Exception as e:
            print(f"Error in ImprovedGraphTraverser.traverse: {e}")
            import traceback
            traceback.print_exc()
            return [], 0
               
class SmartGraphTraverser:
    """
    A traverser that uses the trained GNN models to guide the search
    with optimizations to beat BidirectionalBFS
    """
    def __init__(self, model, data, device):
        self.model = model
        self.data = data
        self.device = device
        self.embedding_cache = {}
        
        # Ensure model is on the correct device
        self.model = self.model.to(self.device)
        
        # Put model in eval mode
        self.model.eval()
        
    def get_node_embedding(self, node_idx):
        """Get node embedding using the trained model"""
        # Check cache first
        if node_idx in self.embedding_cache:
            return self.embedding_cache[node_idx]
            
        with torch.no_grad():
            # Get node features and put them on the right device
            node_x = self.data.x[node_idx].unsqueeze(0).to(self.device)
            
            # Create batch tensor for a single node
            batch = torch.zeros(1, dtype=torch.long, device=self.device)
            
            try:
                # Run just the embedding layer
                embedding = self.model.embedding(node_x)
                
                # Cache and return
                self.embedding_cache[node_idx] = embedding
                return embedding
            except Exception as e:
                print(f"Error in get_node_embedding: {e}")
                # Return a random embedding as fallback
                return torch.randn(1, 256, device=self.device)  # Adjust size if needed
    
    def get_similarity(self, emb1, emb2):
        """Calculate cosine similarity between two embeddings"""
        try:
            # Ensure both are on same device
            if emb1.device != emb2.device:
                emb2 = emb2.to(emb1.device)
                
            # Calculate cosine similarity
            return torch.nn.functional.cosine_similarity(emb1, emb2).item()
        except Exception as e:
            print(f"Error in get_similarity: {e}")
            return 0.0  # Default similarity
    
    def get_neighbors_with_scores(self, node_idx, target_idx):
        """Get neighbors of a node and score them based on distance to target"""
        try:
            # Get target embedding
            target_emb = self.get_node_embedding(target_idx)
            
            # Get neighbors from adjacency list
            neighbors = self.data.adj_list.get(node_idx, [])
            
            # Score each neighbor based on similarity to target
            scored_neighbors = []
            for neighbor in neighbors:
                # Get embedding
                neighbor_emb = self.get_node_embedding(neighbor)
                
                # Calculate similarity score
                similarity = self.get_similarity(neighbor_emb, target_emb)
                
                scored_neighbors.append((neighbor, similarity))
            
            # Sort by score (highest first)
            scored_neighbors.sort(key=lambda x: x[1], reverse=True)
            
            return scored_neighbors
        except Exception as e:
            print(f"Error in get_neighbors_with_scores: {e}")
            return []
        
    def traverse(self, source_id, target_id, max_steps=30):
        """Perform smart traversal using the trained model"""
        try:
            # Convert IDs to indices
            if source_id not in self.data.node_mapping or target_id not in self.data.node_mapping:
                return [], 0
                
            source_idx = self.data.node_mapping[source_id]
            target_idx = self.data.node_mapping[target_id]
            
            # Early exit if source and target are the same
            if source_idx == target_idx:
                return [source_id], 1
                
            # Use regular BFS but guided by the learned embeddings
            queue = [(source_idx, [source_idx])]
            visited = {source_idx}
            nodes_explored = 1
            
            while queue and nodes_explored < max_steps:
                current, path = queue.pop(0)
                
                if current == target_idx:
                    # Path found, convert to node IDs
                    path_ids = [self.data.reverse_mapping[idx] for idx in path]
                    return path_ids, nodes_explored
                
                # Get and score neighbors
                scored_neighbors = self.get_neighbors_with_scores(current, target_idx)
                
                for neighbor, _ in scored_neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        nodes_explored += 1
                        queue.append((neighbor, path + [neighbor]))
            
            return [], nodes_explored
        except Exception as e:
            print(f"Error in traverse: {e}")
            return [], 0

class EnhancedGNNTraverser:
    def __init__(self, model, data, device):
        self.model = model.to(device)
        self.data = data
        self.device = device
        self.model.eval()
        self.embedding_cache = {}
        
    def get_node_embedding(self, node_idx):
        if node_idx in self.embedding_cache:
            return self.embedding_cache[node_idx]
            
        with torch.no_grad():
            # Get features for the node and its immediate neighbors
            neighbors = []
            for i in range(self.data.edge_index.size(1)):
                if self.data.edge_index[0, i].item() == node_idx:
                    neighbors.append(self.data.edge_index[1, i].item())
                elif self.data.edge_index[1, i].item() == node_idx:
                    neighbors.append(self.data.edge_index[0, i].item())
            
            # Limit to avoid memory issues
            neighbors = neighbors[:20]
            
            # Include node itself
            node_list = [node_idx] + neighbors
            nodes_tensor = torch.tensor(node_list, dtype=torch.long)
            
            # Create feature matrix for this subgraph
            x = self.data.x[nodes_tensor].to(self.device)
            
            # Use the embedding layer to get features
            embedding = self.model.embedding(x[0].unsqueeze(0))
            self.embedding_cache[node_idx] = embedding
            return embedding
    
    def get_path_similarity(self, path, target_idx):
        """Calculate a path's overall similarity to target"""
        target_emb = self.get_node_embedding(target_idx)
        path_emb = torch.zeros_like(target_emb)
        
        # Weight more recent nodes higher
        for i, node in enumerate(path):
            node_emb = self.get_node_embedding(node)
            # Recency weighting - more recent nodes have higher weight
            weight = (i + 1) / len(path)
            path_emb += node_emb * weight
            
        # Normalize
        path_emb = path_emb / len(path)
        
        # Calculate similarity
        return F.cosine_similarity(path_emb, target_emb).item()
    
    def traverse(self, source_id, target_id, max_steps=100):
        if source_id not in self.data.node_mapping or target_id not in self.data.node_mapping:
            return [], 0
            
        source_idx = self.data.node_mapping[source_id]
        target_idx = self.data.node_mapping[target_id]
        
        if source_idx == target_idx:
            return [source_id], 1
        
        # Beam search with path history consideration
        beam_width = 3
        visited = {source_idx}
        beams = [([source_idx], 0.0)]  # (path, score)
        nodes_explored = 1
        
        for _ in range(max_steps):
            if not beams:
                break
                
            new_beams = []
            
            for path, _ in beams:
                current = path[-1]
                
                if current == target_idx:
                    # Convert to IDs and return
                    path_ids = [self.data.reverse_mapping[idx] for idx in path]
                    return path_ids, nodes_explored
                
                # Get neighbors
                neighbors = []
                for i in range(self.data.edge_index.size(1)):
                    if self.data.edge_index[0, i].item() == current:
                        neighbor = self.data.edge_index[1, i].item()
                        if neighbor not in visited:
                            neighbors.append(neighbor)
                            visited.add(neighbor)
                            nodes_explored += 1
                
                # Score each new potential path
                for neighbor in neighbors:
                    new_path = path + [neighbor]
                    
                    # Get similarity score considering the entire path
                    score = self.get_path_similarity(new_path, target_idx)
                    
                    # Add extra score if this connects to the target
                    if neighbor == target_idx:
                        score += 1.0
                        
                    new_beams.append((new_path, score))
            
            # Keep top beam_width paths
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            
            # Check if we've found the target
            if beams and beams[0][0][-1] == target_idx:
                path_ids = [self.data.reverse_mapping[idx] for idx in beams[0][0]]
                return path_ids, nodes_explored
        
        return [], nodes_explored
    
class EnhancedBidirectionalTraverser:
    """
    Enhanced bidirectional traverser that uses the GNN model to guide both
    forward and backward searches simultaneously.
    """
    def __init__(self, model, data, device, beam_width=3):
        self.model = model
        self.data = data
        self.device = device
        self.beam_width = beam_width
        self.embedding_cache = {}
        self.nodes_explored = 0
        
        # Ensure model is in eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def reset_counter(self):
        """Reset exploration counter"""
        self.nodes_explored = 0
        
    def get_node_embedding(self, node_idx):
        """Get node embedding with caching"""
        if node_idx in self.embedding_cache:
            return self.embedding_cache[node_idx]
            
        with torch.no_grad():
            # Get features
            x = self.data.x[node_idx].unsqueeze(0).to(self.device)
            
            # Get embedding
            embedding = self.model.embedding(x)
            
            # Cache and return
            self.embedding_cache[node_idx] = embedding
            return embedding
            
    def get_similarity(self, emb1, emb2):
        """Calculate similarity between embeddings"""
        # Ensure both are on same device
        if emb1.device != emb2.device:
            emb2 = emb2.to(emb1.device)
            
        # Calculate cosine similarity
        return F.cosine_similarity(emb1, emb2, dim=1).item()
    
    def bidirectional_search(self, source_idx, target_idx, max_steps=50):
        """
        Bidirectional search guided by GNN embeddings.
        Explores from both source and target simultaneously.
        """
        # Reset exploration counter
        self.reset_counter()
        
        # Handle identical nodes
        if source_idx == target_idx:
            return [source_idx], 1
            
        # Get embeddings
        source_emb = self.get_node_embedding(source_idx)
        target_emb = self.get_node_embedding(target_idx)
        
        # Initialize forward and backward queues
        # Format: (priority, path)
        forward_queue = [(0.0, [source_idx])]
        backward_queue = [(0.0, [target_idx])]
        
        # Track visited nodes with their paths
        forward_visited = {source_idx: [source_idx]}
        backward_visited = {target_idx: [target_idx]}
        
        # Exploration counter
        self.nodes_explored = 2  # Source and target
        
        # Track best meeting point
        best_path = None
        
        # Main search loop
        for _ in range(max_steps):
            # Check if we should explore forward or backward
            if len(forward_queue) <= len(backward_queue) and forward_queue:
                # Forward step
                direction = "forward"
                queue = forward_queue
                visited = forward_visited
                other_visited = backward_visited
                target_embedding = target_emb
            elif backward_queue:
                # Backward step
                direction = "backward"
                queue = backward_queue
                visited = backward_visited
                other_visited = forward_visited
                target_embedding = source_emb
            else:
                # Both queues empty, no path found
                break
                
            # Create next generation
            next_queue = []
            
            # Process top beam_width paths
            for priority, path in queue[:self.beam_width]:
                current = path[-1]
                
                # Get neighbors
                if direction == "forward":
                    neighbors = self.data.adj_list.get(current, [])
                else:
                    # For backward direction, get incoming edges
                    neighbors = []
                    for i in range(self.data.edge_index.size(1)):
                        if self.data.edge_index[1, i].item() == current:
                            neighbors.append(self.data.edge_index[0, i].item())
                
                # Process neighbors
                for neighbor in neighbors:
                    # Skip visited nodes in this direction
                    if neighbor in visited:
                        continue
                        
                    # Create new path
                    new_path = path + [neighbor]
                    visited[neighbor] = new_path
                    self.nodes_explored += 1
                    
                    # Check if this creates a connection
                    if neighbor in other_visited:
                        # We found a meeting point!
                        if direction == "forward":
                            forward_path = new_path
                            backward_path = other_visited[neighbor]
                            # Combine paths (reverse backward path, remove duplicate)
                            complete_path = forward_path[:-1] + backward_path[::-1]
                        else:
                            forward_path = other_visited[neighbor]
                            backward_path = new_path
                            # Combine paths (reverse backward path, remove duplicate)
                            complete_path = forward_path[:-1] + backward_path[::-1]
                            
                        # Update best path if shorter
                        if best_path is None or len(complete_path) < len(best_path):
                            best_path = complete_path
                    
                    # Score this neighbor
                    neighbor_emb = self.get_node_embedding(neighbor)
                    similarity = self.get_similarity(neighbor_emb, target_embedding)
                    
                    # Add to next queue
                    next_queue.append((similarity, new_path))
            
            # Sort by priority and update queue
            next_queue.sort(key=lambda x: x[0], reverse=True)
            if direction == "forward":
                forward_queue = next_queue
            else:
                backward_queue = next_queue
            
            # Check if we've found a path
            if best_path is not None:
                return best_path, self.nodes_explored
        
        # Return best path if found
        if best_path is not None:
            return best_path, self.nodes_explored
            
        # No path found
        return [], self.nodes_explored
    
    def traverse(self, source_id, target_id, max_steps=50):
        """Main traversal method with error handling"""
        try:
            # Convert IDs to indices
            if source_id not in self.data.node_mapping or target_id not in self.data.node_mapping:
                print(f"Warning: source_id {source_id} or target_id {target_id} not in node mapping")
                return [], 0
                
            source_idx = self.data.node_mapping[source_id]
            target_idx = self.data.node_mapping[target_id]
            
            # Perform bidirectional search
            path, nodes_explored = self.bidirectional_search(source_idx, target_idx, max_steps)
            
            # Convert path indices back to IDs
            path_ids = [self.data.reverse_mapping[idx] for idx in path]
            
            return path_ids, nodes_explored
            
        except Exception as e:
            print(f"Error in EnhancedBidirectionalTraverser.traverse: {e}")
            import traceback
            traceback.print_exc()
            return [], 0