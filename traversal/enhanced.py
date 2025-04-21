import torch
import torch.nn.functional as F
import heapq
import numpy as np
from traversal.base import BaseTraverser
from gensim.utils import simple_preprocess

class Word2VecEnhancedTraverser:
    """
    A graph traverser that leverages Word2Vec semantics for more intelligent path finding.
    This traverser combines GNN embeddings with Word2Vec semantic understanding.
    """
    def __init__(self, model, data, device, beam_width=5, heuristic_weight=1.5, 
                 max_memory_nodes=1000, num_neighbors=20, num_hops=2):
        self.model = model
        self.data = data
        self.device = device
        self.beam_width = beam_width
        self.heuristic_weight = heuristic_weight
        self.max_memory_nodes = max_memory_nodes
        self.num_neighbors = num_neighbors
        self.num_hops = num_hops
        
        # State tracking
        self.cache = {}  # Cache for node embeddings
        self.nodes_explored = 0
        
        # Get Word2Vec model from the GNN model if available
        self.word2vec_model = getattr(model, 'word2vec_model', None)
        
        # Check if node titles are available
        self.has_titles = hasattr(data, 'node_titles') and data.node_titles
        
        # Set semantic similarity type
        self.use_word2vec_similarity = self.word2vec_model is not None and self.has_titles
        
        if self.use_word2vec_similarity:
            print("Using Word2Vec semantic similarity for traversal")
        else:
            print("Word2Vec not available, using only GNN embeddings")
    
    def reset_exploration_counter(self):
        """Reset the counter for nodes explored during traversal"""
        self.nodes_explored = 0
        
    def get_node_embedding(self, node_idx):
        """
        Get the embedding for a single node.
        
        Args:
            node_idx: Node index
            
        Returns:
            embedding: Node embedding vector
        """
        if node_idx in self.cache:
            return self.cache[node_idx]
        
        with torch.no_grad():
            # Create a single-node feature matrix
            node_x = self.data.x[node_idx].unsqueeze(0).to(self.device)
            
            # Create a batch indicator (just zeros for a single node)
            batch = torch.zeros(1, dtype=torch.long, device=self.device)
            
            # Create a local neighborhood subgraph
            import torch_geometric.utils as utils
            
            # Get node neighbors at 1-hop distance
            neighbors = []
            for i in range(self.data.edge_index.size(1)):
                if self.data.edge_index[0, i].item() == node_idx:
                    neighbors.append(self.data.edge_index[1, i].item())
                elif self.data.edge_index[1, i].item() == node_idx:
                    neighbors.append(self.data.edge_index[0, i].item())
            
            # Include the node itself
            subgraph_nodes = [node_idx] + neighbors
            subgraph_nodes = list(set(subgraph_nodes))  # Remove duplicates
            
            # Convert to tensor
            sub_nodes = torch.tensor(subgraph_nodes, dtype=torch.long)
            
            # Extract subgraph
            sub_x = self.data.x[sub_nodes].to(self.device)
            
            # Create mapping for edge indices
            mapping = {n: i for i, n in enumerate(subgraph_nodes)}
            
            # Extract edges that connect nodes in the subgraph
            edge_list = []
            for i in range(self.data.edge_index.size(1)):
                src = self.data.edge_index[0, i].item()
                dst = self.data.edge_index[1, i].item()
                if src in mapping and dst in mapping:
                    edge_list.append([mapping[src], mapping[dst]])
            
            # Create edge index tensor for subgraph
            if edge_list:
                sub_edge_index = torch.tensor(edge_list, dtype=torch.long).t().to(self.device)
            else:
                # If no edges, create a self-loop
                sub_edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(self.device)
            
            # Create batch indicator for subgraph
            sub_batch = torch.zeros(sub_x.size(0), dtype=torch.long, device=self.device)
            
            # Run the model on the subgraph
            node_embeddings = self.model(sub_x, sub_edge_index, sub_batch)
            
            # Find the embedding of the target node
            embedding = node_embeddings[mapping[node_idx]]
            
            # Cache the result if we have space
            if len(self.cache) < self.max_memory_nodes:
                self.cache[node_idx] = embedding
                
            return embedding
    
    def get_word2vec_similarity(self, node1_idx, node2_idx):
        """
        Calculate semantic similarity between two nodes using Word2Vec.
        
        Args:
            node1_idx: First node index
            node2_idx: Second node index
            
        Returns:
            float: Similarity score
        """
        if not self.use_word2vec_similarity:
            return 0.0
        
        # Get node titles
        title1 = self.data.node_titles.get(node1_idx, "")
        title2 = self.data.node_titles.get(node2_idx, "")
        
        if not title1 or not title2:
            return 0.0
        
        # Preprocess titles
        tokens1 = simple_preprocess(title1.replace("_", " "), deacc=True)
        tokens2 = simple_preprocess(title2.replace("_", " "), deacc=True)
        
        if not tokens1 or not tokens2:
            return 0.0
        
        # Calculate average word vectors for each title
        vecs1 = []
        vecs2 = []
        
        for token in tokens1:
            if token in self.word2vec_model.wv:
                vecs1.append(self.word2vec_model.wv[token])
        
        for token in tokens2:
            if token in self.word2vec_model.wv:
                vecs2.append(self.word2vec_model.wv[token])
        
        if not vecs1 or not vecs2:
            return 0.0
        
        # Average the vectors
        vec1 = np.mean(vecs1, axis=0)
        vec2 = np.mean(vecs2, axis=0)
        
        # Calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        sim = cosine_similarity([vec1], [vec2])[0][0]
        
        return sim
    
    def sample_neighbors(self, node_idx, exclude_nodes=None):
        """
        Sample neighbors for a node.
        
        Args:
            node_idx: Index of the node
            exclude_nodes: Set of nodes to exclude
            
        Returns:
            list: List of sampled neighbor node indices
        """
        if exclude_nodes is None:
            exclude_nodes = set()
            
        # Get neighbors from adjacency list
        if node_idx in self.data.adj_list:
            neighbors = self.data.adj_list[node_idx]
        else:
            # Fallback to edge_index lookup
            neighbors = []
            for i in range(self.data.edge_index.size(1)):
                if self.data.edge_index[0, i].item() == node_idx:
                    neighbors.append(self.data.edge_index[1, i].item())
        
        # Filter out excluded nodes
        valid_neighbors = [n for n in neighbors if n not in exclude_nodes]
        
        # Update exploration counter
        self.nodes_explored += len(valid_neighbors)
        
        return valid_neighbors
    
    def score_node(self, node_idx, target_idx, current_idx=None, path_cost=0):
        """
        Score a node based on similarity to target and path cost.
        Enhanced with Word2Vec semantic guidance.
        
        Args:
            node_idx: Node to score
            target_idx: Target node
            current_idx: Current node (for path-based scoring)
            path_cost: Cost of the path so far
            
        Returns:
            float: Node score
        """
        # Get GNN embeddings
        node_emb = self.get_node_embedding(node_idx)
        target_emb = self.get_node_embedding(target_idx)
        
        # Get GNN similarity (cosine similarity)
        gnn_sim = F.cosine_similarity(node_emb.unsqueeze(0), target_emb.unsqueeze(0)).item()
        
        # Calculate Word2Vec semantic similarity
        w2v_sim = 0.0
        if self.use_word2vec_similarity:
            w2v_sim = self.get_word2vec_similarity(node_idx, target_idx)
        
        # If we have a current node, consider path continuity
        path_continuity = 0.0
        if current_idx is not None:
            current_emb = self.get_node_embedding(current_idx)
            node_emb = self.get_node_embedding(node_idx)
            
            # Path continuity score (cosine similarity with current node)
            path_continuity = F.cosine_similarity(
                current_emb.unsqueeze(0), node_emb.unsqueeze(0)
            ).item() * 0.3  # Lower weight for continuity
        
        # Combine scores
        gnn_weight = 0.6  # GNN embedding weight
        w2v_weight = 0.4  # Word2Vec semantic weight
        
        # If Word2Vec is not available, use only GNN similarity
        if not self.use_word2vec_similarity:
            gnn_weight = 1.0
            w2v_weight = 0.0
        
        # Final score combines similarity and path cost
        similarity = (gnn_sim * gnn_weight) + (w2v_sim * w2v_weight) + path_continuity
        
        # A* formula: path_cost + heuristic * weight
        score = path_cost + (1.0 - similarity) * self.heuristic_weight
        
        return score
    
    def bidirectional_semantic_search(self, start_idx, target_idx, max_steps=30):
        """
        Bidirectional search guided by both GNN embeddings and Word2Vec semantics.
        
        Args:
            start_idx: Starting node index
            target_idx: Target node index
            max_steps: Maximum number of steps to take
            
        Returns:
            tuple: (path, nodes_explored)
        """
        if start_idx == target_idx:
            return [start_idx], 1
        
        # Reset exploration counter
        self.reset_exploration_counter()
        
        # Initialize forward and backward search queues
        # Format: (score, cost, node_idx, path)
        forward_queue = [(0, 0, start_idx, [start_idx])]
        backward_queue = [(0, 0, target_idx, [target_idx])]
        
        # Visited sets
        forward_visited = {start_idx: 0}  # node -> cost
        backward_visited = {target_idx: 0}
        
        # Best paths
        forward_paths = {start_idx: [start_idx]}
        backward_paths = {target_idx: [target_idx]}
        
        # For tracking the best meeting point
        best_meeting_node = None
        best_meeting_cost = float('inf')
        
        for step in range(max_steps):
            # Decide which direction to expand based on queue sizes
            if len(forward_queue) <= len(backward_queue) and forward_queue:
                # Expand forward
                _, cost, node, path = heapq.heappop(forward_queue)
                
                # Check if we've found a meeting point
                if node in backward_visited:
                    total_cost = cost + backward_visited[node]
                    if total_cost < best_meeting_cost:
                        best_meeting_node = node
                        best_meeting_cost = total_cost
                
                # Get neighbors
                neighbors = self.sample_neighbors(node)
                
                # Score and expand neighbors
                for neighbor in neighbors:
                    new_cost = cost + 1
                    
                    # Only consider if we haven't found a better path already
                    if neighbor not in forward_visited or new_cost < forward_visited[neighbor]:
                        # Score this neighbor
                        score = self.score_node(
                            neighbor, target_idx, current_idx=node, path_cost=new_cost
                        )
                        
                        # Update path
                        new_path = path + [neighbor]
                        forward_visited[neighbor] = new_cost
                        forward_paths[neighbor] = new_path
                        
                        # Add to queue
                        heapq.heappush(forward_queue, (score, new_cost, neighbor, new_path))
                        
                        # Check if this is a meeting point
                        if neighbor in backward_visited:
                            total_cost = new_cost + backward_visited[neighbor]
                            if total_cost < best_meeting_cost:
                                best_meeting_node = neighbor
                                best_meeting_cost = total_cost
            
            elif backward_queue:
                # Expand backward
                _, cost, node, path = heapq.heappop(backward_queue)
                
                # Check if we've found a meeting point
                if node in forward_visited:
                    total_cost = cost + forward_visited[node]
                    if total_cost < best_meeting_cost:
                        best_meeting_node = node
                        best_meeting_cost = total_cost
                
                # Get neighbors (for backward search, we need incoming edges)
                # This is a simplification - in a directed graph, we would need to find incoming edges
                neighbors = self.sample_neighbors(node)
                
                # Score and expand neighbors
                for neighbor in neighbors:
                    new_cost = cost + 1
                    
                    # Only consider if we haven't found a better path already
                    if neighbor not in backward_visited or new_cost < backward_visited[neighbor]:
                        # Score this neighbor
                        score = self.score_node(
                            neighbor, start_idx, current_idx=node, path_cost=new_cost
                        )
                        
                        # Update path
                        new_path = path + [neighbor]
                        backward_visited[neighbor] = new_cost
                        backward_paths[neighbor] = new_path
                        
                        # Add to queue
                        heapq.heappush(backward_queue, (score, new_cost, neighbor, new_path))
                        
                        # Check if this is a meeting point
                        if neighbor in forward_visited:
                            total_cost = new_cost + forward_visited[neighbor]
                            if total_cost < best_meeting_cost:
                                best_meeting_node = neighbor
                                best_meeting_cost = total_cost
            
            else:
                # Both queues empty, no path exists
                break
            
            # Early termination if we've found a good meeting point
            if best_meeting_node is not None:
                # Check if we can terminate
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
        
        # No path found
        return [], self.nodes_explored
    
    def semantic_beam_search(self, start_idx, target_idx, max_steps=30):
        """
        Beam search with semantic guidance from Word2Vec and GNN embeddings.
        
        Args:
            start_idx: Starting node index
            target_idx: Target node index
            max_steps: Maximum number of steps to take
            
        Returns:
            tuple: (path, nodes_explored)
        """
        if start_idx == target_idx:
            return [start_idx], 1
        
        # Reset exploration counter
        self.reset_exploration_counter()
        
        # Initialize beam search
        # Format: (score, path)
        current_beams = [(0, [start_idx])]
        visited = {start_idx}
        
        for step in range(max_steps):
            if not current_beams:
                break
            
            # Create next generation of beams
            next_beams = []
            
            # Process current beams
            for _, path in current_beams:
                current_idx = path[-1]
                
                # Check if we've reached the target
                if current_idx == target_idx:
                    return path, self.nodes_explored
                
                # Get neighbors
                neighbors = self.sample_neighbors(current_idx, exclude_nodes=set(path))
                
                # Score and add neighbors
                for neighbor in neighbors:
                    if neighbor not in visited:
                        # Score this neighbor
                        score = self.score_node(
                            neighbor, target_idx, current_idx=current_idx, path_cost=len(path)
                        )
                        
                        # Create new path
                        new_path = path + [neighbor]
                        
                        # Add to candidates
                        next_beams.append((score, new_path))
                        visited.add(neighbor)
                        
                        # Early success
                        if neighbor == target_idx:
                            return new_path, self.nodes_explored
            
            # No candidates found
            if not next_beams:
                break
            
            # Select top-k beams for next iteration
            next_beams.sort(key=lambda x: x[0])  # Sort by score (lower is better for A*)
            current_beams = next_beams[:self.beam_width]
        
        # Return best path found if any
        if current_beams:
            best_path = min(current_beams, key=lambda x: x[0])[1]
            return best_path, self.nodes_explored
        
        return [], self.nodes_explored
    
    def hybrid_semantic_search(self, start_idx, target_idx, max_steps=30):
        """
        Hybrid search combining both bidirectional and beam search with
        semantic guidance from Word2Vec.
        
        Args:
            start_idx: Starting node index
            target_idx: Target node index
            max_steps: Maximum number of steps to take
            
        Returns:
            tuple: (path, nodes_explored)
        """
        # Calculate semantic similarity between start and target
        start_emb = self.get_node_embedding(start_idx)
        target_emb = self.get_node_embedding(target_idx)
        similarity = F.cosine_similarity(start_emb.unsqueeze(0), target_emb.unsqueeze(0)).item()
        
        # Add Word2Vec similarity if available
        if self.use_word2vec_similarity:
            w2v_sim = self.get_word2vec_similarity(start_idx, target_idx)
            similarity = (similarity * 0.6) + (w2v_sim * 0.4)
        
        # Choose search strategy based on similarity
        if similarity > 0.5:  # Nodes are semantically close
            # Try beam search first (faster for closely related nodes)
            path, nodes1 = self.semantic_beam_search(
                start_idx, target_idx, max_steps=max_steps//2
            )
            
            if path and path[-1] == target_idx:
                return path, nodes1
            
            # Fall back to bidirectional search
            path, nodes2 = self.bidirectional_semantic_search(
                start_idx, target_idx, max_steps=max_steps
            )
            
            return path, nodes1 + nodes2
        else:
            # For semantically distant nodes, use bidirectional search first
            path, nodes = self.bidirectional_semantic_search(
                start_idx, target_idx, max_steps=max_steps
            )
            
            return path, nodes
    
    def traverse(self, start_node_id, target_node_id, max_steps=30, method="auto"):
        """
        Main traversal method with multiple search strategies.
        
        Args:
            start_node_id: Starting node ID
            target_node_id: Target node ID
            max_steps: Maximum number of steps to take
            method: Search method to use
                ("auto", "beam", "bidirectional", or "hybrid")
            
        Returns:
            tuple: (path_ids, nodes_explored)
        """
        # Convert IDs to indices
        if start_node_id not in self.data.node_mapping:
            print(f"Error: Start node ID {start_node_id} not found in the graph")
            return [], 0
        
        if target_node_id not in self.data.node_mapping:
            print(f"Error: Target node ID {target_node_id} not found in the graph")
            return [], 0
        
        start_idx = self.data.node_mapping[start_node_id]
        target_idx = self.data.node_mapping[target_node_id]
        
        # Choose search method
        if method == "auto":
            # Calculate similarity to choose method
            start_emb = self.get_node_embedding(start_idx)
            target_emb = self.get_node_embedding(target_idx)
            similarity = F.cosine_similarity(start_emb.unsqueeze(0), target_emb.unsqueeze(0)).item()
            
            # Add Word2Vec similarity if available
            if self.use_word2vec_similarity:
                w2v_sim = self.get_word2vec_similarity(start_idx, target_idx)
                similarity = (similarity * 0.6) + (w2v_sim * 0.4)
            
            # Choose based on similarity
            if similarity > 0.7:
                method = "beam"  # Close nodes: use beam search
            elif similarity > 0.3:
                method = "hybrid"  # Medium distance: use hybrid search
            else:
                method = "bidirectional"  # Far nodes: use bidirectional search
        
        # Execute the selected method
        if method == "beam":
            path, nodes_explored = self.semantic_beam_search(
                start_idx, target_idx, max_steps=max_steps
            )
        elif method == "bidirectional":
            path, nodes_explored = self.bidirectional_semantic_search(
                start_idx, target_idx, max_steps=max_steps
            )
        elif method == "hybrid":
            path, nodes_explored = self.hybrid_semantic_search(
                start_idx, target_idx, max_steps=max_steps
            )
        else:
            print(f"Unknown method: {method}, using hybrid search")
            path, nodes_explored = self.hybrid_semantic_search(
                start_idx, target_idx, max_steps=max_steps
            )
        
        # Convert path indices back to node IDs
        path_ids = [self.data.reverse_mapping[idx] for idx in path]
        
        return path_ids, nodes_explored

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