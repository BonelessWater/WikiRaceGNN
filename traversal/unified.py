import torch
import torch.nn.functional as F
import heapq
import numpy as np
from gensim.utils import simple_preprocess
from abc import ABC, abstractmethod

class TraversalStrategy(ABC):
    """Base strategy interface for traversal algorithms"""
    
    def __init__(self, traverser):
        self.traverser = traverser
        
    @abstractmethod
    def traverse(self, start_idx, target_idx, max_steps):
        """Execute the traversal strategy"""
        pass

class BeamStrategy(TraversalStrategy):
    """Beam search strategy"""
    
    def traverse(self, start_idx, target_idx, max_steps):
        """Beam search with semantic guidance"""
        if start_idx == target_idx:
            return [start_idx], 1
        
        traverser = self.traverser
        traverser.reset_exploration_counter()
        
        # Initialize beam search
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
                
                if current_idx == target_idx:
                    return path, traverser.nodes_explored
                
                # Get neighbors
                neighbors = traverser.sample_neighbors(current_idx, exclude_nodes=set(path))
                
                # Score and add neighbors
                for neighbor in neighbors:
                    if neighbor not in visited:
                        # Score this neighbor
                        score = traverser.score_node(
                            neighbor, target_idx, current_idx=current_idx, path_cost=len(path)
                        )
                        
                        # Create new path
                        new_path = path + [neighbor]
                        
                        # Add to candidates
                        next_beams.append((score, new_path))
                        visited.add(neighbor)
                        
                        # Early success
                        if neighbor == target_idx:
                            return new_path, traverser.nodes_explored
            
            # No candidates found
            if not next_beams:
                break
            
            # Select top-k beams for next iteration
            next_beams.sort(key=lambda x: x[0])  # Sort by score
            current_beams = next_beams[:traverser.beam_width]
        
        # Return best path found if any
        if current_beams:
            best_path = min(current_beams, key=lambda x: x[0])[1]
            return best_path, traverser.nodes_explored
        
        return [], traverser.nodes_explored

class AdaptiveBeamStrategy(TraversalStrategy):
    """Advanced beam search with diversity promotion and exploration penalties"""
    
    def traverse(self, start_idx, target_idx, max_steps):
        """
        Improved beam search with adaptive expansion and path diversity promotion.
        """
        traverser = self.traverser
        
        # Reset counter
        traverser.reset_exploration_counter()
        
        # Handle identical nodes
        if start_idx == target_idx:
            return [start_idx], 1
        
        # Get embeddings
        source_emb = traverser.get_embedding(start_idx)
        target_emb = traverser.get_embedding(target_idx)
        
        # Initialize beam search
        # Format: (score, diversity_score, path)
        visited = {start_idx: 0}  # node -> depth first visited
        current_beams = [(0.0, 0.0, [start_idx])]
        
        # Exploration counter and expansion tracking
        traverser.nodes_explored = 1
        expansions = 0
        
        # Track node visit frequency for diversity promotion
        visit_counts = {start_idx: 1}
        
        for step in range(max_steps):
            if not current_beams or expansions >= traverser.max_expansions:
                break
                
            # Create next generation of beams
            next_beams = []
            
            # Dynamic beam expansion based on similarity to target
            for score, diversity_score, path in current_beams:
                current = path[-1]
                current_emb = traverser.get_embedding(current)
                current_to_target_sim = traverser.get_similarity(current_emb, target_emb)
                
                # Check if we've reached the target
                if current == target_idx:
                    return path, traverser.nodes_explored
                
                # Get neighbors
                neighbors = traverser.data.adj_list.get(current, [])
                
                # Sort neighbors by embedding similarity to target for prioritized expansion
                neighbor_scores = []
                for neighbor in neighbors:
                    # Skip already deeply explored paths
                    if neighbor in visited and visited[neighbor] < len(path) - 2:
                        continue
                    
                    # Get embedding and calculate similarity to target
                    neighbor_emb = traverser.get_embedding(neighbor)
                    similarity = traverser.get_similarity(neighbor_emb, target_emb)
                    
                    # Calculate exploration score with penalties/bonuses
                    # - Penalty for revisiting nodes
                    # - Bonus for nodes that might have been visited at worse depths
                    revisit_penalty = 0
                    if neighbor in visited:
                        revisit_penalty = traverser.exploration_penalty
                        # But if we're revisiting at a better depth, give a bonus
                        if visited[neighbor] > len(path):
                            revisit_penalty = -traverser.revisit_bonus
                    
                    # More penalties for frequently visited nodes to promote diversity
                    frequency_penalty = 0
                    if neighbor in visit_counts:
                        frequency_penalty = (visit_counts[neighbor] / (step + 1)) * traverser.exploration_penalty
                    
                    # Combined score
                    exploration_score = similarity - revisit_penalty - frequency_penalty
                    
                    neighbor_scores.append((neighbor, exploration_score))
                
                # Sort by score and take top candidates
                neighbor_scores.sort(key=lambda x: x[1], reverse=True)
                top_neighbors = neighbor_scores[:min(traverser.beam_width, len(neighbor_scores))]
                
                # Add each candidate to beams
                for neighbor, _ in top_neighbors:
                    # Update visit tracking
                    visit_counts[neighbor] = visit_counts.get(neighbor, 0) + 1
                    visited[neighbor] = min(visited.get(neighbor, float('inf')), len(path))
                    
                    # Create new path
                    new_path = path + [neighbor]
                    
                    # Calculate score components
                    path_length_penalty = len(new_path) * 0.01
                    neighbor_emb = traverser.get_embedding(neighbor)
                    similarity = traverser.get_similarity(neighbor_emb, target_emb)
                    
                    # Path diversity score - measures how different this path is from others
                    # Higher is better for diversity
                    diversity = sum(1 for node in new_path if visit_counts.get(node, 0) <= 1)
                    diversity_score = diversity / len(new_path)
                    
                    # Combined score (lower is better)
                    combined_score = path_length_penalty - similarity
                    
                    # Add to candidates - notice we keep diversity as a separate score
                    next_beams.append((combined_score, diversity_score, new_path))
                    traverser.nodes_explored += 1
                    
                    # Early success detection
                    if neighbor == target_idx:
                        return new_path, traverser.nodes_explored
                    
                # Count this as an expansion
                expansions += 1
                if expansions >= traverser.max_expansions:
                    break
            
            # If no valid expansions, break
            if not next_beams:
                break
                
            # Select next beams with diversity promotion
            # Sort first by score, then by diversity as a tiebreaker
            next_beams.sort(key=lambda x: (x[0], -x[1]))
            
            # Take top beams but ensure some diversity
            diverse_beams = []
            regular_beams = []
            
            for beam in next_beams:
                if beam[1] > 0.5:  # High diversity score
                    diverse_beams.append(beam)
                else:
                    regular_beams.append(beam)
            
            # Ensure at least some proportion of diverse paths in the mix
            diverse_count = min(traverser.beam_width // 3, len(diverse_beams))
            regular_count = traverser.beam_width - diverse_count
            
            current_beams = diverse_beams[:diverse_count] + regular_beams[:regular_count]
        
        # Return best path found
        if current_beams:
            best_beam = min(current_beams, key=lambda x: x[0])
            return best_beam[2], traverser.nodes_explored
        
        # No path found
        return [], traverser.nodes_explored

class BidirectionalStrategy(TraversalStrategy):
    """Bidirectional search strategy"""
    
    def traverse(self, start_idx, target_idx, max_steps):
        """Bidirectional search with embeddings guidance"""
        if start_idx == target_idx:
            return [start_idx], 1
        
        traverser = self.traverser
        traverser.reset_exploration_counter()
        
        # Initialize forward and backward search queues
        forward_queue = [(0, 0, start_idx, [start_idx])]
        backward_queue = [(0, 0, target_idx, [target_idx])]
        
        # Visited sets
        forward_visited = {start_idx: 0}
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
                neighbors = traverser.sample_neighbors(node)
                
                # Score and expand neighbors
                for neighbor in neighbors:
                    new_cost = cost + 1
                    
                    # Only consider if we haven't found a better path already
                    if neighbor not in forward_visited or new_cost < forward_visited[neighbor]:
                        # Score this neighbor
                        score = traverser.score_node(
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
                
                # Get neighbors
                neighbors = traverser.sample_neighbors(node)
                
                # Score and expand neighbors
                for neighbor in neighbors:
                    new_cost = cost + 1
                    
                    # Only consider if we haven't found a better path already
                    if neighbor not in backward_visited or new_cost < backward_visited[neighbor]:
                        # Score this neighbor
                        score = traverser.score_node(
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
            
            return full_path, traverser.nodes_explored
        
        # No path found
        return [], traverser.nodes_explored

class HybridStrategy(TraversalStrategy):
    """Hybrid search combining beam search and bidirectional search"""
    
    def traverse(self, start_idx, target_idx, max_steps):
        """Hybrid search combining both strategies based on similarity"""
        traverser = self.traverser
        
        # Calculate similarity to determine best approach
        start_emb = traverser.get_embedding(start_idx)
        target_emb = traverser.get_embedding(target_idx)
        similarity = F.cosine_similarity(start_emb.unsqueeze(0), target_emb.unsqueeze(0)).item()
        
        # Add Word2Vec similarity if available
        if traverser.use_word2vec_similarity:
            w2v_sim = traverser.get_word2vec_similarity(start_idx, target_idx)
            similarity = (similarity * 0.6) + (w2v_sim * 0.4)
        
        # Choose search strategy based on similarity
        if similarity > 0.5:  # Nodes are semantically close
            # Try beam search first (faster for closely related nodes)
            if hasattr(traverser, 'use_adaptive_beam') and traverser.use_adaptive_beam:
                beam_strategy = AdaptiveBeamStrategy(traverser)
            else:
                beam_strategy = BeamStrategy(traverser)
                
            path, nodes1 = beam_strategy.traverse(
                start_idx, target_idx, max_steps=max_steps//2
            )
            
            if path and path[-1] == target_idx:
                return path, nodes1
            
            # Fall back to bidirectional search
            bi_strategy = BidirectionalStrategy(traverser)
            path, nodes2 = bi_strategy.traverse(
                start_idx, target_idx, max_steps=max_steps
            )
            
            return path, nodes1 + nodes2
        else:
            # For semantically distant nodes, use bidirectional search
            bi_strategy = BidirectionalStrategy(traverser)
            path, nodes = bi_strategy.traverse(
                start_idx, target_idx, max_steps=max_steps
            )
            
            return path, nodes

class GraphTraverser:
    """
    Unified graph traverser with multiple search strategies.
    """
    
    def __init__(self, model, data, device, beam_width=5, heuristic_weight=1.5, 
                 max_memory_nodes=1000, num_neighbors=20, num_hops=2,
                 exploration_penalty=0.1, revisit_bonus=0.05, max_expansions=50,
                 use_adaptive_beam=False):
        self.model = model
        self.data = data
        self.device = device
        self.beam_width = beam_width
        self.heuristic_weight = heuristic_weight
        self.max_memory_nodes = max_memory_nodes
        self.num_neighbors = num_neighbors
        self.num_hops = num_hops
        
        # Advanced beam search parameters
        self.exploration_penalty = exploration_penalty
        self.revisit_bonus = revisit_bonus
        self.max_expansions = max_expansions
        self.use_adaptive_beam = use_adaptive_beam
        
        # State tracking
        self.cache = {}  # Cache for node embeddings
        self.nodes_explored = 0
        
        # Get Word2Vec model from the GNN model if available
        self.word2vec_model = getattr(model, 'word2vec_model', None)
        
        # Check if node titles are available
        self.has_titles = hasattr(data, 'node_titles') and data.node_titles
        
        # Set semantic similarity type
        self.use_word2vec_similarity = self.word2vec_model is not None and self.has_titles
        
        # Create strategies
        self.strategies = {
            'beam': BeamStrategy(self),
            'adaptive_beam': AdaptiveBeamStrategy(self),
            'bidirectional': BidirectionalStrategy(self),
            'hybrid': HybridStrategy(self)
        }
        
        # Ensure model is in eval mode
        self.model.eval()
        
        if self.use_word2vec_similarity:
            print("Using Word2Vec semantic similarity for traversal")
        else:
            print("Word2Vec not available, using only GNN embeddings")
    
    def reset_exploration_counter(self):
        """Reset the counter for nodes explored during traversal"""
        self.nodes_explored = 0
        
    def get_embedding(self, node_idx):
        """Get the embedding for a single node with caching."""
        if node_idx in self.cache:
            return self.cache[node_idx]
        
        with torch.no_grad():
            # Create a single-node feature matrix
            node_x = self.data.x[node_idx].unsqueeze(0).to(self.device)
            
            # Create a batch indicator
            batch = torch.zeros(1, dtype=torch.long, device=self.device)
            
            # Get node neighbors for subgraph
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
    
    def get_similarity(self, emb1, emb2):
        """Calculate cosine similarity between two embeddings"""
        return F.cosine_similarity(emb1, emb2, dim=0).item()
    
    def get_word2vec_similarity(self, node1_idx, node2_idx):
        """Calculate semantic similarity between two nodes using Word2Vec."""
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
        """Sample neighbors for a node."""
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
        """Score a node based on similarity to target and path cost."""
        # Get GNN embeddings
        node_emb = self.get_embedding(node_idx)
        target_emb = self.get_embedding(target_idx)
        
        # Get GNN similarity (cosine similarity)
        gnn_sim = F.cosine_similarity(node_emb.unsqueeze(0), target_emb.unsqueeze(0)).item()
        
        # Calculate Word2Vec semantic similarity
        w2v_sim = 0.0
        if self.use_word2vec_similarity:
            w2v_sim = self.get_word2vec_similarity(node_idx, target_idx)
        
        # If we have a current node, consider path continuity
        path_continuity = 0.0
        if current_idx is not None:
            current_emb = self.get_embedding(current_idx)
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
    
    def select_strategy(self, start_idx, target_idx):
        """Auto-select the best strategy based on node similarity."""
        # Calculate similarity
        start_emb = self.get_embedding(start_idx)
        target_emb = self.get_embedding(target_idx)
        similarity = F.cosine_similarity(start_emb.unsqueeze(0), target_emb.unsqueeze(0)).item()
        
        # Add Word2Vec similarity if available
        if self.use_word2vec_similarity:
            w2v_sim = self.get_word2vec_similarity(start_idx, target_idx)
            similarity = (similarity * 0.6) + (w2v_sim * 0.4)
        
        # Choose based on similarity
        if similarity > 0.7:
            # For very similar nodes, use appropriate beam search
            return 'adaptive_beam' if self.use_adaptive_beam else 'beam'
        elif similarity > 0.3:
            return 'hybrid'  # Medium distance: use hybrid search
        else:
            return 'bidirectional'  # Far nodes: use bidirectional search
    
    def traverse(self, start_node_id, target_node_id, max_steps=30, method="auto"):
        """Main traversal method that selects appropriate strategy."""
        # Reset exploration counter
        self.reset_exploration_counter()
        
        # Convert IDs to indices
        if start_node_id not in self.data.node_mapping:
            print(f"Error: Start node ID {start_node_id} not found in the graph")
            return [], 0
        
        if target_node_id not in self.data.node_mapping:
            print(f"Error: Target node ID {target_node_id} not found in the graph")
            return [], 0
        
        start_idx = self.data.node_mapping[start_node_id]
        target_idx = self.data.node_mapping[target_node_id]
        
        # Map methods to strategies
        strategy_mapping = {
            'beam': 'beam',
            'adaptive_beam': 'adaptive_beam',
            'parallel_beam': 'adaptive_beam' if self.use_adaptive_beam else 'beam',
            'bidirectional': 'bidirectional',
            'bidirectional_guided': 'bidirectional',
            'hybrid': 'hybrid'
        }
        
        # Auto-select method if needed
        if method == "auto":
            method = self.select_strategy(start_idx, target_idx)
        
        # Map to strategy name
        strategy_name = strategy_mapping.get(method, 'hybrid')
        
        # Get and execute strategy
        strategy = self.strategies.get(strategy_name, self.strategies['hybrid'])
        path, nodes_explored = strategy.traverse(start_idx, target_idx, max_steps)
        
        # Convert path indices back to node IDs
        path_ids = [self.data.reverse_mapping[idx] for idx in path]
        
        return path_ids, nodes_explored


class AdvancedGraphTraverser(GraphTraverser):
    """
    Enhanced version of GraphTraverser that uses adaptive beam search by default.
    This is a drop-in replacement for the original GraphNeuralNetworkTraverser.
    """
    
    def __init__(self, model, data, device, beam_width=3, heuristic_weight=1.2, 
                 exploration_penalty=0.1, revisit_bonus=0.05, max_expansions=50,
                 max_memory_nodes=1000, num_neighbors=20, num_hops=2):
        super().__init__(
            model, data, device, 
            beam_width=beam_width,
            heuristic_weight=heuristic_weight,
            max_memory_nodes=max_memory_nodes,
            num_neighbors=num_neighbors,
            num_hops=num_hops,
            exploration_penalty=exploration_penalty,
            revisit_bonus=revisit_bonus,
            max_expansions=max_expansions,
            use_adaptive_beam=True
        )

# For backwards compatibility
GraphNeuralNetworkTraverser = AdvancedGraphTraverser