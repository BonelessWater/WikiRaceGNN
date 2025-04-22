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
    def traverse(self, start_idx, target_idx, max_steps):
        # clear any previous history
        self.traverser.history.clear()

        if start_idx == target_idx:
            return [start_idx], 1
        
        traverser = self.traverser
        traverser.reset_exploration_counter()
        
        current_beams = [(0, [start_idx])]
        visited = {start_idx}
        
        for step in range(max_steps):
            if not current_beams:
                break
            next_beams = []
            for _, path in current_beams:
                node = path[-1]
                if node == target_idx:
                    return path, traverser.nodes_explored
                for nbr in traverser.sample_neighbors(node, exclude_nodes=set(path)):
                    if nbr not in visited:
                        score = traverser.score_node(nbr, target_idx,
                                                     current_idx=node,
                                                     path_cost=len(path))
                        new_path = path + [nbr]
                        next_beams.append((score, new_path))
                        visited.add(nbr)
                        if nbr == target_idx:
                            return new_path, traverser.nodes_explored
            if not next_beams:
                break
            next_beams.sort(key=lambda x: x[0])
            current_beams = next_beams[:traverser.beam_width]
            
            # record just the “tails” of each beam as our forward frontier
            beam_frontier = {path[-1] for _, path in current_beams}
            self.traverser.history.append((beam_frontier, None, step))

        if current_beams:
            best = min(current_beams, key=lambda x: x[0])[1]
            return best, traverser.nodes_explored
        return [], traverser.nodes_explored

class AdaptiveBeamStrategy(TraversalStrategy):

    def traverse(self, start_idx, target_idx, max_steps):
        traverser = self.traverser
        traverser.reset_exploration_counter()
        traverser.history.clear()

        if start_idx == target_idx:
            return [start_idx], 1

        source_emb = traverser.get_embedding(start_idx)
        target_emb = traverser.get_embedding(target_idx)
        visited = {start_idx: 0}
        current_beams = [(0.0, 0.0, [start_idx])]
        traverser.nodes_explored = 1
        expansions = 0
        visit_counts = {start_idx: 1}

        for step in range(max_steps):
            if not current_beams or expansions >= traverser.max_expansions:
                break
            next_beams = []
            for score, div_score, path in current_beams:
                node = path[-1]
                if node == target_idx:
                    return path, traverser.nodes_explored

                nbrs = traverser.data.adj_list.get(node, [])
                nbr_scores = []
                for nbr in nbrs:
                    if nbr in visited and visited[nbr] < len(path) - 2:
                        continue
                    emb = traverser.get_embedding(nbr)
                    sim = traverser.get_similarity(emb, target_emb)
                    revisit = traverser.exploration_penalty if nbr in visited else 0
                    if nbr in visited and visited[nbr] > len(path):
                        revisit = -traverser.revisit_bonus
                    freq_pen = ((visit_counts[nbr]/(step+1))
                               * traverser.exploration_penalty) if nbr in visit_counts else 0
                    nbr_scores.append((nbr, sim - revisit - freq_pen))

                nbr_scores.sort(key=lambda x: x[1], reverse=True)
                top = nbr_scores[:min(traverser.beam_width, len(nbr_scores))]

                for nbr, _ in top:
                    visit_counts[nbr] = visit_counts.get(nbr, 0) + 1
                    visited[nbr] = min(visited.get(nbr, float('inf')), len(path))
                    new_path = path + [nbr]
                    length_pen = len(new_path) * 0.01
                    emb = traverser.get_embedding(nbr)
                    sim = traverser.get_similarity(emb, target_emb)
                    diversity = (sum(1 for x in new_path if visit_counts.get(x,0)<=1)
                                 / len(new_path))
                    combined = length_pen - sim
                    next_beams.append((combined, diversity, new_path))
                    traverser.nodes_explored += 1
                    if nbr == target_idx:
                        return new_path, traverser.nodes_explored

                expansions += 1
                if expansions >= traverser.max_expansions:
                    break

            if not next_beams:
                break

            next_beams.sort(key=lambda x: (x[0], -x[1]))
            diverse = [b for b in next_beams if b[1] > 0.5]
            regular = [b for b in next_beams if b[1] <= 0.5]
            dcount = min(traverser.beam_width // 3, len(diverse))
            rcount = traverser.beam_width - dcount
            current_beams = diverse[:dcount] + regular[:rcount]

            beam_front = {path[-1] for _, _, path in current_beams}
            self.traverser.history.append((beam_front, None, step))

        if current_beams:
            best = min(current_beams, key=lambda x: x[0])[2]
            return best, traverser.nodes_explored
        return [], traverser.nodes_explored

class BidirectionalStrategy(TraversalStrategy):
    def traverse(self, start_idx, target_idx, max_steps):
        self.traverser.history.clear()
        if start_idx == target_idx:
            return [start_idx], 1

        traverser = self.traverser
        traverser.reset_exploration_counter()

        f_q = [(0,0,start_idx,[start_idx])]
        b_q = [(0,0,target_idx,[target_idx])]
        f_vis = {start_idx:0}
        b_vis = {target_idx:0}
        f_paths = {start_idx:[start_idx]}
        b_paths = {target_idx:[target_idx]}
        meet, best_cost = None, float('inf')

        for step in range(max_steps):
            if f_q and (len(f_q) <= len(b_q)):
                _, cost, node, path = heapq.heappop(f_q)
                if node in b_vis:
                    total = cost + b_vis[node]
                    if total < best_cost:
                        meet, best_cost = node, total
                for nbr in traverser.sample_neighbors(node):
                    nc = cost + 1
                    if nbr not in f_vis or nc < f_vis[nbr]:
                        score = traverser.score_node(nbr, target_idx,
                                                     current_idx=node,
                                                     path_cost=nc)
                        new_p = path + [nbr]
                        f_vis[nbr] = nc; f_paths[nbr] = new_p
                        heapq.heappush(f_q,(score, nc, nbr, new_p))
                        if nbr in b_vis:
                            total = nc + b_vis[nbr]
                            if total < best_cost:
                                meet, best_cost = nbr, total

            elif b_q:
                _, cost, node, path = heapq.heappop(b_q)
                if node in f_vis:
                    total = cost + f_vis[node]
                    if total < best_cost:
                        meet, best_cost = node, total
                for nbr in traverser.sample_neighbors(node):
                    nc = cost + 1
                    if nbr not in b_vis or nc < b_vis[nbr]:
                        score = traverser.score_node(nbr, start_idx,
                                                     current_idx=node,
                                                     path_cost=nc)
                        new_p = path + [nbr]
                        b_vis[nbr] = nc; b_paths[nbr] = new_p
                        heapq.heappush(b_q, (score, nc, nbr, new_p))
                        if nbr in f_vis:
                            total = nc + f_vis[nbr]
                            if total < best_cost:
                                meet, best_cost = nbr, total
            else:
                break

            # record both frontiers
            self.traverser.history.append((set(f_vis), set(b_vis), step))

            if meet is not None and ((not f_q or f_q[0][0]>=best_cost)
                                     and (not b_q or b_q[0][0]>=best_cost)):
                break

        if meet is not None:
            fwd = f_paths[meet]
            bwd = b_paths[meet]
            return fwd + bwd[::-1][1:], traverser.nodes_explored

        return [], traverser.nodes_explored

class HybridStrategy(TraversalStrategy):
    def traverse(self, start_idx, target_idx, max_steps):
        self.traverser.history.clear()

        emb1 = self.traverser.get_embedding(start_idx)
        emb2 = self.traverser.get_embedding(target_idx)
        sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        if self.traverser.use_word2vec_similarity:
            w2v = self.traverser.get_word2vec_similarity(start_idx, target_idx)
            sim = 0.6*sim + 0.4*w2v

        if sim > 0.5:
            strat = (AdaptiveBeamStrategy(self.traverser)
                     if self.traverser.use_adaptive_beam
                     else BeamStrategy(self.traverser))
            p1, n1 = strat.traverse(start_idx, target_idx, max_steps//2)
            if p1 and p1[-1] == target_idx:
                return p1, n1
            p2, n2 = BidirectionalStrategy(self.traverser).traverse(
                        start_idx, target_idx, max_steps)
            return p2, n1 + n2
        else:
            return BidirectionalStrategy(self.traverser).traverse(
                        start_idx, target_idx, max_steps)

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
        self.history = []        # ← will hold tuples of (forward_frontier, backward_frontier, step)
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
    def get_history(self):
        """Return the recorded (forward, backward, step) tuples."""
        return self.history
    
    def get_traversal_history(self):
        """Return the recorded (forward, backward, step) history"""
        return self.history
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