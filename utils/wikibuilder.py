import requests
import os
import csv
import time
import json
import numpy as np
import concurrent.futures
from collections import deque
from tqdm import tqdm

# For Word2Vec embeddings
try:
    from gensim.models import Word2Vec
    from gensim.models.phrases import Phrases, Phraser
    from gensim.utils import simple_preprocess
    GENSIM_AVAILABLE = True
except ImportError:
    print("Warning: Gensim not available. Install with: pip install gensim")
    GENSIM_AVAILABLE = False

class WikiGraphBuilder:
    """
    A parallelizable Wikipedia graph dataset builder that uses Word2Vec for embeddings.
    
    Creates edge lists and node metadata from Wikipedia pages
    with embeddings based on page titles created using Word2Vec.
    """
    
    def __init__(self, 
                 seed_pages=["Computer_science"],
                 max_nodes=10000,
                 batch_size=100,
                 n_workers=8,
                 output_dir="wiki_graph_data",
                 embedding_dim=64,
                 use_word2vec=True):
        """
        Initialize the graph builder.
        
        Args:
            seed_pages: List of Wikipedia page titles to start crawling from
            max_nodes: Maximum number of nodes in the graph
            batch_size: Number of pages to process in parallel
            n_workers: Number of worker threads/processes
            output_dir: Directory to save output files
            embedding_dim: Dimension for embeddings
            use_word2vec: Whether to use Word2Vec for embeddings
        """
        self.seed_pages = seed_pages
        self.max_nodes = max_nodes
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.output_dir = output_dir
        self.embedding_dim = embedding_dim
        self.use_word2vec = use_word2vec and GENSIM_AVAILABLE
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # State tracking
        self.visited = set()
        self.queue = deque()
        self.edges = []
        self.node_metadata = {}
        self.session = requests.Session()
        
        # For Word2Vec
        self.page_titles = []
        self.word2vec_model = None
        
        # Rate limiting
        self.requests_per_second = 5
        self.last_request_time = 0
        
        print(f"Initialized WikiGraphBuilder with {len(seed_pages)} seed pages")
        if self.use_word2vec:
            print(f"Using Word2Vec for embeddings with dimension {embedding_dim}")
        else:
            print(f"Using random embeddings with dimension {embedding_dim}")
    
    def rate_limit_request(self):
        """Implement simple rate limiting to avoid overloading the API."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        sleep_time = max(0, 1/self.requests_per_second - time_since_last)
        
        if sleep_time > 0:
            time.sleep(sleep_time)
            
        self.last_request_time = time.time()
    
    def fetch_links(self, page_title):
        """
        Fetch outgoing links from a Wikipedia page using the Wikipedia API.
        Retrieve all links but randomly select 5 to reduce density.
        """
        self.rate_limit_request()
        url = "https://en.wikipedia.org/w/api.php"
        
        # Clean up the page title
        page_title = page_title.replace(" ", "_")
        
        params = {
            "action": "query",
            "format": "json",
            "titles": page_title,
            "prop": "links",
            "pllimit": "500"  # Request a higher number of links
        }
        
        all_links = []
        continue_params = {}
        
        try:
            while True:
                # Add continue parameters if we have them
                if continue_params:
                    params.update(continue_params)
                
                response = self.session.get(url=url, params=params, timeout=10)
                response.raise_for_status()  # Raise exception for HTTP errors
                data = response.json()
                
                # Extract links from response
                if "query" in data and "pages" in data["query"]:
                    pages = data["query"]["pages"]
                    
                    for page_id, page_data in pages.items():
                        if "links" in page_data:
                            for link in page_data["links"]:
                                # Filter out non-main namespace links
                                if ":" not in link["title"] and link["title"] not in self.visited:
                                    all_links.append(link["title"])
                
                # Check if we need to continue
                if "continue" in data and "plcontinue" in data["continue"]:
                    continue_params = {"plcontinue": data["continue"]["plcontinue"]}
                else:
                    break
            
            # Randomly select 5 links if we have more than 5
            import random
            if len(all_links) > 5:
                selected_links = random.sample(all_links, 5)
            else:
                selected_links = all_links
                
            return selected_links
                
        except Exception as e:
            print(f"Error fetching links for {page_title}: {e}")
            return []
        
    def preprocess_title(self, title):
        """
        Preprocess a page title for Word2Vec.
        
        Args:
            title: Wikipedia page title
            
        Returns:
            list: List of tokens
        """
        # Replace underscores with spaces
        title = title.replace("_", " ")
        
        # Tokenize
        return simple_preprocess(title, deacc=True)  # deacc=True removes accents
        
    def train_word2vec_model(self):
        """
        Train a Word2Vec model on the page titles.
        """
        if not self.use_word2vec or not GENSIM_AVAILABLE:
            print("Word2Vec not available or not enabled")
            return
        
        # Preprocess page titles
        print("Preprocessing page titles for Word2Vec...")
        preprocessed_titles = [self.preprocess_title(title) for title in self.page_titles]
        
        # Train Phrases model (for bigrams)
        print("Building phrases (bigrams)...")
        phrases = Phrases(preprocessed_titles, min_count=5, threshold=10)
        bigram = Phraser(phrases)
        
        # Apply bigram model
        preprocessed_titles = [bigram[title] for title in preprocessed_titles]
        
        # Train Word2Vec model
        print("Training Word2Vec model...")
        self.word2vec_model = Word2Vec(
            sentences=preprocessed_titles,
            vector_size=self.embedding_dim,
            window=5,
            min_count=1,  # We want embeddings for all words in titles
            workers=self.n_workers,
            sg=1,  # Skip-gram model
            epochs=10
        )
        
        print(f"Word2Vec model trained with {len(self.word2vec_model.wv)} word vectors")
    
    def get_embedding(self, title):
        """
        Get embedding for a page title.
        
        Args:
            title: Page title
            
        Returns:
            np.ndarray: Embedding vector
        """
        if self.use_word2vec and GENSIM_AVAILABLE and self.word2vec_model is not None:
            # Preprocess title
            tokens = self.preprocess_title(title)
            
            if not tokens:
                return np.random.uniform(-1, 1, self.embedding_dim)
            
            # Get embeddings for each token
            token_embeddings = []
            for token in tokens:
                if token in self.word2vec_model.wv:
                    token_embeddings.append(self.word2vec_model.wv[token])
            
            # Average token embeddings
            if token_embeddings:
                return np.mean(token_embeddings, axis=0)
            else:
                return np.random.uniform(-1, 1, self.embedding_dim)
        else:
            # Random embeddings if Word2Vec not available
            return np.random.uniform(-1, 1, self.embedding_dim)
    
    def process_page_batch(self, batch):
        """
        Process a batch of pages, with controls to limit density.
        """
        results = []
        
        # Process each page in the batch
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_page = {executor.submit(self.fetch_links, page): page for page in batch}
            
            for future in concurrent.futures.as_completed(future_to_page):
                source_page = future_to_page[future]
                try:
                    target_pages = future.result()
                    results.append((source_page, target_pages))
                except Exception as e:
                    print(f"Error processing {source_page}: {e}")
        
        # Process results to generate edges and metadata
        batch_edges = []
        batch_metadata = {}
        
        # Track connections per node to limit density
        connections_per_node = {}
        
        for source_page, target_pages in results:
            # Skip if no outgoing links
            if not target_pages:
                continue
            
            # Add source page metadata if not already present
            if source_page not in self.node_metadata:
                source_url = f"https://en.wikipedia.org/wiki/{source_page.replace(' ', '_')}"
                
                # Track page title for Word2Vec training
                if source_page not in self.page_titles:
                    self.page_titles.append(source_page)
                
                batch_metadata[source_page] = {
                    "title": source_page,
                    "url": source_url,
                    "embedding": None  # Placeholder, will be updated later
                }
            
            # Initialize connection counter for source if not exists
            if source_page not in connections_per_node:
                connections_per_node[source_page] = 0
                
            # Limit connections per source node
            max_connections = 5  # Maximum outgoing connections per node
            
            # Process target pages
            for target_page in target_pages:
                # Skip self-loops and limit connections
                if target_page != source_page and connections_per_node[source_page] < max_connections:
                    # Add edge
                    batch_edges.append((source_page, target_page))
                    connections_per_node[source_page] = connections_per_node.get(source_page, 0) + 1
                    
                    # Add target page metadata if not already present
                    if target_page not in self.node_metadata and target_page not in batch_metadata:
                        target_url = f"https://en.wikipedia.org/wiki/{target_page.replace(' ', '_')}"
                        
                        # Track page title for Word2Vec training
                        if target_page not in self.page_titles:
                            self.page_titles.append(target_page)
                        
                        batch_metadata[target_page] = {
                            "title": target_page,
                            "url": target_url,
                            "embedding": None  # Placeholder, will be updated later
                        }
        
        return batch_edges, batch_metadata
    
    def update_embeddings(self):
        """
        Update all node embeddings using Word2Vec.
        """
        if self.use_word2vec and GENSIM_AVAILABLE:
            # Train Word2Vec model if not already trained
            if self.word2vec_model is None:
                self.train_word2vec_model()
            
            # Update embeddings for all nodes
            print("Updating embeddings for all nodes...")
            for title in tqdm(self.node_metadata.keys()):
                self.node_metadata[title]["embedding"] = self.get_embedding(title).tolist()
        else:
            # Use random embeddings
            print("Using random embeddings for all nodes...")
            for title in tqdm(self.node_metadata.keys()):
                self.node_metadata[title]["embedding"] = np.random.uniform(-1, 1, self.embedding_dim).tolist()
    
    def save_progress(self):
        """Save current graph data to disk."""
        # Save edges to CSV
        edge_file = os.path.join(self.output_dir, "wiki_edges.csv")
        with open(edge_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id1', 'id2'])
            for source, target in self.edges:
                writer.writerow([source, target])
        
        # Save node metadata
        metadata_file = os.path.join(self.output_dir, "wiki_nodes.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.node_metadata, f, ensure_ascii=False, indent=2)
        
        # Save node IDs mapping
        id_mapping_file = os.path.join(self.output_dir, "wiki_id_mapping.csv")
        with open(id_mapping_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'title', 'url'])
            for i, node_id in enumerate(self.node_metadata.keys()):
                node = self.node_metadata[node_id]
                writer.writerow([i, node['title'], node['url']])
        
        # Save embeddings
        embeddings_file = os.path.join(self.output_dir, "wiki_embeddings.npy")
        embeddings = np.array([node['embedding'] for node in self.node_metadata.values()])
        np.save(embeddings_file, embeddings)
        
        print(f"Saved {len(self.edges)} edges and {len(self.node_metadata)} nodes")
        
        # Save Word2Vec model if it exists
        if self.word2vec_model is not None:
            model_file = os.path.join(self.output_dir, "word2vec_model")
            self.word2vec_model.save(model_file)
            print(f"Saved Word2Vec model to {model_file}")
    
    def build_graph(self):
        """Build the Wikipedia graph with bidirectional edges."""
        print(f"Starting graph construction with {len(self.seed_pages)} seed pages")
        
        # Initialize with seed pages
        for page in self.seed_pages:
            if page not in self.visited:
                self.queue.append(page)
                self.visited.add(page)
                self.page_titles.append(page)  # Track for Word2Vec
        
        # Process pages in batches
        with tqdm(total=self.max_nodes) as pbar:
            pbar.update(len(self.visited))
            
            while self.queue and len(self.visited) < self.max_nodes:
                # Get next batch of pages
                batch = []
                for _ in range(min(self.batch_size, len(self.queue))):
                    if self.queue:
                        batch.append(self.queue.popleft())
                
                if not batch:
                    break
                
                # Process batch
                batch_edges, batch_metadata = self.process_page_batch(batch)
                
                # Update edges and metadata
                self.edges.extend(batch_edges)
                self.node_metadata.update(batch_metadata)
                
                # Update queue with new pages
                new_pages = 0
                for _, target in batch_edges:
                    if target not in self.visited and len(self.visited) < self.max_nodes:
                        self.queue.append(target)
                        self.visited.add(target)
                        new_pages += 1
                
                # Update progress bar
                pbar.update(new_pages)
        
        # Make edges bidirectional
        print("Ensuring edges are bidirectional...")
        bidirectional_edges = set()
        for source, target in self.edges:
            bidirectional_edges.add((source, target))
            bidirectional_edges.add((target, source))  # Add the reverse edge
        
        # Convert back to list
        self.edges = list(bidirectional_edges)
        
        # Train Word2Vec model and update embeddings
        if self.use_word2vec and GENSIM_AVAILABLE:
            self.train_word2vec_model()
            self.update_embeddings()
        else:
            # Use random embeddings
            self.update_embeddings()
        
        # Save final results
        self.save_progress()
        
        # Create adjacency list for faster lookups
        print("Creating adjacency list...")
        adj_list = {}
        for source, target in self.edges:
            if source not in adj_list:
                adj_list[source] = []
            adj_list[source].append(target)
        
        # Save adjacency list
        adj_list_file = os.path.join(self.output_dir, "wiki_adjacency_list.json")
        with open(adj_list_file, 'w', encoding='utf-8') as f:
            json.dump(adj_list, f, ensure_ascii=False, indent=2)
        
        print(f"Graph construction complete! Created graph with {len(self.node_metadata)} nodes and {len(self.edges)} edges")
        return self.edges, self.node_metadata
    
    def convert_to_pytorch_geometric(self):
        """Convert the graph to PyTorch Geometric format with bidirectional edges."""
        try:
            import torch
            from torch_geometric.data import Data
            from torch_geometric.utils import to_undirected
            
            # Create node mapping
            node_mapping = {node: i for i, node in enumerate(self.node_metadata.keys())}
            
            # Create edge index tensor
            edge_index = torch.tensor([[node_mapping[source], node_mapping[target]] 
                                    for source, target in self.edges 
                                    if source in node_mapping and target in node_mapping], 
                                    dtype=torch.long).t()
            
            # Ensure edges are undirected/bidirectional
            edge_index = to_undirected(edge_index)
            
            # Create node features tensor
            x = torch.tensor([self.node_metadata[node]['embedding'] for node in node_mapping.keys()], 
                            dtype=torch.float)
            
            # Create data object
            data = Data(x=x, edge_index=edge_index)
            
            # Add mappings for reference
            data.node_mapping = node_mapping
            data.reverse_mapping = {i: node for node, i in node_mapping.items()}
            
            # Create adjacency list for the data object
            data.adj_list = {}
            for i in range(edge_index.size(1)):
                source = edge_index[0, i].item()
                target = edge_index[1, i].item()
                
                if source not in data.adj_list:
                    data.adj_list[source] = []
                
                data.adj_list[source].append(target)
            
            # Save to file
            torch.save(data, os.path.join(self.output_dir, "wiki_graph.pt"))
            print(f"Saved PyTorch Geometric data object with {data.num_nodes} nodes and {data.num_edges} edges")
            
            return data
        except ImportError:
            print("Could not import torch_geometric. PyTorch Geometric format will not be saved.")
            return None
        except Exception as e:
            print(f"Error converting to PyTorch Geometric format: {e}")
            return None

def create_wiki_edge_list(output_dir="data", max_nodes=5000, use_word2vec=True):
    """
    Main function to create a wiki edge list suitable for the WikiRaceGNN project.
    
    Args:
        output_dir: Directory to save the data
        max_nodes: Maximum number of nodes to collect
        use_word2vec: Whether to use Word2Vec for embeddings
        
    Returns:
        Path to the created edge list
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    seed_pages = [
        "Computer_science"
    ]
    
    # Create graph builder
    builder = WikiGraphBuilder(
        seed_pages=seed_pages,
        max_nodes=max_nodes,
        batch_size=50,
        n_workers=8,
        output_dir=output_dir,
        embedding_dim=64,  # Match feature_dim in WikiRaceGNN's load_graph_data
        use_word2vec=use_word2vec
    )
    
    # Build the graph
    builder.build_graph()
    
    # Return the path to the edge list
    edge_list_path = os.path.join(output_dir, "wiki_edges.csv")
    return edge_list_path
