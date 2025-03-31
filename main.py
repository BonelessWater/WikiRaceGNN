import requests
import random
from neo4j import GraphDatabase
from collections import deque

class WikiGraphCrawler:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(WikiGraphCrawler, cls).__new__(cls)
        return cls._instance

    def __init__(self, 
                 neo4j_uri="bolt://localhost:7687", 
                 neo4j_username="neo4j", 
                 neo4j_password="rootroot", 
                 seed_page="Nuisance", 
                 max_nodes=10000, 
                 batch_size=100):
        # Prevent reinitialization in subsequent calls
        if hasattr(self, "_initialized") and self._initialized:
            return
        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        self.seed_page = seed_page
        self.max_nodes = max_nodes
        self.batch_size = batch_size

        self.driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_username, self.neo4j_password))
        self._initialized = True

    @staticmethod
    def insert_relationships(tx, relationships):
        query = """
        UNWIND $relationships as rel
        MERGE (a:WikipediaPage {title: rel.source})
          ON CREATE SET a.embedding = rel.source_embedding, 
                        a.url = rel.source_url
        MERGE (b:WikipediaPage {title: rel.target})
          ON CREATE SET b.embedding = rel.target_embedding, 
                        b.url = rel.target_url
        MERGE (a)-[:LINKS_TO]->(b)
        """
        tx.run(query, relationships=relationships)

    def fetch_links(self, page_title):
        """Fetch all outgoing links from a Wikipedia page using the Wikipedia API."""
        session_requests = requests.Session()
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "titles": page_title,
            "prop": "links",
            "pllimit": "max"
        }
        links = []
        while True:
            response = session_requests.get(url=url, params=params)
            data = response.json()
            pages = data["query"]["pages"]
            for key, val in pages.items():
                if "links" in val:
                    for link in val["links"]:
                        links.append(link["title"])
            if "continue" in data:
                params["plcontinue"] = data["continue"]["plcontinue"]
            else:
                break
        return links

    @staticmethod
    def random_embedding():
        """Return a list of 384 random floats in the range [-1, 1]."""
        return [random.uniform(-1, 1) for _ in range(384)]

    def crawl(self):
        visited = set()      # Track pages we've visited
        queue = deque()      # Pages to process

        visited.add(self.seed_page)
        queue.append(self.seed_page)

        relationship_batch = []

        while queue and len(visited) < self.max_nodes:
            current_page = queue.popleft()
            print(f"Processing: {current_page} (Total visited: {len(visited)})")
            try:
                outgoing_links = self.fetch_links(current_page)
            except Exception as e:
                print(f"Error fetching links for {current_page}: {e}")
                continue

            for target_title in outgoing_links:
                rel = {
                    "source": current_page,
                    "source_url": f"https://en.wikipedia.org/wiki/{current_page.replace(' ', '_')}",
                    "source_embedding": self.random_embedding(),
                    "target": target_title,
                    "target_url": f"https://en.wikipedia.org/wiki/{target_title.replace(' ', '_')}",
                    "target_embedding": self.random_embedding()
                }
                relationship_batch.append(rel)
                if target_title not in visited and len(visited) < self.max_nodes:
                    visited.add(target_title)
                    queue.append(target_title)
            if len(relationship_batch) >= self.batch_size:
                with self.driver.session() as session:
                    session.execute_write(WikiGraphCrawler.insert_relationships, relationship_batch)
                relationship_batch = []

        if relationship_batch:
            with self.driver.session() as session:
                session.execute_write(WikiGraphCrawler.insert_relationships, relationship_batch)

        self.driver.close()
        print("Crawling complete. Total pages visited:", len(visited))


if __name__ == "__main__":
    crawler = WikiGraphCrawler()
    crawler.crawl()
