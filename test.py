#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
from collections import deque
import random
from neo4j import GraphDatabase
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx
import pickle

# -----------------------------------------------------------------------------
# 1) Crawler with bidirectional inserts (unchanged)
# -----------------------------------------------------------------------------
class WikiGraphCrawler:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, neo4j_uri="bolt://localhost:7687",
                 neo4j_username="neo4j", neo4j_password="rootroot",
                 seed_page="Nuisance", max_nodes=100, batch_size=10):
        if getattr(self, "_initialized", False):
            return
        self.driver = GraphDatabase.driver(neo4j_uri,
                                           auth=(neo4j_username, neo4j_password))
        self.seed_page, self.max_nodes, self.batch_size = seed_page, max_nodes, batch_size
        self._initialized = True

    @staticmethod
    def insert_relationships(tx, rels):
        tx.run("""
        UNWIND $rels AS r
        MERGE (a:WikipediaPage {title: r.source})
          ON CREATE SET a.embedding=r.source_embedding, a.url=r.source_url
        MERGE (b:WikipediaPage {title: r.target})
          ON CREATE SET b.embedding=r.target_embedding, b.url=r.target_url
        MERGE (a)-[:LINKS_TO]->(b)
        MERGE (b)-[:LINKS_TO]->(a)
        """, rels=rels)

    def fetch_links(self, title):
        S = requests.Session()
        params = {"action":"query","format":"json",
                  "titles":title,"prop":"links","pllimit":"max"}
        url = "https://en.wikipedia.org/w/api.php"
        out = []
        while True:
            resp = S.get(url, params=params).json()
            for p in resp["query"]["pages"].values():
                out += [l["title"] for l in p.get("links",[])]
            if "continue" in resp:
                params["plcontinue"] = resp["continue"]["plcontinue"]
            else:
                break
        return out

    @staticmethod
    def random_embedding(dim=384):
        return [random.uniform(-1,1) for _ in range(dim)]

    def crawl(self):
        visited, queue, batch = {self.seed_page}, deque([self.seed_page]), []
        while queue and len(visited)<self.max_nodes:
            page = queue.popleft()
            print(f"Processing: {page} (visited={len(visited)})")
            try: outs = self.fetch_links(page)
            except: continue
            for tgt in outs:
                batch.append({
                  "source": page,
                  "source_url": f"https://en.wikipedia.org/wiki/{page.replace(' ','_')}",
                  "source_embedding": self.random_embedding(),
                  "target": tgt,
                  "target_url": f"https://en.wikipedia.org/wiki/{tgt.replace(' ','_')}",
                  "target_embedding": self.random_embedding()
                })
                if tgt not in visited and len(visited)<self.max_nodes:
                    visited.add(tgt); queue.append(tgt)
            if len(batch)>=self.batch_size:
                with self.driver.session() as s:
                    s.write_transaction(self.insert_relationships, batch)
                batch.clear()
        if batch:
            with self.driver.session() as s:
                s.write_transaction(self.insert_relationships, batch)
        self.driver.close()
        print("Crawl complete:", len(visited), "pages")


# -----------------------------------------------------------------------------
# 2) Load undirected edges into PyG
# -----------------------------------------------------------------------------
def load_edges(uri, user, pwd):
    drv = GraphDatabase.driver(uri, auth=(user,pwd))
    with drv.session() as s:
        res = s.run("""
          MATCH (a:WikipediaPage)-[:LINKS_TO]->(b:WikipediaPage)
          RETURN a.title AS src, b.title AS dst
        """)
        edges = [(r["src"],r["dst"]) for r in res]
    drv.close()
    return edges

crawler = WikiGraphCrawler(max_nodes=100, batch_size=10)
crawler.crawl()
edges = load_edges("bolt://localhost:7687","neo4j","rootroot")

titles = sorted({u for u,v in edges}|{v for u,v in edges})
idx    = {t:i for i,t in enumerate(titles)}
src    = [idx[u] for u,v in edges]
dst    = [idx[v] for u,v in edges]
edge_index = torch.tensor([src,dst],dtype=torch.long)
num_nodes  = len(titles)

# -----------------------------------------------------------------------------
# 3) Build next-hop triples & adjacency
# -----------------------------------------------------------------------------
G = nx.Graph(); G.add_edges_from(edges)
triples = []
import random as rd
for s in rd.sample(list(G.nodes()),min(5000,G.number_of_nodes())):
    for t in rd.sample(list(G.nodes()),k=3):
        if s==t: continue
        try: path=nx.shortest_path(G,s,t)
        except: continue
        for a,b in zip(path,path[1:]):
            triples.append((idx[a],idx[t],idx[b]))
adj = {i:[] for i in range(num_nodes)}
for u,v in edges:
    adj[idx[u]].append(idx[v])
# filter out bad triples
triples = [(c,t,n) for (c,t,n) in triples if n in adj[c]]
print("Training samples:", len(triples))

# -----------------------------------------------------------------------------
# 4) Next-Hop GNN
# -----------------------------------------------------------------------------
class NextHopGNN(torch.nn.Module):
    def __init__(self,N,e=128,h=64):
        super().__init__()
        self.emb    = torch.nn.Embedding(N,e)
        self.conv1  = GCNConv(e,h)
        self.conv2  = GCNConv(h,h)
        self.policy = torch.nn.Sequential(
            torch.nn.Linear(h*2,h),
            torch.nn.ReLU(),
            torch.nn.Linear(h,1)
        )
    def forward(self, edge_index):
        x = self.emb.weight
        h = F.relu(self.conv1(x,edge_index))
        return self.conv2(h,edge_index)
    def score(self,h,c,t,adj):
        hc,ht=h[c],h[t]; best,bs=c,float("-inf")
        for j in adj[c]:
            sc=self.policy(torch.cat([hc,ht])) .item()
            if sc>bs: bs, best=sc,j
        return best

data  = Data(edge_index=edge_index)
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NextHopGNN(num_nodes).to(device)
opt   = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)

# -----------------------------------------------------------------------------
# 5) TRAIN: move forward() INSIDE the batch loop
# -----------------------------------------------------------------------------
currs = torch.tensor([c for c,_,_ in triples],dtype=torch.long)
targs = torch.tensor([t for _,t,_ in triples],dtype=torch.long)
nexts = torch.tensor([n for _,_,n in triples],dtype=torch.long)
ds     = torch.utils.data.TensorDataset(currs,targs,nexts)
loader = torch.utils.data.DataLoader(ds,batch_size=512,shuffle=True)

for epoch in range(1,11):
    model.train()
    total_loss=0.0
    for c_batch,t_batch,n_batch in loader:
        c_batch,t_batch,n_batch = [x.to(device) for x in (c_batch,t_batch,n_batch)]
        # RECOMPUTE the full graph embedding each batch
        h_full = model(data.edge_index.to(device))  
        losses = []
        for c,t,n in zip(c_batch.tolist(),t_batch.tolist(),n_batch.tolist()):
            neighs=adj[c]
            if len(neighs)<2 or n not in neighs: 
                continue
            hc,ht = h_full[c], h_full[t]
            inp   = torch.cat([hc,ht]).unsqueeze(0).repeat(len(neighs),1)
            logits= model.policy(inp).squeeze()
            label = neighs.index(n)
            losses.append(
              F.cross_entropy(logits.unsqueeze(0),
                              torch.tensor([label],device=device))
            )
        if not losses:
            continue
        loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()      # now safe: fresh graph
        opt.step()
        total_loss += loss.item()*c_batch.size(0)

    print(f"Epoch {epoch:02d} | Avg Loss: {total_loss/len(ds):.4f}")

# -----------------------------------------------------------------------------
# 6) Precompute & cache embeddings
# -----------------------------------------------------------------------------
model.eval()
with torch.no_grad():
    final_h = model(data.edge_index.to(device)).cpu()
pickle.dump({"titles":titles,"emb":final_h,"adj":adj},
            open("wiki_embeddings_bidirectional.pkl","wb"))
print("✅ embeddings cached")

# -----------------------------------------------------------------------------
# 7) Inference (unchanged)
# -----------------------------------------------------------------------------
def find_path(start,end,max_hops=50):
    D = pickle.load(open("wiki_embeddings_bidirectional.pkl","rb"))
    titles,emb,adj = D["titles"],D["emb"],D["adj"]
    idx_map={t:i for i,t in enumerate(titles)}
    s,t = idx_map[start], idx_map[end]
    path=[s]
    for _ in range(max_hops):
        cur=path[-1]
        if cur==t: break
        nxt=model.score(emb,cur,t,adj)
        if nxt==cur or nxt in path: break
        path.append(nxt)
    return [titles[i] for i in path]

print(find_path("Nuisance","Quantum_mechanics"))
