# Week 5: Network Analysis of User Interactions
# ---------------------------------------------
# Builds a user mention network and identifies top influencers and communities.

import pandas as pd
import re
import networkx as nx
import os
from collections import Counter
import matplotlib.pyplot as plt
import community as community_louvain  # python-louvain

DATA_PATH = "data/features/reddit_features.csv"
OUTPUT_METRICS = "reports/week5_top_pagerank.csv"

os.makedirs("reports", exist_ok=True)

# =============== LOAD DATA ===============
df = pd.read_csv(DATA_PATH)
df['text'] = df['text'].astype(str)

# =============== EXTRACT MENTIONS ===============
def extract_mentions(text):
    return re.findall(r'@([A-Za-z0-9_]+)', text)

edges = []
for _, row in df.iterrows():
    author = str(row.get('author_id', 'unknown'))
    mentions = extract_mentions(row['text'])
    for mention in mentions:
        edges.append((author, mention))

print(f"[INFO] Extracted {len(edges)} edges.")

# =============== BUILD GRAPH ===============
G = nx.DiGraph()
G.add_edges_from(edges)
print(f"[INFO] Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")

# =============== CENTRALITY METRICS ===============
pr = nx.pagerank(G, alpha=0.85)
deg = nx.degree_centrality(G)

top_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:20]
pd.DataFrame(top_pr, columns=["user", "pagerank"]).to_csv(OUTPUT_METRICS, index=False)
print(f"[INFO] Saved PageRank top users → {OUTPUT_METRICS}")

# =============== COMMUNITY DETECTION ===============
G_undirected = G.to_undirected()
partition = community_louvain.best_partition(G_undirected)
num_communities = len(set(partition.values()))
print(f"[INFO] Detected {num_communities} communities.")

# =============== PLOT NETWORK ===============
import matplotlib.pyplot as plt
import networkx as nx

plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G, seed=42)

# give each community a simple color
unique_comms = list(set(partition.values()))
color_map = {comm: i for i, comm in enumerate(unique_comms)}
node_colors = [color_map[partition[n]] for n in G.nodes()]

nx.draw_networkx_nodes(G, pos,
                       node_color=node_colors,
                       cmap=plt.cm.Set3,
                       node_size=500,
                       alpha=0.8)
nx.draw_networkx_edges(G, pos, alpha=0.4)
nx.draw_networkx_labels(G, pos, font_size=8)

plt.title("User Interaction Network")
plt.axis("off")
plt.tight_layout()
plt.savefig("reports/week5_network_graph.png", dpi=300)
plt.show()

print("[INFO] Saved network visualization → reports/week5_network_graph.png")
