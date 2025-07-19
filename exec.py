import pandas as pd
import networkx as nx

# Load node and edge data
nodes = pd.read_csv("nodes.csv")
edges = pd.read_csv("edges.csv")

# Build directed graph
G = nx.DiGraph()
for _, row in nodes.iterrows():
    G.add_node(row["id"], label=row["label"])

for _, row in edges.iterrows():
    G.add_edge(row["source"], row["target"], weight=row["weight"])

# Define source and target nodes
source = [n for n, d in G.nodes(data=True) if d["label"] == "MIE"][0]
target = [n for n, d in G.nodes(data=True) if d["label"] == "AO"][0]

# Dynamic Programming: Topological DP path optimization
def dp_optimal_path(G, s, t):
    topo_order = list(nx.topological_sort(G))
    f = {v: float('-inf') for v in G.nodes}
    prev = {v: None for v in G.nodes}
    f[s] = 0

    for v in topo_order:
        for u in G.predecessors(v):
            weight = G[u][v]['weight']
            if f[u] + weight > f[v]:
                f[v] = f[u] + weight
                prev[v] = u

    # Backtrace path
    path = []
    current = t
    while current:
        path.append(current)
        current = prev[current]
    path.reverse()

    return path, f[t]

# Run
path, score = dp_optimal_path(G, source, target)

print("Optimal AOP path:", " -> ".join(path))
print("Cumulative path score:", round(score, 4))
