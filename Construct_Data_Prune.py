import json
import random
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.subplots as sp

pio.kaleido.scope.mathjax = None

def visualize_scores(go_importance, protein_scores):
    go_scores = list(go_importance.values())
    protein_scores_list = list(protein_scores.values())

    go_bins = np.linspace(0, 5000, 101)
    protein_bins = np.linspace(0, 50, 101)

    go_hist, go_edges = np.histogram(go_scores, bins=go_bins)
    protein_hist, protein_edges = np.histogram(protein_scores_list, bins=protein_bins)

    fig = sp.make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "<b>GO Node Importance Scores</b>",
            "<b>Protein Node Importance Scores</b>"
        ),
        horizontal_spacing=0.1
    )


    fig.add_trace(go.Bar(
        x=go_edges[:-1],
        y=go_hist,
        marker=dict(color='#90C9E6', line=dict(width=1, color='black')),
        showlegend=False
    ), row=1, col=1)


    fig.add_trace(go.Bar(
        x=protein_edges[:-1],
        y=protein_hist,
        marker=dict(color='#DBCA76', line=dict(width=1, color='black')),
        showlegend=False
    ), row=1, col=2)

    fig.update_layout(
        # title=dict(
        #     text="GO and Protein Node Importance Scores",
        #     x=0.5,
        #     font=dict(size=24, family='Arial')
        # ),
        annotations=[
            dict(
                text="<b>GO Node Importance Scores</b>",
                x=0.225, y=1.05, showarrow=False, font=dict(size=20, family='Arial'), xref="paper", yref="paper"
            ),
            dict(
                text="<b>Protein Node Importance Scores</b>",
                x=0.775, y=1.05, showarrow=False, font=dict(size=20, family='Arial'), xref="paper", yref="paper"
            )
        ],
        plot_bgcolor='white',
        xaxis1=dict(
            title="Score",
            titlefont=dict(size=16),
            range=[0, 5000],
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True
        ),
        xaxis2=dict(
            title="Score",
            titlefont=dict(size=16),
            range=[0, 50],
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True
        ),
        yaxis1=dict(
            title="Frequency",
            titlefont=dict(size=16),
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True
        ),
        yaxis2=dict(
            title="Frequency",
            titlefont=dict(size=16),
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True
        ),
        width=1200,
        height=600,
        margin=dict(l=60, r=60, t=80, b=60)
    )

    pio.write_image(fig, "Importance_scores_plotly.pdf", format="pdf")
    return fig


# visualize_scores_plotly(go_importance, protein_scores)


toxin_df = pd.read_excel('./data/toxin_train.xlsx')
toxic_proteins = set(toxin_df['Entry'])

protein2id = {}
id2protein = {}
with open('./data/PSKG/pretrain_data/protein2id_updated.txt', 'r') as f:
    for line in f:
        entry, pid = line.strip().split()
        protein2id[entry] = pid
        id2protein[pid] = entry


protein_seq = {}
with open('./data/PSKG/pretrain_data/protein_seq_map.txt', 'r') as f:
    for line in f:
        entry, seq = line.strip().split()
        protein_seq[entry] = seq

protein_go = []
with open('./data/PSKG/pretrain_data/protein_go_triplet_updated.txt', 'r') as f:
    for line in f:
        h, r, t = line.strip().split()
        protein_go.append((h, r, t))

go_nodes = []
with open('./data/PSKG/pretrain_data/go2id.txt', 'r') as f:
    for line in f:
        GO, id = line.strip().split()
        go_nodes.append(id)


go_go = []
with open('./data/PSKG/pretrain_data/go_go_triplet.txt', 'r') as f:
    for line in f:
        h, r, t = line.strip().split()
        go_go.append((h, r, t))



G = nx.Graph()


for h, r, t in protein_go:
    G.add_edge(h, t, relation=r)


for h, r, t in go_go:
    G.add_edge(h, t, relation=r)

non_toxic_proteins = list(set(protein2id.keys()) - toxic_proteins)

print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

toxic_protein_ids = {protein2id[entry] for entry in toxic_proteins if entry in protein2id}
non_toxic_protein_ids = {protein2id[entry] for entry in non_toxic_proteins if entry in protein2id}

go_importance = {}


for go_node in go_nodes:

    neighbors = set(G.neighbors(go_node))


    toxic_neighbors = neighbors.intersection(toxic_protein_ids)
    non_toxic_neighbors = neighbors.intersection(non_toxic_protein_ids)

    if toxic_neighbors and non_toxic_neighbors:

        bridging_score = len(toxic_neighbors) * len(non_toxic_neighbors)

        go_importance[go_node] = bridging_score + len(non_toxic_neighbors) * 0.5


threshold = sorted(go_importance.values(), reverse=True)[int(len(go_importance) * 0.5)]
important_go_nodes = {go for go, score in go_importance.items() if score >= threshold}

alpha, beta = 0.7, 0.3
protein_scores = {}

nodes_to_keep = set(toxic_protein_ids)

nodes_to_keep.update(important_go_nodes)

for protein in non_toxic_protein_ids:

    high_score_connections = sum(1 for go in G.neighbors(protein) if go in important_go_nodes)

    degree_centrality = G.degree(protein)

    protein_scores[protein] = alpha * high_score_connections + beta * degree_centrality

threshold_score = sorted(protein_scores.values(), reverse=True)[int(len(protein_scores) * 0.5)]
top_proteins = {protein for protein, score in protein_scores.items() if score >= threshold_score}
nodes_to_keep.update(top_proteins)

reduced_G = G.subgraph(nodes_to_keep).copy()

G = reduced_G
print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")


if nx.is_connected(G):
    print("Pruned graph is connected.")
else:
    print("Pruned graph is not connected.")

visualize_scores(go_importance, protein_scores)


def find_close_proteins(protein, max_distance):
    """Find non-toxic proteins within a certain jump distance using BFS."""
    close_proteins = []
    for target, distance in nx.single_source_shortest_path_length(G, source=protein2id[protein], cutoff=max_distance).items():
        if int(target)<44296:
            continue
        entry = id2protein[target]
        if entry in protein2id and entry not in toxic_proteins:
            close_proteins.append((entry, distance))
    return close_proteins


def transE_embedding():
    protein_embeddings = np.load('./data/protein_embeddings.npy', allow_pickle=True).item()
    return protein_embeddings


def calculate_semantic_similarity(embedding_a, embedding_b):
    """Calculate semantic similarity using cosine similarity between embeddings."""

    if len(embedding_a.shape) == 3:
        embedding_a = embedding_a.reshape(-1)
    elif len(embedding_a.shape) == 2:
        embedding_a = embedding_a.flatten()

    if len(embedding_b.shape) == 3:
        embedding_b = embedding_b.reshape(-1)
    elif len(embedding_b.shape) == 2:
        embedding_b = embedding_b.flatten()


    return cosine_similarity([embedding_a], [embedding_b])


protein_embeddings = transE_embedding()


def find_best_non_toxic_protein(toxic_protein, max_distance=3, alpha=0.5, beta=0.5):
    close_proteins = find_close_proteins(toxic_protein, max_distance)
    best_protein = None
    best_score = float('-inf')

    toxic_embedding = protein_embeddings[toxic_protein]

    for non_toxic_protein, graph_distance in close_proteins:
        non_toxic_embedding = protein_embeddings[non_toxic_protein]
        semantic_similarity = calculate_semantic_similarity(toxic_embedding, non_toxic_embedding)

        combined_score = alpha * (1 / (graph_distance + 1)) + beta * semantic_similarity

        if combined_score > best_score:
            best_score = combined_score
            best_protein = non_toxic_protein

    return best_protein


output = []
null_num = 0
for toxic_protein in tqdm(toxic_proteins, desc="Processing toxic proteins"):
    if toxic_protein not in protein2id.keys():
        continue
    best_non_toxic_protein = find_best_non_toxic_protein(toxic_protein)

    toxic_protein_seq = protein_seq.get(toxic_protein)

    non_toxic_protein_seq = protein_seq.get(best_non_toxic_protein)

    if not non_toxic_protein_seq:

        random_non_toxic_protein = random.choice(non_toxic_proteins)
        non_toxic_protein_seq = protein_seq.get(random_non_toxic_protein)
        null_num = null_num+1

    output.append({
        "prompt": "<|endoftext|>",
        "chosen": non_toxic_protein_seq,
        "rejected": toxic_protein_seq
    })

with open('KPO_prune_protein_output_ProtGPT2.json', 'w') as f:
    json.dump(output, f, indent=4)

print("JSON file 'KPO_prune_protein_output_ProtGPT2.json' has been created.")
print("protein null num:{}".format(null_num))