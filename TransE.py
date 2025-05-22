import os
import torch
import pandas as pd
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pykeen.models import TransE
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def read_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def parse_go2id(lines):
    go2id = {}
    for line in lines:
        go, id_ = line.split()
        go2id[go] = int(id_)
    return go2id

def parse_protein2id(lines):
    protein2id = {}
    for line in lines:
        protein, id_ = line.split()
        protein2id[protein] = int(id_)
    return protein2id

def parse_relation2id(lines):
    relation2id = {}
    for line in lines:
        relation, id_ = line.split()
        relation2id[relation] = int(id_)
    return relation2id

def parse_triplets(lines):
    triplets = []
    for line in lines:
        h, r, t = line.split()
        triplets.append((h, int(r), t))
    return triplets

go_type = read_file('./data/PSKG/pretrain_data/go_type.txt')
go2id = parse_go2id(read_file('./data/PSKG/pretrain_data/go2id.txt'))
go_go_triplets = parse_triplets(read_file('./data/PSKG/pretrain_data/go_go_triplet.txt'))
protein_seq = read_file('./data/PSKG/pretrain_data/protein_seq.txt')
protein2id = parse_protein2id(read_file('./data/PSKG/pretrain_data/protein2id_updated.txt'))
protein_go_train_triplets = parse_triplets(read_file('./data/PSKG/pretrain_data/protein_go_triplet_updated.txt'))
relation2id = parse_relation2id(read_file('./data/PSKG/pretrain_data/relation2id.txt'))

id2relation = {v: k for k, v in relation2id.items()}

triples = []
gid = go2id.values()
pid = protein2id.values()
all_entities = set()
for h, r, t in go_go_triplets + protein_go_train_triplets:
    relation = id2relation[r]
    triples.append((h, relation, t))


df = pd.DataFrame(triples, columns=['head', 'relation', 'tail'])

tf = TriplesFactory.from_labeled_triples(df.values)
tf_train, tf_test = tf.split([0.8, 0.2])

pipeline_result = pipeline(
    training=tf_train,
    testing=tf_test,
    model='TransE',
    training_kwargs=dict(num_epochs=100),
)

model = pipeline_result.model
entity_representations = model.entity_representations[0]

protein_embeddings = {protein: entity_representations(torch.tensor([protein2id[protein]])).cpu().detach().numpy() for protein in protein2id}

for protein, embedding in list(protein_embeddings.items())[:5]:
    print(f"Protein: {protein}, Embedding: {embedding}")

import numpy as np
np.save('./data/PSKG/protein_embeddings.npy', protein_embeddings)
