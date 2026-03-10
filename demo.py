import torch
import pickle
import random

# ==============================
# LOAD GRAPH
# ==============================

with open("processed_graph.pkl", "rb") as f:
    saved = pickle.load(f)

data = saved["data"]
chem_map = saved["chem_map"]
dis_map = saved["disease_map"]

# Reverse maps for printing
int_to_chem = {v: k for k, v in chem_map.items()}
int_to_dis = {v: k for k, v in dis_map.items()}

# ==============================
# LOAD MODEL
# ==============================

from train import DrugGNN, LinkPredictor

model = DrugGNN()
predictor = LinkPredictor()

model.load_state_dict(torch.load("drug_gnn.pt", map_location="cpu"))

model.eval()

# ==============================
# REPOSITION FUNCTION
# ==============================

def reposition_for_disease(disease_mesh_id):

    if disease_mesh_id not in dis_map:
        print("Disease not in graph.")
        return

    disease_idx = dis_map[disease_mesh_id]

    with torch.no_grad():

        x_dict = model(data.x_dict, data.edge_index_dict)

        drug_emb = x_dict["chemical"]
        disease_emb = x_dict["disease"]

        target_disease_emb = disease_emb[disease_idx]

        scores = torch.matmul(drug_emb, target_disease_emb)

        probs = torch.sigmoid(scores)

    top_vals, top_idx = torch.topk(probs, 10)

    print(f"\nTop Repositioning Candidates for {disease_mesh_id}:\n")

    for val, idx in zip(top_vals, top_idx):

        chem_id = int_to_chem[idx.item()]

        print(f"{chem_id} → {val.item():.2%} confidence")


# ==============================
# RUN EXAMPLE
# ==============================

random_disease = random.choice(list(dis_map.keys()))

print("Testing disease:", random_disease)

reposition_for_disease(random_disease)