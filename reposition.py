import torch
import pickle
import random
from model import DrugGNN

print("Loading graph...")

with open("processed_graph.pkl","rb") as f:
    saved = pickle.load(f)

data = saved["data"]
chem_map = saved["chem_map"]
dis_map = saved["disease_map"]

# Reverse maps
int_to_chem = {v:k for k,v in chem_map.items()}
int_to_dis = {v:k for k,v in dis_map.items()}

model = DrugGNN()
model.load_state_dict(torch.load("drug_gnn.pt",map_location="cpu"))
model.eval()


# ===============================
# DRUG REPURPOSING FUNCTION
# ===============================

def repurpose_drug(drug_id):

    if drug_id not in chem_map:
        print("Drug not found in graph")
        return

    drug_idx = chem_map[drug_id]

    with torch.no_grad():

        x_dict = model(data.x_dict,data.edge_index_dict)

        drug_emb = x_dict["chemical"][drug_idx]
        disease_emb = x_dict["disease"]

        # score drug against ALL diseases
        scores = torch.matmul(disease_emb, drug_emb)

        probs = torch.sigmoid(scores)

    vals,indices = torch.topk(probs,10)

    print(f"\nDrug Repurposing Candidates for {drug_id}\n")

    for v,i in zip(vals,indices):

        disease = int_to_dis[i.item()]

        print(f"{disease}  →  {v.item():.2%}")


# ===============================
# RUN EXAMPLE
# ===============================

random_drug = random.choice(list(chem_map.keys()))

print("Testing drug:",random_drug)

repurpose_drug(random_drug)