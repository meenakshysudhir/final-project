import torch
import pickle
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model import DrugGNN
from mesh_names_local import MESH_NAMES as mesh_names


# ===============================
# DRUG REPURPOSING + GRAPHING
# ===============================

def repurpose_drug_with_names(drug_id, mesh_names, data, chem_map, int_to_dis, model):

    if drug_id not in chem_map:
        print(f"Drug {drug_id} not found in graph")
        return

    drug_idx = chem_map[drug_id]

    with torch.no_grad():

        x_dict = model(data.x_dict, data.edge_index_dict)

        drug_emb = x_dict["chemical"][drug_idx]
        disease_emb = x_dict["disease"]

        # =====================================
        # FIXED SCORING (COSINE SIMILARITY)
        # =====================================

        drug_emb = F.normalize(drug_emb, dim=0)
        disease_emb = F.normalize(disease_emb, dim=1)

        scores = torch.matmul(disease_emb, drug_emb)

        # convert -1..1 → 0..1 probability
        probs = (scores + 1) / 2

    vals, indices = torch.topk(probs, 10)

    mesh_ids = []
    confidences = []
    disease_names = []

    print(f"\n{'='*80}")
    print(f"Drug Repurposing Candidates for {drug_id}")
    print(f"{'='*80}\n")

    print(f"{'MeSH ID':<15} {'Confidence':<12} {'Disease Name'}")
    print(f"{'-'*15} {'-'*12} {'-'*50}")

    for v, idx in zip(vals, indices):

        mesh_id = int_to_dis[idx.item()]
        disease_name = mesh_names.get(mesh_id, f"{mesh_id}")
        confidence = v.item() * 100

        mesh_ids.append(mesh_id)
        confidences.append(confidence)
        disease_names.append(disease_name)

        bar_length = int(confidence / 5)
        bar = '█' * bar_length + '░' * (20 - bar_length)

        print(f"{mesh_id:<15} {confidence:>6.2f}% {bar} {disease_name}")

    generate_graphs(drug_id, mesh_ids, confidences, disease_names, probs)


# ===============================
# GRAPH GENERATION
# ===============================
def generate_graphs(drug_id, mesh_ids, confidences, disease_names, all_probs):

    print("\nOpening graphs in separate windows...")

    import numpy as np
    import matplotlib.pyplot as plt

    scores = all_probs.cpu().numpy()

    plt.style.use('seaborn-v0_8-darkgrid')

    # =================================
    # 1️⃣ Top Prediction Bar Chart
    # =================================
    fig1, ax1 = plt.subplots(figsize=(9,5))

    ax1.barh(mesh_ids[::-1], confidences[::-1], color="#4E79A7")

    ax1.set_xlabel("Confidence (%)")
    ax1.set_ylabel("Disease (MeSH ID)")
    ax1.set_title(f"Top Drug Repurposing Predictions\nDrug: {drug_id}", fontsize=14)

    plt.tight_layout()
    plt.show()



    # =================================
    # 2️⃣ Confidence Distribution
    # =================================
    fig2, ax2 = plt.subplots(figsize=(8,5))

    ax2.hist(scores, bins=40, color="#F28E2B")

    ax2.set_xlabel("Prediction Probability")
    ax2.set_ylabel("Number of Diseases")
    ax2.set_title(f"Prediction Confidence Distribution\nDrug: {drug_id}", fontsize=14)

    plt.tight_layout()
    plt.show()



    # =================================
    # 3️⃣ Ranking Curve
    # =================================
    sorted_scores = sorted(scores, reverse=True)[:50]

    fig3, ax3 = plt.subplots(figsize=(8,5))

    ax3.plot(sorted_scores, marker='o')

    ax3.set_xlabel("Ranked Disease Index")
    ax3.set_ylabel("Confidence Score")
    ax3.set_title(f"Top-50 Disease Ranking Curve\nDrug: {drug_id}", fontsize=14)

    plt.tight_layout()
    plt.show()



    # =================================
    # 4️⃣ Prediction Heatmap
    # =================================
    heat = np.array(confidences).reshape(1,-1)

    fig4, ax4 = plt.subplots(figsize=(10,2))

    im = ax4.imshow(heat, aspect="auto")

    ax4.set_yticks([])
    ax4.set_xticks(range(len(mesh_ids)))
    ax4.set_xticklabels(mesh_ids, rotation=45)

    ax4.set_title(f"Prediction Confidence Heatmap\nDrug: {drug_id}")

    plt.colorbar(im)

    plt.tight_layout()
    plt.show()
# ===============================
# MAIN EXECUTION
# ===============================

if __name__ == "__main__":

    print("\nLoading graph data...")

    with open("processed_graph.pkl", "rb") as f:
        saved = pickle.load(f)

    data = saved["data"]
    chem_map = saved["chem_map"]
    dis_map = saved["disease_map"]

    int_to_chem = {v: k for k, v in chem_map.items()}
    int_to_dis = {v: k for k, v in dis_map.items()}

    print("\nLoaded graph data:")
    print(f"Total drugs: {len(chem_map)}")
    print(f"Total diseases: {len(dis_map)}")

    print("\nLoading model...")

    model = DrugGNN()
    model.load_state_dict(torch.load("drug_gnn_best.pt", map_location="cpu"))
    model.eval()

    print("\n" + "="*80)
    print("TESTING DRUG REPURPOSING")
    print("="*80)

    test_drugs = random.sample(list(chem_map.keys()), min(2, len(chem_map)))

    for i, drug in enumerate(test_drugs, 1):

        print(f"\n{'#'*80}")
        print(f"# TEST {i} of {len(test_drugs)}")
        print(f"{'#'*80}")

        repurpose_drug_with_names(
            drug,
            mesh_names,
            data,
            chem_map,
            int_to_dis,
            model
        )

    print(f"\n{'='*80}")
    print("STATISTICS")
    print(f"{'='*80}")

    print(f"Total drugs in graph: {len(chem_map)}")
    print(f"Total diseases in graph: {len(dis_map)}")
    print(f"MeSH names loaded: {len(mesh_names)}")

    mesh_ids_with_names = set(mesh_names.keys())
    covered = set(int_to_dis.values()).intersection(mesh_ids_with_names)

    print(
        f"Diseases in graph with readable names: "
        f"{len(covered)}/{len(dis_map)} "
        f"({len(covered)/len(dis_map)*100:.1f}%)"
    )