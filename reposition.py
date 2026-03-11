# import torch
# import pickle
# import random
# from model import DrugGNN  # Make sure this is your GNN model
# from mesh_names_local import MESH_NAMES as mesh_names  # Load local MeSH names

# # ===============================
# # DRUG REPURPOSING FUNCTION WITH READABLE NAMES
# # ===============================

# def repurpose_drug_with_names(drug_id, mesh_names, data, chem_map, int_to_dis, model):
#     """
#     Drug repurposing function with human-readable disease names
#     """
#     if drug_id not in chem_map:
#         print(f"Drug {drug_id} not found in graph")
#         return

#     drug_idx = chem_map[drug_id]

#     with torch.no_grad():
#         x_dict = model(data.x_dict, data.edge_index_dict)
#         drug_emb = x_dict["chemical"][drug_idx]
#         disease_emb = x_dict["disease"]

#         # Score drug against ALL diseases
#         scores = torch.matmul(disease_emb, drug_emb)
#         probs = torch.sigmoid(scores)

#     vals, indices = torch.topk(probs, 10)

#     print(f"\n{'='*80}")
#     print(f"Drug Repurposing Candidates for {drug_id}")
#     print(f"{'='*80}\n")
#     print(f"{'MeSH ID':<15} {'Confidence':<12} {'Disease Name'}")
#     print(f"{'-'*15} {'-'*12} {'-'*50}")

#     for v, idx in zip(vals, indices):
#         mesh_id = int_to_dis[idx.item()]
#         disease_name = mesh_names.get(mesh_id, f"{mesh_id} (name not found)")
#         confidence = v.item() * 100

#         # Progress bar visualization
#         bar_length = int(confidence / 5)
#         bar = '█' * bar_length + '░' * (20 - bar_length)

#         print(f"{mesh_id:<15} {confidence:>6.2f}% {bar} {disease_name}")

# # ===============================
# # MAIN EXECUTION
# # ===============================

# if __name__ == "__main__":
#     # Step 1: Load graph data
#     print("\nLoading graph data...")
#     with open("processed_graph.pkl", "rb") as f:
#         saved = pickle.load(f)

#     data = saved["data"]
#     chem_map = saved["chem_map"]
#     dis_map = saved["disease_map"]

#     # Reverse maps
#     int_to_chem = {v: k for k, v in chem_map.items()}
#     int_to_dis = {v: k for k, v in dis_map.items()}

#     print("\nLoaded graph data:")
#     print(f"Total drugs: {len(chem_map)}")
#     print(f"Total diseases: {len(dis_map)}")

#     # Step 2: Load model
#     print("\nLoading model...")
#     model = DrugGNN()
#     model.load_state_dict(torch.load("drug_gnn.pt", map_location="cpu"))
#     model.eval()

#     # Step 3: Test with random drugs
#     print("\n" + "="*80)
#     print("TESTING DRUG REPURPOSING")
#     print("="*80)

#     test_drugs = random.sample(list(chem_map.keys()), min(2, len(chem_map)))
#     for i, drug in enumerate(test_drugs, 1):
#         print(f"\n{'#'*80}")
#         print(f"# TEST {i} of {len(test_drugs)}")
#         print(f"{'#'*80}")
#         repurpose_drug_with_names(drug, mesh_names, data, chem_map, int_to_dis, model)

#     # Step 4: Show statistics
#     print(f"\n{'='*80}")
#     print("STATISTICS")
#     print(f"{'='*80}")
#     print(f"Total drugs in graph: {len(chem_map)}")
#     print(f"Total diseases in graph: {len(dis_map)}")
#     print(f"MeSH names loaded: {len(mesh_names)}")

#     # Check coverage
#     mesh_ids_with_names = set(mesh_names.keys())
#     covered = set(int_to_dis.values()).intersection(mesh_ids_with_names)
#     print(f"Diseases in graph with readable names: {len(covered)}/{len(dis_map)} "
#           f"({len(covered)/len(dis_map)*100:.1f}%)")
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
        disease_name = mesh_names.get(mesh_id, f"{mesh_id} (name not found)")
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

    print("\nGenerating graphs...")

    # ------------------------------
    # Top predictions bar chart
    # ------------------------------

    plt.figure()

    plt.barh(mesh_ids[::-1], confidences[::-1])

    plt.xlabel("Confidence (%)")
    plt.title(f"Top Drug Repurposing Predictions for {drug_id}")

    plt.tight_layout()
    plt.savefig("top_predictions.png")

    print("Saved: top_predictions.png")

    # ------------------------------
    # Score distribution
    # ------------------------------

    plt.figure()

    scores = all_probs.cpu().numpy()

    plt.hist(scores, bins=50)

    plt.xlabel("Prediction Probability")
    plt.ylabel("Count")
    plt.title("Distribution of Drug-Disease Prediction Scores")

    plt.tight_layout()
    plt.savefig("confidence_distribution.png")

    print("Saved: confidence_distribution.png")


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