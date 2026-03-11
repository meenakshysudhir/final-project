# import torch
# import pickle
# import random
# from torch import nn
# from torch_geometric.transforms import ToUndirected
# from sklearn.metrics import roc_auc_score, average_precision_score
# from model import DrugGNN, LinkPredictor


# print("Loading graph...")

# with open("processed_graph.pkl","rb") as f:
#     saved = pickle.load(f)

# data = saved["data"]

# data = ToUndirected()(data)

# edge_index = data["chemical","treats","disease"].edge_index

# num_edges = edge_index.size(1)

# indices = list(range(num_edges))
# random.shuffle(indices)

# split = int(num_edges*0.8)

# train_idx = indices[:split]
# test_idx = indices[split:]

# train_edges = edge_index[:,train_idx]
# test_edges = edge_index[:,test_idx]


# model = DrugGNN()
# predictor = LinkPredictor()

# optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

# loss_fn = nn.BCEWithLogitsLoss()


# print("Training model...")

# for epoch in range(100):

#     model.train()

#     optimizer.zero_grad()

#     x_dict = model(data.x_dict,data.edge_index_dict)

#     drug_emb = x_dict["chemical"]
#     disease_emb = x_dict["disease"]

#     # ======================
#     # POSITIVE TRAIN EDGES
#     # ======================

#     pos_drugs = train_edges[0]
#     pos_dis = train_edges[1]

#     pos_score = predictor(
#         drug_emb[pos_drugs],
#         disease_emb[pos_dis]
#     )

#     # ======================
#     # NEGATIVE SAMPLING
#     # ======================

#     neg_drugs = torch.randint(0,drug_emb.size(0),(len(pos_drugs),))
#     neg_dis = torch.randint(0,disease_emb.size(0),(len(pos_dis),))

#     neg_score = predictor(
#         drug_emb[neg_drugs],
#         disease_emb[neg_dis]
#     )

#     labels = torch.cat([
#         torch.ones(len(pos_score)),
#         torch.zeros(len(neg_score))
#     ])

#     scores = torch.cat([pos_score,neg_score])

#     loss = loss_fn(scores,labels)

#     loss.backward()

#     optimizer.step()

#     # ======================
#     # EVALUATION
#     # ======================

#     model.eval()

#     with torch.no_grad():

#         x_dict = model(data.x_dict,data.edge_index_dict)

#         drug_emb = x_dict["chemical"]
#         disease_emb = x_dict["disease"]

#         pos_drugs = test_edges[0]
#         pos_dis = test_edges[1]

#         pos_score = predictor(
#             drug_emb[pos_drugs],
#             disease_emb[pos_dis]
#         )

#         neg_drugs = torch.randint(0,drug_emb.size(0),(len(pos_drugs),))
#         neg_dis = torch.randint(0,disease_emb.size(0),(len(pos_dis),))

#         neg_score = predictor(
#             drug_emb[neg_drugs],
#             disease_emb[neg_dis]
#         )

#         scores = torch.cat([pos_score,neg_score])

#         labels = torch.cat([
#             torch.ones(len(pos_score)),
#             torch.zeros(len(neg_score))
#         ])

#         probs = torch.sigmoid(scores).cpu().numpy()
#         labels_np = labels.cpu().numpy()

#         auroc = roc_auc_score(labels_np, probs)
#         auprc = average_precision_score(labels_np, probs)

#     print(
#         f"Epoch {epoch} | "
#         f"Loss {loss.item():.4f} | "
#         f"AUROC {auroc:.4f} | "
#         f"AUPRC {auprc:.4f}"
#     )


# torch.save(model.state_dict(),"drug_gnn.pt")

# print("Model saved.")


import torch
import pickle
import random
from torch import nn
from torch_geometric.transforms import ToUndirected
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from model import DrugGNN, LinkPredictor


print("Loading graph...")

with open("processed_graph.pkl","rb") as f:
    saved = pickle.load(f)

data = saved["data"]

data = ToUndirected()(data)

edge_index = data["chemical","treats","disease"].edge_index
num_edges = edge_index.size(1)

indices = list(range(num_edges))
random.shuffle(indices)

# =============================
# TRAIN / VAL / TEST SPLIT
# =============================

train_split = int(num_edges * 0.8)
val_split = int(num_edges * 0.9)

train_idx = indices[:train_split]
val_idx = indices[train_split:val_split]
test_idx = indices[val_split:]

train_edges = edge_index[:,train_idx]
val_edges = edge_index[:,val_idx]
test_edges = edge_index[:,test_idx]

# =============================
# REMOVE TEST EDGES FROM GRAPH
# =============================

data["chemical","treats","disease"].edge_index = train_edges

# node counts
num_drugs = data["chemical"].x.size(0)
num_diseases = data["disease"].x.size(0)

# =============================
# FIXED NEGATIVE EDGES
# =============================

def sample_negatives(num_samples):

    neg_drugs = torch.randint(0,num_drugs,(num_samples,))
    neg_dis = torch.randint(0,num_diseases,(num_samples,))

    return neg_drugs,neg_dis


val_neg_drugs,val_neg_dis = sample_negatives(val_edges.size(1))
test_neg_drugs,test_neg_dis = sample_negatives(test_edges.size(1))

# =============================
# MODEL
# =============================

model = DrugGNN()
predictor = LinkPredictor()

optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
loss_fn = nn.BCEWithLogitsLoss()

best_val_auc = 0

print("Training model...")

for epoch in range(100):

    model.train()

    optimizer.zero_grad()

    x_dict = model(data.x_dict,data.edge_index_dict)

    drug_emb = x_dict["chemical"]
    disease_emb = x_dict["disease"]

    pos_drugs = train_edges[0]
    pos_dis = train_edges[1]

    pos_score = predictor(
        drug_emb[pos_drugs],
        disease_emb[pos_dis]
    )

    # negative samples for training
    neg_drugs,neg_dis = sample_negatives(len(pos_drugs))

    neg_score = predictor(
        drug_emb[neg_drugs],
        disease_emb[neg_dis]
    )

    labels = torch.cat([
        torch.ones(len(pos_score)),
        torch.zeros(len(neg_score))
    ])

    scores = torch.cat([pos_score,neg_score])

    loss = loss_fn(scores,labels)

    loss.backward()
    optimizer.step()

    # =============================
    # VALIDATION
    # =============================

    model.eval()

    with torch.no_grad():

        x_dict = model(data.x_dict,data.edge_index_dict)

        drug_emb = x_dict["chemical"]
        disease_emb = x_dict["disease"]

        pos_drugs = val_edges[0]
        pos_dis = val_edges[1]

        pos_score = predictor(
            drug_emb[pos_drugs],
            disease_emb[pos_dis]
        )

        neg_score = predictor(
            drug_emb[val_neg_drugs],
            disease_emb[val_neg_dis]
        )

        scores = torch.cat([pos_score,neg_score])

        labels = torch.cat([
            torch.ones(len(pos_score)),
            torch.zeros(len(neg_score))
        ])

        probs = torch.sigmoid(scores).cpu().numpy()
        labels_np = labels.cpu().numpy()

        val_auc = roc_auc_score(labels_np,probs)
        val_auprc = average_precision_score(labels_np,probs)

    # save best model
    if val_auc > best_val_auc:

        best_val_auc = val_auc
        torch.save(model.state_dict(),"drug_gnn_best.pt")

    print(
        f"Epoch {epoch} | "
        f"Loss {loss.item():.4f} | "
        f"Val AUROC {val_auc:.4f} | "
        f"Val AUPRC {val_auprc:.4f}"
    )

# =============================
# FINAL TEST EVALUATION
# =============================

print("\nEvaluating on test set...")

model.load_state_dict(torch.load("drug_gnn_best.pt"))
model.eval()

with torch.no_grad():

    x_dict = model(data.x_dict,data.edge_index_dict)

    drug_emb = x_dict["chemical"]
    disease_emb = x_dict["disease"]

    pos_drugs = test_edges[0]
    pos_dis = test_edges[1]

    pos_score = predictor(
        drug_emb[pos_drugs],
        disease_emb[pos_dis]
    )

    neg_score = predictor(
        drug_emb[test_neg_drugs],
        disease_emb[test_neg_dis]
    )

    scores = torch.cat([pos_score,neg_score])

    labels = torch.cat([
        torch.ones(len(pos_score)),
        torch.zeros(len(neg_score))
    ])

    probs = torch.sigmoid(scores).cpu().numpy()
    labels_np = labels.cpu().numpy()

    # AUROC & AUPRC
    test_auc = roc_auc_score(labels_np,probs)
    test_auprc = average_precision_score(labels_np,probs)

    # Convert probabilities → binary predictions
    preds = (probs > 0.5).astype(int)

    # Precision / Recall / F1
    precision = precision_score(labels_np,preds)
    recall = recall_score(labels_np,preds)
    f1 = f1_score(labels_np,preds)

print("\nFinal Test Metrics")
print("AUROC:",test_auc)
print("AUPRC:",test_auprc)
print("Precision:",precision)
print("Recall:",recall)
print("F1 Score:",f1)