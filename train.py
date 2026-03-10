import torch
import pickle
import random
from torch import nn
from torch_geometric.transforms import ToUndirected
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

split = int(num_edges*0.8)

train_idx = indices[:split]

train_edges = edge_index[:,train_idx]


model = DrugGNN()
predictor = LinkPredictor()

optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

loss_fn = nn.BCEWithLogitsLoss()


print("Training model...")

for epoch in range(30):

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

    neg_drugs = torch.randint(0,drug_emb.size(0),(len(pos_drugs),))
    neg_dis = torch.randint(0,disease_emb.size(0),(len(pos_dis),))

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

    print(f"Epoch {epoch} | Loss {loss.item():.4f}")


torch.save(model.state_dict(),"drug_gnn.pt")

print("Model saved.")