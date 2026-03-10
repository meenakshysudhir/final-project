import torch
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv


class DrugGNN(torch.nn.Module):

    def __init__(self, in_dim=128, hidden_dim=64):

        super().__init__()

        # Project all node types → hidden_dim
        self.proj = nn.ModuleDict({
            "chemical": nn.Linear(in_dim, hidden_dim),
            "disease": nn.Linear(in_dim, hidden_dim),
            "side_effect": nn.Linear(in_dim, hidden_dim),
        })

        self.conv1 = HeteroConv({
            ("chemical","treats","disease"): SAGEConv((hidden_dim,hidden_dim), hidden_dim),
            ("disease","rev_treats","chemical"): SAGEConv((hidden_dim,hidden_dim), hidden_dim),
        })

        self.conv2 = HeteroConv({
            ("chemical","treats","disease"): SAGEConv((hidden_dim,hidden_dim), hidden_dim),
            ("disease","rev_treats","chemical"): SAGEConv((hidden_dim,hidden_dim), hidden_dim),
        })


    def safe_update(self, old_dict, new_dict):

        out = {}

        for key in old_dict:

            if key in new_dict and new_dict[key] is not None:
                out[key] = new_dict[key]
            else:
                out[key] = old_dict[key]

        return out


    def forward(self, x_dict, edge_index_dict):

        # Step 1: project features to hidden size
        x_dict = {
            k: self.proj[k](v)
            for k,v in x_dict.items()
        }

        # Step 2: first GNN layer
        out = self.conv1(x_dict, edge_index_dict)
        out = self.safe_update(x_dict, out)

        out = {k: v.relu() for k,v in out.items()}

        # Step 3: second GNN layer
        out2 = self.conv2(out, edge_index_dict)
        out2 = self.safe_update(out, out2)

        return out2


class LinkPredictor(nn.Module):

    def forward(self, drug_emb, disease_emb):

        return (drug_emb * disease_emb).sum(dim=-1)