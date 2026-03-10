import pandas as pd
import torch
import pickle
import os
import re
import gzip
from io import StringIO
from torch_geometric.data import HeteroData

DATA_DIR = "data"

CHEM_VOCAB = os.path.join(DATA_DIR, "CTD_chemicals.csv")
CTD_TREATS = os.path.join(DATA_DIR, "CTD_curated_chemicals_diseases.csv.gz")
SIDER_FILE = os.path.join(DATA_DIR, "meddra_all_se.tsv.gz")


# =========================================
# LOAD CTD FILE (works for .csv AND .gz)
# =========================================

def load_ctd_file(path):

    header = None
    data_lines = []

    # open gzip or normal file
    if path.endswith(".gz"):
        f = gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    else:
        f = open(path, "r", encoding="utf-8", errors="ignore")

    with f:

        for line in f:

            if line.startswith("# Fields:"):
                header = next(f).strip()
                continue

            if line.startswith("#"):
                continue

            if header:
                data_lines.append(line)

    if header is None:
        raise Exception("CTD header not found")

    csv_data = header + "\n" + "".join(data_lines)

    df = pd.read_csv(StringIO(csv_data))

    return df


# =========================================
# CLEAN SIDER STITCH IDS
# =========================================

def clean_stitch_id(x):

    x = str(x)

    if x.startswith("CID"):
        return x[4:].lstrip("0")

    m = re.search(r"\d+", x)

    if m:
        return m.group().lstrip("0")

    return None


# =========================================
# LOAD CTD CHEMICALS
# =========================================

print("🔗 Loading CTD chemicals...")

chem_vocab = load_ctd_file(CHEM_VOCAB)

print("Columns:", chem_vocab.columns)
print("Chemicals loaded:", len(chem_vocab))


# =========================================
# LOAD CTD CHEMICAL-DISEASE RELATIONS
# =========================================

print("\n🏥 Loading CTD chemical-disease relations...")

ctd = load_ctd_file(CTD_TREATS)

print("Columns:", ctd.columns)
print("Total CTD edges:", len(ctd))

# Keep therapeutic relations if available
if "DirectEvidence" in ctd.columns:

    ctd = ctd[
        ctd["DirectEvidence"]
        .astype(str)
        .str.contains("therapeutic", case=False, na=False)
    ]

    print("Therapeutic edges:", len(ctd))


# =========================================
# LOAD SIDER
# =========================================

print("\n💊 Loading SIDER...")

sider = pd.read_csv(
    SIDER_FILE,
    sep="\t",
    header=None,
    compression="gzip"
)

print("SIDER rows:", len(sider))

sider["cid"] = sider[0].apply(clean_stitch_id)

sider = sider.dropna(subset=["cid"])


# =========================================
# BUILD NODE SETS
# =========================================

print("\n🏗 Building node sets...")

all_chems = sorted(set(ctd["ChemicalID"]))
all_diseases = sorted(set(ctd["DiseaseID"]))
all_side_effects = sorted(set(sider[4].dropna().astype(str)))

print("Drugs:", len(all_chems))
print("Diseases:", len(all_diseases))
print("Side effects:", len(all_side_effects))


# =========================================
# INDEX MAPS
# =========================================

chem_map = {v: i for i, v in enumerate(all_chems)}
disease_map = {v: i for i, v in enumerate(all_diseases)}
se_map = {v: i for i, v in enumerate(all_side_effects)}


# =========================================
# BUILD GRAPH
# =========================================

print("\n🏗 Assembling heterograph...")

data = HeteroData()

# Chemical -> Disease edges

chem_idx = []
dis_idx = []

for _, row in ctd.iterrows():

    chem = row["ChemicalID"]
    dis = row["DiseaseID"]

    if chem in chem_map and dis in disease_map:

        chem_idx.append(chem_map[chem])
        dis_idx.append(disease_map[dis])


data["chemical", "treats", "disease"].edge_index = torch.tensor(
    [chem_idx, dis_idx],
    dtype=torch.long
)


# Chemical -> SideEffect edges

chem_se_idx = []
se_idx = []

for _, row in sider.iterrows():

    cid = row["cid"]
    se = str(row[4])

    if cid in chem_map and se in se_map:

        chem_se_idx.append(chem_map[cid])
        se_idx.append(se_map[se])


data["chemical", "causes", "side_effect"].edge_index = torch.tensor(
    [chem_se_idx, se_idx],
    dtype=torch.long
)


# =========================================
# NODE FEATURES
# =========================================

data["chemical"].x = torch.randn(len(chem_map), 128)
data["disease"].x = torch.randn(len(disease_map), 128)
data["side_effect"].x = torch.randn(len(se_map), 128)


# =========================================
# SAVE GRAPH
# =========================================

print("\n💾 Saving graph...")

with open("processed_graph.pkl", "wb") as f:

    pickle.dump(
        {
            "data": data,
            "chem_map": chem_map,
            "disease_map": disease_map,
            "se_map": se_map
        },
        f
    )


print("\n✅ GRAPH BUILT SUCCESSFULLY")
print("Chemicals:", len(chem_map))
print("Diseases:", len(disease_map))
print("SideEffects:", len(se_map))