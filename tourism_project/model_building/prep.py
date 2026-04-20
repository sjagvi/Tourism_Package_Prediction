# Cleans the raw data and creates train/test splits, then uploads to HF.
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

HF_USER = "iamsubha"
DATA_REPO = f"{HF_USER}/tourism-data"

api = HfApi(token=os.getenv("HF_TOKEN"))

# pull the raw CSV straight from HF hub
df = pd.read_csv(f"hf://datasets/{DATA_REPO}/tourism.csv")
print(f"Loaded data: {df.shape}")

# --- basic cleaning ---
# drop unnamed index cols if they snuck in, and CustomerID (just an ID)
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
df = df.drop(columns=["CustomerID"], errors="ignore")

# fix the 'Fe Male' typo in Gender
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

print(f"After cleaning: {df.shape}")

# --- split features and target ---
target = "ProdTaken"
X = df.drop(columns=[target])
y = df[target]

# stratified split keeps class ratio the same in train and test
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {Xtrain.shape}, Test: {Xtest.shape}")

# save locally first
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# push splits back to the HF dataset repo
for f in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]:
    api.upload_file(
        path_or_fileobj=f,
        path_in_repo=f,
        repo_id=DATA_REPO,
        repo_type="dataset",
    )
    print(f"Uploaded {f}")

print("Data prep done.")
