# Registers the raw dataset on Hugging Face
import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# TODO: replace with your HF username
HF_USER = "your-hf-username"
REPO_ID = f"{HF_USER}/tourism-data"
REPO_TYPE = "dataset"

api = HfApi(token=os.getenv("HF_TOKEN"))

# create the dataset repo if it doesn't already exist
try:
    api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
    print(f"Dataset repo '{REPO_ID}' already there, reusing it.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{REPO_ID}' not found, creating it...")
    create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, private=False)
    print("Created.")

# push the data folder (contains tourism.csv)
api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
)
print("Data uploaded to HF.")
