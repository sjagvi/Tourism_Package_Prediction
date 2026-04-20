# Pushes the deployment files to a Hugging Face Space
import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

HF_USER = "iamsubha"
SPACE_REPO = f"{HF_USER}/tourism-package-prediction"

api = HfApi(token=os.getenv("HF_TOKEN"))

# make sure the Space exists (as a Docker space)
try:
    api.repo_info(repo_id=SPACE_REPO, repo_type="space")
    print(f"Space '{SPACE_REPO}' already exists.")
except RepositoryNotFoundError:
    print(f"Creating new Space '{SPACE_REPO}'...")
    create_repo(repo_id=SPACE_REPO, repo_type="space",
                space_sdk="docker", private=False)

# upload the whole deployment folder
api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id=SPACE_REPO,
    repo_type="space",
    path_in_repo="",
)
print("Deployment files pushed to Space.")
