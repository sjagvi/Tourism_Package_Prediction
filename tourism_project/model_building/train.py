# Trains a Random Forest on the tourism data, tracks with MLflow,
# and pushes the best model to the Hugging Face model hub.
import os
import pandas as pd
import joblib
import mlflow
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

HF_USER = "iamsubha"
DATA_REPO = f"{HF_USER}/tourism-data"
MODEL_REPO = f"{HF_USER}/tourism-package-model"
MODEL_FILE = "tourism_rf_model.joblib"

api = HfApi(token=os.getenv("HF_TOKEN"))

# point MLflow at the local tracking server started by the GitHub Action
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-package-rf")

# --- load splits from HF ---
base = f"hf://datasets/{DATA_REPO}"
Xtrain = pd.read_csv(f"{base}/Xtrain.csv")
Xtest  = pd.read_csv(f"{base}/Xtest.csv")
ytrain = pd.read_csv(f"{base}/ytrain.csv").squeeze()
ytest  = pd.read_csv(f"{base}/ytest.csv").squeeze()
print(f"Train: {Xtrain.shape}, Test: {Xtest.shape}")

# --- figure out numeric vs categorical cols ---
numeric_cols = Xtrain.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = Xtrain.select_dtypes(include=["object"]).columns.tolist()
print("Numeric:", numeric_cols)
print("Categorical:", categorical_cols)

# --- preprocessing pipelines ---
numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_cols),
    ("cat", categorical_pipe, categorical_cols),
])

# --- full pipeline ---
rf = RandomForestClassifier(
    class_weight="balanced",   # handle the 80/20 imbalance
    random_state=42,
    n_jobs=-1,
)

pipe = Pipeline([
    ("prep",  preprocessor),
    ("model", rf),
])

# small grid - keeping this lightweight so CI doesn't take forever
param_grid = {
    "model__n_estimators":    [150, 250],
    "model__max_depth":       [8, 12, None],
    "model__min_samples_leaf": [1, 2],
}

with mlflow.start_run():
    grid = GridSearchCV(pipe, param_grid, cv=3, scoring="f1", n_jobs=-1)
    grid.fit(Xtrain, ytrain)

    mlflow.log_params(grid.best_params_)
    print("Best params:", grid.best_params_)

    best = grid.best_estimator_

    # evaluate on train and test
    train_pred = best.predict(Xtrain)
    test_pred  = best.predict(Xtest)

    train_rep = classification_report(ytrain, train_pred, output_dict=True)
    test_rep  = classification_report(ytest,  test_pred,  output_dict=True)

    mlflow.log_metrics({
        "train_accuracy":  train_rep["accuracy"],
        "train_precision": train_rep["1"]["precision"],
        "train_recall":    train_rep["1"]["recall"],
        "train_f1":        train_rep["1"]["f1-score"],
        "test_accuracy":   test_rep["accuracy"],
        "test_precision":  test_rep["1"]["precision"],
        "test_recall":     test_rep["1"]["recall"],
        "test_f1":         test_rep["1"]["f1-score"],
    })

    print("\nTest classification report:")
    print(classification_report(ytest, test_pred))

    # save + log the model artifact
    joblib.dump(best, MODEL_FILE)
    mlflow.log_artifact(MODEL_FILE, artifact_path="model")

    # make sure the HF model repo exists, then upload
    try:
        api.repo_info(repo_id=MODEL_REPO, repo_type="model")
        print(f"Model repo '{MODEL_REPO}' exists, using it.")
    except RepositoryNotFoundError:
        print(f"Creating model repo '{MODEL_REPO}'...")
        create_repo(repo_id=MODEL_REPO, repo_type="model", private=False)

    api.upload_file(
        path_or_fileobj=MODEL_FILE,
        path_in_repo=MODEL_FILE,
        repo_id=MODEL_REPO,
        repo_type="model",
    )
    print("Model pushed to HF hub.")
