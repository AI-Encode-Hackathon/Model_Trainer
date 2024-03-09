from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from model_trainers import k_means, k_nearest_neighbours, linear_regression, mlp
from pre_process_audio import create_audio_embedding
from pre_process_images import create_image_embedding
from tqdm import tqdm

app = FastAPI()

origins = [
    "http://localhost:8000",
    "http://localhost:4943",
    "http://0.0.0.0:8000",
    "http://0.0.0.0:4943",
]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"])
#

class Response(BaseModel):
    accuracy: str
    training_method: str
    training_time: str
    threshold: float
    learning_rate: float

@app.post("/train_model")
def train_model(labels: UploadFile, training_method: str, learning_class: str, learning_rate: float, threshold: float) -> Response:
    print("loading csv")
    labels_and_paths = pd.read_csv(labels.file)

    print("creating embeddings")
    embeddings = []
    classes = []
    beg = "C:/Users/Student/Downloads/archive/"
    pbar = tqdm(desc="create embeddings", total=14493)
    for label, path in zip(labels_and_paths["label"], labels_and_paths["path"]):
        full_path = beg+path
        try:
            if path.split(".")[1] == "mp4":
                emb = create_audio_embedding(full_path)
            elif path.split(".")[1] == "jpg":
                emb = create_image_embedding(full_path)

        except FileNotFoundError:
            print("path does not exist")
            continue

        pbar.update(1)
        embeddings.append(emb)
        classes.append(label)

    print("splitting dataset")
    X_train, X_test, y_train, y_test = train_test_split(embeddings, classes, test_size=0.2, random_state=42)

    print("selecting training method")
    if learning_class == "unsupervised learning":
        model = k_means.train_model()
    else:
        match training_method:
            case "k-nearest neighbours":
                model = k_nearest_neighbours.train_model(int(learning_rate), X_train, y_train)
            case "mlp":
                model = mlp.train_model()
            case "regression":
                model = linear_regression.train_model()

    print("evaluating model")
    accuracy = model.score(X_test, y_test)

    return {
        "accuracy": str(accuracy),
        "training_method": training_method,
        "training_time": "10s",
        "threshold": threshold,
        "learning_rate": learning_rate
    }

@app.get("/download_model")
def download_model():
    return {}