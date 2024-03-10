from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from model_trainers import k_means, k_nearest_neighbours, linear_regression, mlp, decision_tree
from pre_process_audio import create_audio_embedding
from pre_process_images import create_image_embedding
from tqdm import tqdm
import pickle
import torch

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
    k: int

@app.post("/train_model")
def train_model(labels: UploadFile, training_method: str="regression", learning_class: str="supervised learning", k: int=7) -> Response:
    start = datetime.now()

    print("loading csv")
    labels_and_paths = pd.read_csv(labels.file)

    # Create embeddings for training data
    print("creating embeddings")
    embeddings = []
    classes = []
    beg = "C:/Users/Student/Downloads/archive (1)"
    pbar = tqdm(desc="create embeddings", total=14493)
    for label, path in zip(labels_and_paths["label"], labels_and_paths["path"]):
        full_path = beg+"/"+path
        try:
            if path.split(".")[1] == "wav":
                wav2mel = torch.jit.load("wav2mel.pt")
                dvector = torch.jit.load("dvector.pt").eval()
                emb = create_audio_embedding(full_path, wav2mel, dvector)
            elif path.split(".")[1] == "jpg":
                emb = create_image_embedding(full_path)
            embeddings.append(emb)
            classes.append(label)
            pbar.update(1)

        except FileNotFoundError:
            print("path does not exist")
            continue


    print("selecting training method")
    if learning_class == "unsupervised learning":
        model = k_means.train_model(k, embeddings)
    else:
        # Split dataset into test and train
        print("splitting dataset")
        X_train, X_test, y_train, y_test = train_test_split(embeddings, classes, test_size=0.2, random_state=42)

        match training_method:
            case "k-nearest neighbours":
                model = k_nearest_neighbours.train_model(k, X_train, y_train)
            case "mlp":
                k="n/a"
                model = mlp.train_model(X_train, y_train)
            case "regression":
                k="n/a"
                model = linear_regression.train_model(X_train, y_train)
            case "decision tree":
                k = "n/a"
                model = decision_tree.train_model(X_train, y_train)

    end = datetime.now()
    time_taken = f"{end - start} seconds"
    print("evaluating model")

    if learning_class == "supervised":
        accuracy = model.score(X_test, y_test)
    else:
        accuracy = "n/a"

    # Save model
    pickle.dump(model, open((training_method+".pkl"), 'wb'))

    return {
        "accuracy": str(accuracy),
        "training_method": training_method,
        "training_time": time_taken,
        "k": k
    }

