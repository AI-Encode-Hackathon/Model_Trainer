from sklearn.cluster import KMeans

def train_model(X_train, k):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X_train)

    return model
