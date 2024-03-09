from sklearn.neighbors import KNeighborsClassifier

def train_model(k, X_train, y_train):
    # print("training k nearest neighbours")
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    return model
