from sklearn.linear_model import LinearRegression

def train_model(X_train, y_train):
    print("training linear regression model")
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model