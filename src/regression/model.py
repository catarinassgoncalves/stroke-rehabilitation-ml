import joblib

model = joblib.load("models/regression_model.pkl")

def predict(X_test):
    return model.predict(X_test)