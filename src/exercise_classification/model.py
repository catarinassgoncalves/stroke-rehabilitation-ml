import joblib
import numpy as np
from preprocessing import Normalize, Flip, SaySide

model = joblib.load("models/exercise_classifier.pkl")

def predict(X_test):
    X_test_processed = np.hstack([Flip(Normalize(X_test.copy())), SaySide(X_test)])
    return model.predict(X_test_processed)