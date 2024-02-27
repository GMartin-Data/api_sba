import joblib
import pandas as pd


def load_model(path: str = 'model.pkl'):
    model = joblib.load(path)
    return model

def predict(model, data: pd.DataFrame) -> int:
    predictions = model.predict(data)
    return predictions[0]
