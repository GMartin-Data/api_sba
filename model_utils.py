import joblib


def load_model(path: str = 'model.pkl'):
    model = joblib.load(path)
    return model

def prediction(model, data):
    predictions = model.predict(data)
    return predictions
