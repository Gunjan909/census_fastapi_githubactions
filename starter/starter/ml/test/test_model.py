import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ..data import process_data
from ..model import train_model, inference, compute_model_metrics
import os
import pickle

# ---------------------
# Sample fixture data
# ---------------------
@pytest.fixture
def sample_data():
    data = pd.DataFrame({
        "age": [25, 32, 47],
        "education": ["Bachelors", "Masters", "PhD"],
        "salary": ["<=50K", ">50K", ">50K"]
    })
    categorical_features = ["education"]
    label = "salary"
    return data, categorical_features, label


# ---------------------
# process_data tests
# ---------------------
def test_process_data_training(sample_data):
    df, cat_features, label = sample_data
    X_processed, y, encoder, lb = process_data(
        df, categorical_features=cat_features, label=label, training=True
    )

    # Check output shapes
    assert X_processed.shape[0] == df.shape[0]
    assert y.shape[0] == df.shape[0]
    assert encoder is not None
    assert lb is not None


def test_process_data_inference(sample_data):
    df, cat_features, label = sample_data
    X_train, y_train, encoder, lb = process_data(
        df, categorical_features=cat_features, label=label, training=True
    )

    # Use same encoder/lb for inference
    X_test, y_test, _, _ = process_data(
        df, categorical_features=cat_features, label=label,
        training=False, encoder=encoder, lb=lb
    )

    assert X_test.shape == X_train.shape
    assert np.array_equal(y_test, y_train)


# ---------------------
# train_model tests
# ---------------------
def test_train_model(sample_data):
    df, cat_features, label = sample_data
    X, y, _, _ = process_data(df, categorical_features=cat_features, label=label)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


# ---------------------
# inference tests
# ---------------------
def test_inference(sample_data):
    df, cat_features, label = sample_data
    X, y, _, _ = process_data(df, categorical_features=cat_features, label=label)
    model = train_model(X, y)
    preds = inference(model, X)

    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})


# ---------------------
# Metrics tests
# ---------------------
def test_compute_model_metrics_perfect_preds():
    y_true = np.array([0, 1, 1])
    y_pred = np.array([0, 1, 1])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0



def test_model_loading_and_prediction():
    # Path to the saved model
    model_path = "/home/gunjan/workspace/census_fastapi_githubactions/starter/model/model.pkl"
    
    assert os.path.exists(model_path), "model.pkl file not found"

    # Load the model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Assert model has predict method
    assert hasattr(model, "predict"), "Loaded object is not a valid model"
