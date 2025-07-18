from data import process_data, get_processed_column_names
import pandas as pd
from sklearn.model_selection import train_test_split
from model import train_model, inference, compute_model_metrics
import random
import numpy as np
import pickle

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

import numpy as np

def predict_by_slice(model, X, y, feature_names, prefix):
    """
    For each column matching the prefix (e.g. 'native-country'), run model predictions
    on rows where that column equals 1, i.e. that slice of the feature given by "prefix"

    Parameters
    ----------
    model : sklearn classifier
        Trained model with `.predict()`.
    X : np.ndarray
        Processed feature matrix.
    feature_names : list[str]
        Column names corresponding to X.
    prefix : str
        Prefix to match (e.g. 'native-country').
    y: array
        True labels

    Returns
    -------
    results : dict
        Dictionary mapping column name to a dict:
            {
                "predictions": np.ndarray,
                "indices": np.ndarray,
                "X_subset": np.ndarray
            }
    """
    results = {}

    # Get all column indices matching the prefix
    for i, col in enumerate(feature_names):
        if col.startswith(prefix + "_"):
            mask = X[:, i] == 1
            X_subset = X[mask]
            indices = np.where(mask)[0]

            if X_subset.shape[0] == 0:
                continue  # skip if no matching rows

            preds = model.predict(X_subset)
            precision, recall, fbeta = compute_model_metrics(y[mask], preds)

            #preds = inference(model, X_test)

            results[col] = {
                "predictions": preds,
                "indices": indices,
                "X_subset": X_subset,
                "precision": precision,
                "recall": recall,
                "fbeta": fbeta
            }

    return results



DATA_DIR = '/home/gunjan/workspace/census_fastapi_githubactions/starter/data/'
df_raw = pd.read_csv(DATA_DIR+'census.csv')
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

#clean the data of whitespace
# Remove whitespace from column names
df_raw.columns = df_raw.columns.str.strip()

# Remove whitespace from string values in all columns (if they are strings)
for col in df_raw.columns:
    if df_raw[col].dtype == "object":  # only apply to string columns
        df_raw[col] = df_raw[col].str.strip()

train, test = train_test_split(df_raw, test_size=0.20)

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)



X_test, y_test, encoder_test, lb_test = process_data(
    test, categorical_features=cat_features, label='salary', encoder=encoder, lb=lb, training=False
)


feature_names = get_processed_column_names(test, cat_features, encoder)


#load the model back in
with open(DATA_DIR + "/../model/model.pkl", "rb") as f:
    model = pickle.load(f)

results_country = predict_by_slice(model, X_test, y_test, feature_names, prefix="native-country")
results_occupation = predict_by_slice(model, X_test, y_test, feature_names, prefix="occupation")


for results in [results_country, results_occupation]:
    for col, data in results.items():
        print(f"\n== {col} ==")
        print(f"Rows: {len(data['indices'])}")
        print(f"Precision: {data['precision']}")
        print(f"Recall: {data['recall']}")
        print(f"fbeta: {data['fbeta']}")

    #print(f"Predictions: {data['predictions'][:5]}")  # just show first 5