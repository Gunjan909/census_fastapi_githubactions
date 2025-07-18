# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from ml.data import process_data, get_processed_column_names
from ml.model import train_model, inference, compute_model_metrics
import os
import pandas as pd
import numpy as np
import pickle
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Add code to load in the data.
DATA_DIR = '/home/gunjan/workspace/census_fastapi_githubactions/starter/data/'
df_raw = pd.read_csv(DATA_DIR+'census.csv')

#clean the data of whitespace
# Remove whitespace from column names
df_raw.columns = df_raw.columns.str.strip()

# Remove whitespace from string values in all columns (if they are strings)
for col in df_raw.columns:
    if df_raw[col].dtype == "object":  # only apply to string columns
        df_raw[col] = df_raw[col].str.strip()

train, test = train_test_split(df_raw, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)


X_test, y_test, encoder_test, lb_test = process_data(
    test, categorical_features=cat_features, label='salary', encoder=encoder, lb=lb, training=False
)

#print(X_train.shape)

# Train and save a model.
model = train_model(X_train, y_train)
with open(DATA_DIR + "/../model/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open(DATA_DIR + "/../model/encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

with open(DATA_DIR + "/../model/lb.pkl", "wb") as f:
    pickle.dump(lb, f)

#test reading the model back in
with open(DATA_DIR + "/../model/model.pkl", "rb") as f:
    model2 = pickle.load(f)


preds = inference(model2, X_test)

precision, recall, fbeta = compute_model_metrics(y_test, preds)

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F-beta:    {fbeta:.4f}")
accuracy = np.mean(np.array(y_test) == np.array(preds))
print(f"Accuracy: {accuracy:.4f}")