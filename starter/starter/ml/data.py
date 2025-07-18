import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def process_data(
    X, categorical_features=[], label=None, training=True, encoder=None, lb=None
):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    #X_continuous = X.drop(*[categorical_features], axis=1)
    X_continuous = X.drop(columns=categorical_features)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb






def get_processed_column_names(X, categorical_features, encoder, label='salary'):
    """
    Get the column names of the processed feature array as output by `process_data`.

    Parameters
    ----------
    X : pd.DataFrame
        Original input DataFrame (before processing).
    categorical_features : list[str]
        List of categorical feature names used in one-hot encoding.
    encoder : sklearn.preprocessing.OneHotEncoder
        Fitted encoder from `process_data`.

    Returns
    -------
    feature_names : list[str]
        Ordered list of column names after processing.
    """
    X = X.drop([label], axis=1)

    # Continuous columns are all columns except the categorical ones; also drop salary
    continuous_columns = X.drop(columns=categorical_features).columns.tolist()

    # Get one-hot encoded feature names
    categorical_encoded_names = encoder.get_feature_names_out(categorical_features).tolist()

    # Final order: continuous first, then one-hot encoded columns
    return continuous_columns + categorical_encoded_names

