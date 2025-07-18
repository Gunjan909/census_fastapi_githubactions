## Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This project contains a Random Forest classifier trained to predict whether an individual's income exceeds $50K/year based on census data. The model leverages several demographic and socio-economic features such as workclass, education, marital status, occupation, relationship, race, sex, and native country.

The model training pipeline includes:

    Data cleaning and whitespace trimming

    Categorical feature encoding with OneHotEncoder

    Label binarization for the target variable salary

    Model training using sklearn.ensemble.RandomForestClassifier

    Saving and loading the trained model with pickle

Evaluation is performed on a held-out test set with standard classification metrics.

## Intended Use

This model is intended to be used for income classification/prediction tasks based on structured census-like demographic data. It could serve applications in socioeconomic research, targeted marketing, or policy-making scenarios that require income level estimation.

It is designed for binary classification of income level as either <=50K or >50K annually.

Note that as current salaries generally are higher than from the time of this census data, that this model may need retraining/updating for use on current data.

## Training Data

The training data consists of a subset of the U.S. Census dataset (provided with Udacity) with records including the following features:

    workclass

    education

    marital-status

    occupation

    relationship

    race

    sex

    native-country

The target variable is salary, binarized into 0 (<=50K) and 1 (>50K).

The data preprocessing pipeline handles:

    Removal of whitespace in column names and string values

    One-hot encoding of categorical variables

    Label binarization of the target

20% of the original dataset is split off as a test set for evaluation.

## Evaluation Data

Evaluation is performed on the held-out 20% test data, processed with the same encoders as the training set. Additionally, model performance is sliced by different feature subsets (e.g., native-country categories) to assess performance disparities across subgroups.

## Metrics

The model is evaluated with the following metrics:

    Precision

    Recall

    F-beta score (with beta=1, harmonic mean of precision and recall)

    Accuracy

Sample performance metrics printed during training are:

Precision: 0.7459
Recall:    0.6149
F-beta:    0.6741
Accuracy: 0.8571

(Exact values depend on training runs and data splits.)

Performance is also evaluated per-slice on categorical subgroups (e.g., by native-country) to monitor fairness and bias.

## Ethical Considerations

    The model reflects biases present in the underlying census dataset, including societal and demographic disparities.

    Predictions may reinforce existing inequalities if used uncritically in decision-making systems.

    Sensitive features such as race, sex, and native-country are included as input, which might raise concerns about fairness and discrimination.

    It is recommended to evaluate fairness thoroughly before deploying in real-world contexts, and to consider mitigation strategies if biases are detected.

## Caveats and Recommendations

    The model depends heavily on data preprocessing consistency; encoders used during training must be reused for inference.

    Feature selection is fixed; adding or removing features requires retraining.

    The model is only as good as the data quality and representativeness of the training set.

    Manual inspection of model outputs across demographic slices is recommended to detect unintended biases.

    For critical applications, consider additional fairness-aware modeling and explainability methods.

    The model currently lacks explicit support for continuous feature scaling; adding normalization or feature engineering could improve performance.

    Focus on ongoing monitoring and updating as underlying population distributions change over time.

