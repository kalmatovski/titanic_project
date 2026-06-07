# Titanic Survival Prediction

This project predicts whether a passenger survived the Titanic disaster using a machine learning model.

The main goal of this project is not only to train a model, but also to build a clean and structured ML project using professional tools from scikit-learn.

---

## Project Overview

This is a binary classification project based on the Titanic dataset.

The target variable is:

- `Survived`
  - `0` = Did not survive
  - `1` = Survived

The project includes:

- data loading
- feature selection
- preprocessing with sklearn Pipeline
- handling missing values
- encoding categorical features
- model training
- model evaluation
- model saving with joblib
- prediction on new passenger data

---

## Project Structure

```text
titanic_project/
│
├── data/
│   └── train.csv
│
├── models/
│   └── .gitkeep
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Features Used

The model uses the following features:

```text
Pclass
Sex
Age
Fare
Embarked
```

### Feature Types

Numerical features:

```text
Age
Fare
```

Categorical features:

```text
Sex
Embarked
```

---

## Model

The current model is:

```text
Logistic Regression
```

The model is wrapped inside an sklearn Pipeline together with preprocessing steps.

---

## Preprocessing

The project uses sklearn preprocessing tools.

### Numerical Features

For numerical columns:

```text
Age
Fare
```

Missing values are filled using:

```python
SimpleImputer(strategy="median")
```

Median is used because it is more robust to outliers than mean.

After missing value imputation, numerical features are scaled using:

```python
StandardScaler()
```

The numerical preprocessing pipeline:

```python
SimpleImputer(strategy="median")
↓
StandardScaler()
```

StandardScaler helps put numerical features on a similar scale, which is useful for models such as Logistic Regression.

### Categorical Features

For categorical columns:

```text
Sex
Embarked
```

Missing values are filled using:

```python
SimpleImputer(strategy="most_frequent")
```

Categorical values are encoded using:

```python
OneHotEncoder(drop="first", handle_unknown="ignore")
```

---

## Pipeline

The project uses an sklearn Pipeline.

The pipeline contains:

```text
preprocessing + model
```

The preprocessing step contains a ColumnTransformer.

The full workflow looks like this:

```text
Raw data
   ↓
ColumnTransformer
   ├── numerical preprocessing
   └── categorical preprocessing
   ↓
Logistic Regression
   ↓
Prediction
```

Using Pipeline helps keep training and prediction consistent.

---

## Why ColumnTransformer?

The Titanic dataset has different types of columns.

Numerical columns and categorical columns need different preprocessing.

For example:

```text
Age, Fare → numerical preprocessing
Sex, Embarked → categorical preprocessing
```

ColumnTransformer allows applying different preprocessing steps to different columns inside one object.

---

## Training

To train the model, run:

```bash
python src/train.py
```

The training script does the following:

1. Loads the dataset from `data/train.csv`
2. Selects features and target
3. Splits data into train and test sets
4. Builds the sklearn Pipeline
5. Trains Logistic Regression
6. Evaluates the model
7. Saves the trained Pipeline to `models/model.joblib`

---

## Model Evaluation

The model is evaluated using:

- accuracy
- precision
- recall
- f1-score
- support
- ROC-AUC

Current result:

```text
Accuracy: 0.7765
ROC-AUC: 0.7966
```

Classification report:

```text
              precision    recall  f1-score   support

           0       0.80      0.85      0.82       110
           1       0.73      0.67      0.70        69

    accuracy                           0.78       179
   macro avg       0.77      0.76      0.76       179
weighted avg       0.77      0.78      0.77       179
```

---

## Evaluation Notes

The model performs better on class `0` than class `1`.

Class `0` means:

```text
Did not survive
```

Class `1` means:

```text
Survived
```

Recall for class `1` is lower, which means the model misses some passengers who actually survived.

This may happen because the dataset is imbalanced and there are fewer examples of survivors.

---

## Prediction

After training, the model is saved as:

```text
models/model.joblib
```

To make a prediction, run:

```bash
python src/predict.py
```

The prediction script does the following:

1. Loads the saved Pipeline from `models/model.joblib`
2. Creates example passenger data
3. Makes a prediction
4. Shows survival probability

Example passenger:

```text
Pclass: 1
Sex: female
Age: 30
Fare: 100
Embarked: C
```

Example output:

```text
Passenger Data:
   Pclass     Sex  Age  Fare Embarked
0       1  female   30   100        C

Prediction: 1
Survival probability: 90.01%
Result: Survived
```

---

## Predict vs Predict Proba

The project uses both:

```python
model.predict()
```

and:

```python
model.predict_proba()
```

### `predict()`

Returns the predicted class:

```text
0 or 1
```

### `predict_proba()`

Returns probabilities for both classes:

```text
[probability_class_0, probability_class_1]
```

Example:

```text
[0.10, 0.90]
```

This means:

```text
10% probability of class 0
90% probability of class 1
```

---

## Saved Model

The trained model is saved using joblib.

```python
joblib.dump(model, "models/model.joblib")
```

Important:

The saved file contains the full Pipeline, not only Logistic Regression.

It includes:

- SimpleImputer
- OneHotEncoder
- ColumnTransformer
- LogisticRegression

This means `predict.py` does not need to repeat preprocessing manually.

---

## How to Install

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Requirements

```text
pandas
scikit-learn
joblib
```

---

## How to Run

### 1. Train the model

```bash
python src/train.py
```

### 2. Make a prediction

```bash
python src/predict.py
```

---

## What I Learned

In this project, I learned how to:

- structure a basic machine learning project
- separate training and prediction scripts
- use sklearn Pipeline
- use ColumnTransformer
- handle missing values with SimpleImputer
- encode categorical features with OneHotEncoder
- train Logistic Regression
- evaluate a classification model
- understand accuracy, precision, recall, f1-score, and support
- save and load a trained model using joblib
- use predict and predict_proba
- refactor code into smaller functions

---

## Current Status

Completed:

- Professional project structure
- Baseline Titanic model
- sklearn Pipeline
- ColumnTransformer
- Missing value handling
- Categorical encoding
- Model evaluation
- Model saving
- Prediction script
- Refactored train.py
- Refactored predict.py

---

## Next Steps

Planned improvements:

- Add command-line arguments for custom passenger input
- Add batch prediction from CSV file
- Add confusion matrix visualization
- Try another model such as Random Forest
- Compare model performance
- Improve README with more examples
- Add more detailed project documentation
- Add ROC-AUC evaluation
- Add confusion matrix visualization
- Add StandardScaler for numerical features if needed
- Try Random Forest and compare with Logistic

---

## Author

Created as part of my Machine Learning learning journey.

Goal:

```text
Build strong ML fundamentals and prepare for ML internship opportunities.
```
