import pandas as pd 
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def main():
    data_path = 'data/train.csv'
    
    df = pd.read_csv(data_path)
    print(df.isnull().sum())

    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    target = 'Survived'

    X = df[features]
    y = df[target]
    
    num_features = ["Age", "Fare"]
    cat_features = ["Sex", "Embarked"]

    #Preprocessing

    num_transformer = SimpleImputer(strategy="median")

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy='most_frequent')),
        ("encoder", OneHotEncoder(drop="first"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features)
        ]
    )

    # Full pipeline

    model = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

   
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

    model.fit(X_train,y_train)
    model_path = 'models/model.joblib'
    joblib.dump(model, model_path)

    print(f"Model saved to {model_path}")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")

    print("Classification Report")
    print(classification_report(y_test,y_pred))

if __name__ == "__main__":
    main()