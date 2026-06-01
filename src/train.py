import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
def main():
    data_path = 'data/train.csv'

    df = pd.read_csv(data_path)

    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    target = "Survived"

    X = df[features]
    y = df[target]

    X = X.copy()

    X['Age'] = X["Age"].fillna(X["Age"].median())
    X['Fare'] = X['Fare'].fillna(X["Fare"].median())

    X = pd.get_dummies(X,columns=["Sex", "Embarked"], drop_first=True)

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

    model = LogisticRegression(max_iter=1000)
    
    model.fit(X_train,y_train)
    
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test,y_pred)

    print(f"Accuracy : {accuracy:.4f}")

    print("Classification Report: ")
    print(classification_report(y_test,y_pred))

if __name__ == "__main__":
    main()