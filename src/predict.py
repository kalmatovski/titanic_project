import pandas as pd
import joblib


def main():
    model_path = "models/model.joblib"

    model = joblib.load(model_path)

    
    passenger = {
    "Pclass": [1],
    "Sex": ["female"],
    "Age": [30],
    "Fare": [100],
    "Embarked": ["C"]
}


    passenger_df = pd.DataFrame(passenger)

    prediction = model.predict(passenger_df)
    probabilities = model.predict_proba(passenger_df)

    survival_probability = probabilities[0][1]

    print("Passenger data:")
    print(passenger_df)
    print()

    print(f"Prediction: {prediction[0]}")
    print(f"Survival probability: {survival_probability:.2%}")

    if prediction[0] == 1:
        print("Result: Survived")
    else:
        print("Result: Did not survive")


if __name__ == "__main__":
    main()