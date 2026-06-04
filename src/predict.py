import pandas as pd
import joblib

MODEL_PATH = "models/model.joblib"

def load_model(model_path):
    model = joblib.load(model_path)
    return model

def create_passenger():
    #Create example passenger data for prediction

    passenger = {
        
    "Pclass" : [1],
    "Sex": ["female"],
    "Age": [30],
    "Fare": [100],
    "Embarked": ["C"]

    }

    passenger_df = pd.DataFrame(passenger)
    return passenger_df

def make_prediction(model,passenger_df):
    #Make prediction and return predicted class and probabilities
    prediction = model.predict(passenger_df)
    probabilities = model.predict_proba(passenger_df)

    predicted_class = prediction[0]
    survival_probability = probabilities[0][1]

    return predicted_class, survival_probability

def print_result(passenger_df, predicted_class,survival_probability):
    #Print passenger data and prediction result 
    print("Passenger Data:")
    print(passenger_df)
    print()

    print(f"Prediction: {predicted_class}")
    print(f"Survival probability: {survival_probability:.2%}")
    print()

    if predicted_class == 1:
        print("Survived yahoo")
    else:
        print("Sorry buddy, u gon die")

def main():
    model = load_model(MODEL_PATH)

    passenger_df = create_passenger()

    predicted_class, survival_probability = make_prediction(model, passenger_df)

    print_result(passenger_df,predicted_class,survival_probability)

if __name__ == "__main__":
    main()