import pandas as pd
import joblib

def main():
    model_path = 'models/model.joblib'

    #1 Load Model 
    model = joblib.load(model_path)

    #Create new data 
    data = {
        'Pclass':[3],
        'Sex' : ['female'],
        'Age' : [30],
        'Fare' : [100],
        'Embarked' : ['S'],
    }

    df = pd.DataFrame(data)

    # Predict

    prediction = model.predict(df)

    print("Prediction: ", prediction[0])

    if prediction[0] == 1:
        print("Survived")
    else:
        print("Sorry buddy")
    
if __name__ == "__main__":
    main()