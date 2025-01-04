from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load the machine learning model
MODEL_PATH = 'ml_model.pkl'
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return "<h1>Health Insurance Cost Prediction</h1>"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.json
        
        # Define expected input features and validate
        required_features = [
            "Diabetes", "BloodPressureProblems", "AnyChronicDiseases", 
            "HistoryOfCancerInFamily", "NumberOfMajorSurgeries", 
            "Smoker", "Drinking", "Age_cat", "BMI_class"
        ]

        missing_features = [f for f in required_features if f not in data]
        if missing_features:
            return jsonify({"error": f"Missing features: {', '.join(missing_features)}"}), 400

        # Convert JSON data to DataFrame
        input_data = pd.DataFrame([data])

        # Ensure correct data types for the model
        for feature in ["Diabetes", "BloodPressureProblems", "AnyChronicDiseases", "HistoryOfCancerInFamily", "Smoker", "Drinking", "Age_cat", "BMI_class"]:
            input_data[feature] = input_data[feature].astype(int)
        input_data["NumberOfMajorSurgeries"] = input_data["NumberOfMajorSurgeries"].astype(float)

        # Make prediction
        prediction = model.predict(input_data)

        # Round the prediction to the nearest integer and add INR symbol
        rounded_prediction = round(prediction[0])
        formatted_prediction = f"\u20B9{rounded_prediction}"

        # Return the prediction as JSON
        return jsonify({"prediction": formatted_prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)