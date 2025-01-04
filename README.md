# HealthCostPrediction

This project provides a Flask-based backend API for predicting insurance premiums using a machine learning model. The frontend collects user inputs and sends them to the backend for prediction.

# Features

Predict insurance premiums based on user inputs such as health metrics and habits.

Frontend created with HTML, CSS, and JavaScript.

BMI and age categories are dynamically calculated from user inputs.

Predictions are returned as nearest integers with the INR (₹) symbol.

# File Structure

app.py: Flask backend with the prediction API.

ml_model.pkl: Pre-trained machine learning model used for predictions.

frontend.html: HTML file for the user interface.

README.md: Documentation for the project.

# Prerequisites

Python 3.7+

Flask

Pandas

Pickle (for loading the ML model)

Setup Instructions

Clone the repository:

git clone <repository-url>
cd <repository-directory>

# Install dependencies:

pip install flask pandas

Place the ML model file:
Ensure the ml_model.pkl file is in the project directory.

# Run the Flask application:

python app.py

The application will run locally on http://127.0.0.1:5000.

Open the Frontend:
Open frontend.html in any web browser.

API Endpoint

POST /predict

# Description:
Receives user data in JSON format, predicts the insurance premium, and returns the result.

# Input JSON Schema:

{
    "Diabetes": 1,
    "BloodPressureProblems": 0,
    "AnyChronicDiseases": 1,
    "HistoryOfCancerInFamily": 0,
    "NumberOfMajorSurgeries": 2,
    "Smoker": 1,
    "Drinking": 0,
    "Age_cat": 1,
    "BMI_class": 2
}

# Output JSON Schema:

{
    "prediction": "₹5000"
}

# Frontend Usage

Fill out the form fields including:

Yes/No dropdowns for health-related questions.

Numerical inputs for age, height, weight, and number of surgeries.

Click the "Predict" button.

The result will display the predicted insurance premium.

# Notes

Age category (Age_cat) and BMI category (BMI_class) are calculated automatically on the frontend based on age, height, and weight inputs.

Ensure the Flask server is running before using the frontend.

Future Enhancements

Add validation for input data in the frontend.

Provide more detailed error messages.

Deploy the application to a cloud platform for public access.
