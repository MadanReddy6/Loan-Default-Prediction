
# from flask import Flask, render_template

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load your trained model
model = joblib.load('loan_default_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/defaultPrediction_page')
def defaultPrediction_page():
    return render_template('defaultPrediction.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data
    form_data = request.form
    # Dynamically extract all form fields
    features = {key: [value] for key, value in form_data.items()}
    print(features)
    features_df = pd.DataFrame(features)  # Convert to DataFrame
    print(features_df)
    # Ensure all columns are present
    expected_columns = ['Age', 'Income', 'LoanAmount', 'LoanTerm', 'InterestRate', 'CreditScore', 
                        'EmploymentType', 'MonthsEmployed', 'NumCreditLines', 'DTIRatio', 
                        'MaritalStatus', 'Education', 'HasDependents', 'HasMortgage', 
                        'LoanPurpose', 'HasCoSigner']
    for col in expected_columns:
        if col not in features_df.columns:
            features_df[col] = 0  # or some default value

    # Make prediction
    prediction = model.predict(features_df)
    print(prediction)

    # Render result template
    return render_template('results.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)