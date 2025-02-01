from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# Load your trained model
model = joblib.load('loan_default_model.pkl')

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/defaultPrediction_page')
def defaultPrediction_page():
    return render_template('defaultPrediction.html')

@app.route('/RiskAnalysis_page')
def RiskAnalysis_page():
    return render_template('RiskAnalysis.html')

@app.route('/LoanApplication_page')
def LoanApplication_page():
    return render_template('LoanApplication.html')

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
    
    prediction_proba = model.predict_proba(features_df)

    # Calculate risk probability and applicant credibility score
    risk_probability = prediction_proba[0][1]  # Probability of default
    credibility_score = (1 - risk_probability) * 100  # Credibility score out of 100

    # Render result template
    return render_template('results.html', prediction=prediction[0], risk_probability=risk_probability, credibility_score=credibility_score)

@app.route('/riskAnalysis_page')
def riskAnalysis_page():
    # Read data from CSV
    df = pd.read_csv('data.csv')

    # Calculate correlations for the heatmap
    correlation_matrix = df[['Default', 'InterestRate', 'LoanAmount', 'NumCreditLines', 'DTIRatio', 'MaritalStatus', 'LoanTerm']].corr().values.tolist()

    # Prepare the data for the scatter plot
    X = df.drop('Default', axis=1)
    y = df['Default']

    # Create a pipeline that fits the model
    model = Pipeline(steps=[
        ('classifier', DecisionTreeClassifier())
    ])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train your model
    model.fit(X_train, y_train)

    # Get default probabilities
    default_probabilities = model.predict_proba(X_test)[:, 1]

    # Get the corresponding Loan Amounts
    loan_amounts = X_test['LoanAmount'].values.tolist()

    # Prepare scatter plot data
    scatter_plot_data = [{'x': loan_amounts[i], 'y': default_probabilities[i]} for i in range(len(loan_amounts))]

    return render_template('RiskAnalysis.html', correlation_matrix=correlation_matrix, scatter_plot_data=scatter_plot_data)

if __name__ == '__main__':
    app.run(debug=True)