from flask import Flask, render_template, request,jsonify
import joblib
import numpy as np
import pandas as pd
import mysql.connector

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# connecting database
try:
    conn = mysql.connector.connect(
        host='localhost',
        user="root",
        password='Reddy@656',
        database='loan_db'
    )
    cursor = conn.cursor()

except mysql.connector.Error as e:
    print('Error connecting to MySQL database:', e)

# Load your trained model
model = joblib.load('loan_default_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/contactus_page')
def contactus_page():
    return render_template('contactus.html')

@app.route('/aboutus_page')
def aboutus_page():
    return render_template('about.html')

@app.route('/personalLoan_page')
def personalLoan_page():
    return render_template('personalloan.html')

@app.route('/autoLoan_page')
def autoLoan_page():
    return render_template('autoloan.html')

@app.route('/homeLoan_page')
def homeLoan_page():
    return render_template('homeloan.html')

@app.route('/defaultPrediction_page')
def defaultPrediction_page():
    return render_template('defaultPrediction.html')

@app.route('/educationLoan_page')
def educationLoan_page():
    return render_template('educationloan.html')

@app.route('/businessLoan_page')
def businessLoan_page():
    return render_template('businessloan.html')

# @app.route('/RiskAnalysis_page')
# def RiskAnalysis_page():
#     return render_template('RiskAnalysis.html')

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

@app.route('/RiskAnalysis_page')
def RiskAnalysis_page():
    # Establish the connection to the database
    
    cursor = conn.cursor()
 
    # cursor.execute('''SELECT LoanTerm, MaritalStatus, DTIRatio, NumCreditLines, LoanAmount, InterestRate, `Default`  FROM loan_data''')
    # result = cursor.fetchall()
    
    cursor.execute('''SELECT LoanTerm, MaritalStatus, DTIRatio, NumCreditLines, LoanAmount, InterestRate, `Default` FROM loan_data''')
    data = cursor.fetchall()

    # Send data in a structure suitable for chart
   
    
    return data

@app.route('/riskAnalysis_data')
def riskAnalysis_data():
    cursor = conn.cursor()
    cursor.execute('''SELECT LoanTerm, MaritalStatus, DTIRatio, NumCreditLines, LoanAmount, InterestRate, `Default` FROM loan_data''')
    data = cursor.fetchall()
    
    # Convert data to DataFrame
    df = pd.DataFrame(data, columns=['LoanTerm', 'MaritalStatus', 'DTIRatio', 'NumCreditLines', 'LoanAmount', 'InterestRate', 'Default'])
    
    # Calculate correlation matrix
    correlation_matrix = df.corr().to_dict()
    
    return jsonify(correlation_matrix)

if __name__ == '__main__':
    app.run(debug=True)