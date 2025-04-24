from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import mysql.connector
from datetime import datetime
from flask_caching import Cache
import os
import smtplib
from flask import Flask, request, jsonify
from email.message import EmailMessage
import seaborn as sns
import matplotlib.pyplot as plt



# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.pipeline import Pipeline



app = Flask(__name__)

Cache = Cache(app, config={'CACHE_TYPE': 'simple'})

CACHE_TIMEOUT = 3600  # 1 hour cache duration

# connecting database
try:
    conn = mysql.connector.connect(
        host='localhost',
        user="root",
        password='Reddy@656',
        database='loan_db',
        
    )
    cursor = conn.cursor()

except mysql.connector.Error as e:
    print('Error connecting to MySQL database:', e)
    conn = None

# Load  trained model
# model = joblib.load('loan_default_model.pkl')
model = joblib.load('loan_default_model.pkl')

# Loading your trained personal loan model
personal_loan_model = joblib.load('personal_loan_default_model.pkl')

# Loading the loan risk classifier model
loan_risk_classifier = joblib.load('loan_risk_classifier_decision_tree.pkl')

# Loading the XG boost model
loan_default_xgb_model = joblib.load('loan_default_xgb_model.pkl')

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/contactus_page')
def contactus_page():
    return render_template('Contactus.html')

@app.route('/Dashboard_page')
def Dashboard_page():
    return render_template('Dashboard.html')

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

@app.route('/LoanApplication_page')
def LoanApplication_page():
    return render_template('LoanApplication.html')

@app.route('/calculator_page')
def EMIcalculator_page():
    return render_template('EMIcalculator.html')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Validate and process form data
        required_fields = {
            'Age': int,
            'Income': float,
            'LoanAmount': float,
            'CreditScore': int,
            'MonthsEmployed': int,
            'NumCreditLines': int,
            'LoanTerm': int,
            'DTIRatio': float,
            'Education': int,
            'EmploymentType': int,
            'MaritalStatus': int,
            'HasMortgage': int,
            'HasDependents': int,
            'LoanPurpose': int,
            'HasCoSigner': int
        }

        processed_data = {}
        for field, field_type in required_fields.items():
            if field not in request.form or not request.form[field].strip():
                return render_template('LoanApplication.html', 
                                    error=f"Missing required field: {field}")
            
            try:
                processed_data[field] = field_type(request.form[field])
            except ValueError:
                return render_template('LoanApplication.html',
                                    error=f"Invalid value for {field}. Expected {field_type.__name__}")

        # 2. Additional validation
        if not (300 <= processed_data['CreditScore'] <= 850):
            return render_template('LoanApplication.html',
                                error="Credit score must be between 300 and 850")
        
        if not (0 <= processed_data['DTIRatio'] <= 100):
            return render_template('LoanApplication.html',
                                error="DTI Ratio must be between 0 and 100")

        # 3. Prepare features for prediction with fixed interest rate
        features = {
            'Age': [processed_data['Age']],
            'Income': [processed_data['Income']],
            'LoanAmount': [processed_data['LoanAmount']],
            'CreditScore': [processed_data['CreditScore']],
            'MonthsEmployed': [processed_data['MonthsEmployed']],
            'NumCreditLines': [processed_data['NumCreditLines']],
            'InterestRate': [24.0],  # Fixed interest rate from your form
            'LoanTerm': [processed_data['LoanTerm']],
            'DTIRatio': [processed_data['DTIRatio']],
            'Education': [processed_data['Education']],
            'EmploymentType': [processed_data['EmploymentType']],
            'MaritalStatus': [processed_data['MaritalStatus']],
            'HasMortgage': [processed_data['HasMortgage']],
            'HasDependents': [processed_data['HasDependents']],
            'LoanPurpose': [processed_data['LoanPurpose']],
            'HasCoSigner': [processed_data['HasCoSigner']]
        }

        features_df = pd.DataFrame(features)

        # 4. Make prediction
        try:
            prediction = int(loan_default_xgb_model.predict(features_df)[0])
            prediction_proba = float(model.predict_proba(features_df)[0][1])
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return render_template('LoanApplication.html',
                                error="Error making prediction. Please try again.")

        # 5. Calculate risk metrics
        credit_score = processed_data['CreditScore']
        
        
        
        # Determine risk category based on credit score
        risk_category = (
            "Very Low Risk" if credit_score >= 800 else
            "Low Risk" if credit_score >= 750 else
            "Moderate Risk" if credit_score >= 700 else
            "High Risk" if credit_score >= 650 else
            "Very High Risk"
        )
        
        # Calculate risk probability aligned with category
        def calculate_risk_probability(score, base_prob):
            if score >= 800:    # Very Low Risk
                return min(10, base_prob * 100)
            elif score >= 750:  # Low Risk
                return max(10, min(30, base_prob * 100))
            elif score >= 700:  # Moderate Risk
                return max(30, min(60, base_prob * 100))
            elif score >= 650:  # High Risk
                return max(60, min(80, base_prob * 100))
            else:               # Very High Risk
                return max(80, min(100, base_prob * 100))
        
        risk_probability = round(calculate_risk_probability(credit_score, prediction_proba), 2)
        try:
            
            cursor = conn.cursor()

            # Map numeric values to string representations
            education_map = {
                0: "High School",
                1: "Bachelor's",
                2: "Master's",
                3: "PhD"
            }
            
            employment_map = {
                0: "Unemployed",
                1: "Self-employed",
                2: "Employed",
                3: "Retired"
            }
            
            marital_map = {
                0: "Single",
                1: "Married",
                2: "Divorced"
            }
            
            purpose_map = {
                0: "Personal",
                1: "Education",
                2: "Medical",
                3: "Business",
                4: "Home",
                5: "Auto"
            }

            # Insert application data
            insert_query = """
            INSERT INTO LoanData (
                Age, Income, LoanAmount, CreditScore, MonthsEmployed,
                NumCreditLines, InterestRate, LoanTerm, DTIRatio,
                Education, EmploymentType, MaritalStatus, HasMortgage, 
                HasDependents, LoanPurpose, HasCoSigner, `Default`
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                processed_data['Age'],
                float(processed_data['Income']),
                float(processed_data['LoanAmount']),
                processed_data['CreditScore'],
                processed_data['MonthsEmployed'],
                processed_data['NumCreditLines'],
                24.0,  # Fixed interest rate from your form
                processed_data['LoanTerm'],
                float(processed_data['DTIRatio']),
                education_map.get(processed_data['Education'], "Unknown"),
                employment_map.get(processed_data['EmploymentType'], "Unknown"),
                marital_map.get(processed_data['MaritalStatus'], "Unknown"),
                "Yes" if processed_data['HasMortgage'] else "No",
                "Yes" if processed_data['HasDependents'] else "No",
                purpose_map.get(processed_data['LoanPurpose'], "Other"),
                "Yes" if processed_data['HasCoSigner'] else "No",
                "Yes" if prediction else "No"  # Convert prediction to Yes/No
            )

            cursor.execute(insert_query, values)
            conn.commit()
            
            # Get the auto-generated LoanID
            loan_id = cursor.lastrowid

        except Exception as e:
            print(f"Database error: {str(e)}")
            conn.rollback()
            return render_template('LoanApplication.html',
                                error="Error saving application. Please try again.")
        

        # 7. Return results with LoanID
        return render_template('results.html',
                            prediction=prediction,
                            risk_probability=risk_probability,
                            risk_category=risk_category,
                            credit_score=credit_score,
                            loan_amount=processed_data['LoanAmount'],
                            income=processed_data['Income'],
                            dti_ratio=processed_data['DTIRatio'],
                            loan_term=processed_data['LoanTerm'],
                            loan_id=loan_id)  # Include LoanID in results

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return render_template('LoanApplication.html',
                            error="An unexpected error occurred. Please try again.")
        
@app.route('/predict_personal_loan', methods=['POST'])
def predict_personal_loan():
    try:
        # Extract form data
        form_data = request.get_json()
        pannumber = form_data.get('pannumber')

        # Check if PAN number already exists
        check_query = "SELECT COUNT(*) FROM PersonalLoanData WHERE pannumber = %s"
        cursor.execute(check_query, (pannumber,))
        if cursor.fetchone()[0] > 0:
            return jsonify({"error": "PAN number already exists"}), 400

        # Validate required financial fields
        required_financial_fields = ['monthlyIncome', 'loanAmount', 'tenure', 
                                   'interestRate', 'CreditScore', 'NumCreditLines']
        missing_fields = [field for field in required_financial_fields if field not in form_data]
        if missing_fields:
            return jsonify({"error": f"Missing required financial fields: {', '.join(missing_fields)}"}), 400

        # Calculate DTI Ratio - now mandatory to provide either DTI or monthlyDebt
        if 'DTIRatio' not in form_data:
            if 'monthlyDebt' not in form_data:
                return jsonify({
                    "error": "Must provide either 'DTIRatio' or 'monthlyDebt'",
                    "solution": "Add either field to calculate debt-to-income ratio"
                }), 400
            
            try:
                monthly_income = float(form_data['monthlyIncome'])
                monthly_debt = float(form_data['monthlyDebt'])
                if monthly_income <= 0:
                    return jsonify({"error": "Monthly income must be greater than 0"}), 400
                form_data['DTIRatio'] = (monthly_debt / monthly_income) * 100
            except (ValueError, TypeError) as e:
                return jsonify({"error": f"Invalid financial values: {str(e)}"}), 400

        # Map form fields to model's expected feature names
        feature_mapping = {
            'monthlyIncome': 'Income',
            'loanAmount': 'LoanAmount',
            'interestRate': 'InterestRate',
            'tenure': 'LoanTerm',
            'CreditScore': 'CreditScore',
            'NumCreditLines': 'NumCreditLines',
            'DTIRatio': 'DTIRatio'
        }

        # Prepare features with correct names
        features = {}
        for form_field, model_field in feature_mapping.items():
            try:
                features[model_field] = [float(form_data[form_field])]
            except KeyError:
                return jsonify({"error": f"Missing required field: {form_field}"}), 400
            except (ValueError, TypeError):
                return jsonify({"error": f"Invalid value for {form_field}"}), 400

        # Calculate age from date of birth
        try:
            dob = datetime.strptime(form_data['dob'], '%Y-%m-%d')
            age = datetime.today().year - dob.year - ((datetime.today().month, datetime.today().day) < (dob.month, dob.day))
            features['Age'] = [age]
        except (KeyError, ValueError):
            return jsonify({"error": "Invalid or missing date of birth (YYYY-MM-DD format required)"}), 400

        # Create DataFrame with correct column order
        expected_columns = ['Age', 'Income', 'LoanAmount', 'CreditScore', 
                          'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']
        features_df = pd.DataFrame(features)[expected_columns]

        # Validate loan amount
        if features['LoanAmount'][0] > 20 * features['Income'][0]:
            return jsonify({"error": "Loan amount cannot exceed 20 times your monthly income"}), 400

        # Make prediction
        prediction = personal_loan_model.predict(features_df)
        prediction_proba = personal_loan_model.predict_proba(features_df)

        # Calculate risk metrics
        credit_score = int(form_data['CreditScore'])
        credibility_score = credit_score
        risk_probability = round(prediction_proba[0][1] * 100, 2)
        
        risk_category = (
            "Very Low Risk" if credit_score >= 800 else
            "Low Risk" if credit_score >= 750 else
            "Moderate Risk" if credit_score >= 700 else
            "High Risk" if credit_score >= 650 else
            "Very High Risk"
        )
        
        # Adjust probability to category range
        risk_probability = {
            "Very Low Risk": min(10, risk_probability),
            "Low Risk": max(10, min(30, risk_probability)),
            "Moderate Risk": max(30, min(60, risk_probability)),
            "High Risk": max(60, min(80, risk_probability)),
            "Very High Risk": max(80, risk_probability)
        }[risk_category]

        credit_score = credit_score
        print('===============','===============',credibility_score)
        print('===============',risk_probability)
        print('===============',risk_category)
        

        # Modified INSERT query without monthlyDebt
        insert_query = """
        INSERT INTO PersonalLoanData 
        (fullName, pannumber, email, phone, dob, monthlyIncome, employmentType, 
         companyName, loanAmount, tenure, interestRate, CreditScore, `Default`) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (
            form_data['fullName'], pannumber, form_data['email'], form_data['phone'],
            form_data['dob'], features['Income'][0],
            form_data['employmentType'], form_data['companyName'], 
            features['LoanAmount'][0], features['LoanTerm'][0],
            features['InterestRate'][0], credit_score, int(prediction[0]),
           
            
        ))
        conn.commit()

        return jsonify({
            "prediction": int(prediction[0]),
            "risk_probability": risk_probability,
            "risk_category": risk_category,
            "credibility_score": credibility_score,
            "loan_details": {
                "amount": features['LoanAmount'][0],
                "tenure": features['LoanTerm'][0],
                "interest_rate": features['InterestRate'][0],
                "dti_ratio": features['DTIRatio'][0],
                "monthly_income": features['Income'][0]
            }
        })

    except Exception as e:
        conn.rollback()
        print(f"Error: {str(e)}")
        return jsonify({"error": "Loan processing failed. Please check your input data."}), 500
    
# @app.route('/predict_personal_loan', methods=['POST'])
# def predict_personal_loan():
#     try:
#         # 1. Extract and validate form data
#         form_data = request.get_json()
        
#         # Required fields validation
#         required_fields = [
#             'pannumber', 'fullName', 'email', 'phone', 'dob',
#             'monthlyIncome', 'loanAmount', 'tenure', 'interestRate',
#             'CreditScore', 'NumCreditLines', 'employmentType', 'companyName'
#         ]
        
#         missing_fields = [field for field in required_fields if field not in form_data]
#         if missing_fields:
#             return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

#         # 2. Check for existing PAN
#         cursor.execute("SELECT COUNT(*) FROM PersonalLoanData WHERE pannumber = %s", (form_data['pannumber'],))
#         if cursor.fetchone()[0] > 0:
#             return jsonify({"error": "PAN number already exists"}), 400

#         # 3. Calculate DTI Ratio (if not provided)
#         if 'DTIRatio' not in form_data:
#             if 'monthlyDebt' not in form_data:
#                 return jsonify({"error": "Must provide either DTIRatio or monthlyDebt"}), 400
#             try:
#                 monthly_income = float(form_data['monthlyIncome'])
#                 monthly_debt = float(form_data['monthlyDebt'])
#                 form_data['DTIRatio'] = (monthly_debt / monthly_income) * 100 if monthly_income > 0 else 0
#             except (ValueError, ZeroDivisionError):
#                 return jsonify({"error": "Invalid income/debt values"}), 400

#         # 4. Prepare features for prediction
#         features = {
#             'Age': [calculate_age(form_data['dob'])],
#             'Income': [float(form_data['monthlyIncome'])],
#             'LoanAmount': [float(form_data['loanAmount'])],
#             'CreditScore': [int(form_data['CreditScore'])],
#             'NumCreditLines': [int(form_data['NumCreditLines'])],
#             'InterestRate': [float(form_data['interestRate'])],
#             'LoanTerm': [int(form_data['tenure'])],
#             'DTIRatio': [float(form_data['DTIRatio'])]
#         }
        
#         # Validate loan amount
#         if features['LoanAmount'][0] > 20 * features['Income'][0]:
#             return jsonify({"error": "Loan amount exceeds 20x monthly income"}), 400

#         # 5. Make prediction
#         features_df = pd.DataFrame(features)
#         prediction = personal_loan_model.predict(features_df)
#         prediction_proba = personal_loan_model.predict_proba(features_df)

#         # 6. Calculate risk metrics
#         credit_score = int(form_data['CreditScore'])
#         model_risk_prob = prediction_proba[0][1]  # Probability of default
        
#         # Risk category determination
#         if credit_score >= 800:
#             risk_category = "Very Low Risk"
#             prob_range = (0, 10)
#         elif credit_score >= 750:
#             risk_category = "Low Risk"
#             prob_range = (10, 30)
#         elif credit_score >= 700:
#             risk_category = "Moderate Risk"
#             prob_range = (30, 60)
#         elif credit_score >= 650:
#             risk_category = "High Risk"
#             prob_range = (60, 80)
#         else:
#             risk_category = "Very High Risk"
#             prob_range = (80, 100)
        
#         # Calculate blended risk probability
#         model_prob_percent = model_risk_prob * 100
#         score_based_prob = ((850 - credit_score) / (850 - 300)) * 100
#         blended_prob = (model_prob_percent * 0.6) + (score_based_prob * 0.4)
#         risk_probability = round(max(prob_range[0], min(prob_range[1], blended_prob), 2))
#         credibility_score = round(100 - risk_probability, 2)

#         # 7. Store results
#         insert_query = """
#         INSERT INTO PersonalLoanData 
#         (fullName, pannumber, email, phone, dob, monthlyIncome,  employmentType, 
#          companyName, loanAmount, tenure, interestRate, CreditScore, `Default`,
#          RiskCategory, RiskProbability, CredibilityScore, DTIRatio) 
#         VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
#         """
#         cursor.execute(insert_query, (
#             form_data['fullName'], form_data['pannumber'], form_data['email'], form_data['phone'],
#             form_data['dob'], features['Income'][0], 
#             form_data['employmentType'], form_data['companyName'], 
#             features['LoanAmount'][0], features['LoanTerm'][0],
#             features['InterestRate'][0], credit_score, int(prediction[0]),
#             risk_category, risk_probability, credibility_score,
#             features['DTIRatio'][0]
#         ))
#         conn.commit()

#         # 8. Return response
#         return jsonify({
#             "prediction": int(prediction[0]),
#             "risk_assessment": {
#                 "category": risk_category,
#                 "probability": risk_probability,
#                 "credibility_score": credibility_score,
#                 "credit_score": credit_score,
#                 "score_range": f"{credit_score} ({risk_category})"
#             },
#             "loan_details": {
#                 "amount": features['LoanAmount'][0],
#                 "term_months": features['LoanTerm'][0],
#                 "interest_rate": features['InterestRate'][0],
#                 "dti_ratio": features['DTIRatio'][0]
#             }
#          })
#     except ValueError as e:
#         conn.rollback()
#         return jsonify({"error": f"Invalid data format: {str(e)}"}), 400
#     except Exception as e:
#         conn.rollback()
#         print(f"Error: {str(e)}")
#         return jsonify({"error": "Loan processing failed"}), 500

# def calculate_age(dob_str):
#     """Helper function to calculate age from date of birth"""
#     dob = datetime.strptime(dob_str, '%Y-%m-%d')
#     today = datetime.today()
#     return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    
@app.route('/RiskAnalysis_page')

def RiskAnalysis_page():
    return render_template('RiskAnalysis.html')

@app.route('/results')
def results():
    try:
        # Get all possible parameters with proper defaults
        prediction = request.args.get('prediction', '0')
        risk_probability = request.args.get('risk_probability', '0')
        credibility_score = request.args.get('credibility_score')
        credit_score=credibility_score
        # credit_score = request.args.get('credit_score', '0')
        print('credit_score+++++++++++++++++++++++++++++++++++++++++++',credit_score)
        print('credibility_score++++++++++++++++++++++++',credibility_score)
        print('risk_probability++++++++++++++++++++++++',risk_probability)
        print('prediction++++++++++++++++++++++++',prediction)
        
        # Convert to proper types with validation
        try:
            prediction = int(prediction) if prediction != 'undefined' else 0
            risk_probability = float(risk_probability) if risk_probability != 'undefined' else 0.0
            credit_score = int(credit_score) if credit_score != 'undefined' else 0
            
            # Calculate credibility score if not provided or invalid
            try:
                credibility_score = float(credibility_score) if credibility_score and credibility_score != 'undefined' else 100 - risk_probability
            except (ValueError, TypeError):
                credibility_score = 100 - risk_probability
                
        except (ValueError, TypeError):
            raise ValueError("Invalid numeric parameters")

        # Calculate risk category
        risk_category = (
            "Very Low Risk" if credit_score >= 800 else
            "Low Risk" if credit_score >= 750 else
            "Moderate Risk" if credit_score >= 700 else
            "High Risk" if credit_score >= 650 else
            "Very High Risk"
        )

        return render_template('results.html',
            prediction=prediction,
            risk_probability=round(risk_probability, 2),
            credibility_score=round(credibility_score, 2),
            credit_score=credit_score,
            risk_category=risk_category,
            loan_details={
                'amount': 0,
                'term_months': 0,
                'interest_rate': 0,
                'monthly_income': 0,
                'dti_ratio': 0
            }
        )
        
    except Exception as e:
        print(f"Error processing results: {str(e)}")
        return render_template('results.html',
            prediction=0,
            risk_probability=0,
            credibility_score=0,
            credit_score=0,
            risk_category="Unknown",
            loan_details={
                'amount': 0,
                'term_months': 0,
                'interest_rate': 0,
                'monthly_income': 0,
                'dti_ratio': 0
            },
            error_message="Invalid result data received"
        )

# Default admin email
ADMIN_EMAIL = "21svdc2079@svdegreecollege.ac.in"  # Replace with your admin email
ADMIN_PASSWORD = "bcjcejkokwxbjkpl"  # Replace with your email password

@app.route('/contact', methods=['POST'])
def contact():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request, no data received"}), 400

    name = data.get('name')
    email = data.get('email')
    subject = data.get('subject')
    message = data.get('message')

    if not all([name, email, subject, message]):
        return jsonify({"error": "All fields are required"}), 400

    try:
        # Email to admin
        admin_email_message = EmailMessage()
        admin_email_message.set_content(f"Name: {name}\nEmail: {email}\nSubject: {subject}\nMessage: {message}")
        admin_email_message["Subject"] = f"New Contact Form Submission from {name}"
        admin_email_message["From"] = ADMIN_EMAIL
        admin_email_message["To"] = ADMIN_EMAIL

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(ADMIN_EMAIL, ADMIN_PASSWORD)
            server.send_message(admin_email_message)

        # Confirmation email to the user
        user_email_message = EmailMessage()
        user_email_message.set_content(
            "Thank you for reaching out! We've received your message and will get back to you shortly."
        )
        user_email_message["Subject"] = "Confirmation: We've received your message"
        user_email_message["From"] = ADMIN_EMAIL
        user_email_message["To"] = email

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(ADMIN_EMAIL, ADMIN_PASSWORD)
            server.send_message(user_email_message)

        return jsonify({"success": "Your message has been sent successfully!"}), 200

    except Exception as e:
        app.logger.error(f"Failed to send email: {e}")
        return jsonify({"error": "Failed to send email. Please try again later."}), 500



@app.route('/combined_dashboard_data')
@Cache.cached(timeout=CACHE_TIMEOUT, key_prefix='combined_dashboard_data')
def combined_dashboard_data():
    try:
        cursor = conn.cursor()

        # Query for correlation data
        cursor.execute('''SELECT Age, Income, LoanAmount, CreditScore, MonthsEmployed, NumCreditLines, 
                                 InterestRate, LoanTerm, DTIRatio, Education, EmploymentType, MaritalStatus, 
                                 HasMortgage, HasDependents, LoanPurpose, HasCoSigner, `Default` FROM LoanData LIMIT 100000''')
        data = cursor.fetchall()
        
        # Convert data to DataFrame for correlation
        columns = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'NumCreditLines', 
                   'InterestRate', 'LoanTerm', 'DTIRatio', 'Education', 'EmploymentType', 'MaritalStatus', 
                   'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner', 'Default']
        df = pd.DataFrame(data, columns=columns)

        # Convert categorical columns to numeric
        categorical_columns = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 
                              'HasDependents', 'LoanPurpose', 'HasCoSigner', 'Default']
        for column in categorical_columns:
            df[column] = df[column].astype('category').cat.codes

        # Calculate correlation matrix
        correlation_matrix = df.corr()

        # Get top features affecting loan default risk
        top_features = correlation_matrix['Default'].abs().sort_values(ascending=False).index.tolist()

        # Convert correlation matrix to dictionary
        cor = correlation_matrix.to_dict()

        # Query for borrower segmentation data (filter out invalid values)
        cursor.execute('''
            SELECT EmploymentType, COUNT(*) as count 
            FROM LoanData 
            WHERE EmploymentType NOT IN ('3', '1', '0') 
            GROUP BY EmploymentType
        ''')
        segmentation_data = cursor.fetchall()
        employment_types = [row[0] for row in segmentation_data]
        counts = [row[1] for row in segmentation_data]

        # Query for default rates by employment type
        cursor.execute('''
            SELECT EmploymentType, 
                   AVG(`Default`) AS DefaultRate 
            FROM LoanData 
            WHERE EmploymentType NOT IN ('3', '1', '0') 
            GROUP BY EmploymentType
        ''')
        default_rate_data = cursor.fetchall()
        default_rates = [row[1] for row in default_rate_data]

        # Query for default probability by age group
        cursor.execute('''
            SELECT 
                CASE 
                    WHEN Age < 20 THEN 'Under 20'
                    WHEN Age BETWEEN 20 AND 29 THEN '20-29'
                    WHEN Age BETWEEN 30 AND 39 THEN '30-39'
                    WHEN Age BETWEEN 40 AND 49 THEN '40-49'
                    WHEN Age BETWEEN 50 AND 59 THEN '50-59'
                    WHEN Age >= 60 THEN '60 and above'
                END AS AgeGroup,
                AVG(`Default`) AS DefaultProbability
            FROM LoanData
            GROUP BY AgeGroup
            ORDER BY DefaultProbability DESC
        ''')
        age_group_data = cursor.fetchall()
        age_groups = [row[0] for row in age_group_data]
        default_probabilities = [row[1] for row in age_group_data]

        # Query for credit score distribution
        cursor.execute('''
            SELECT CreditScore
            FROM LoanData
        ''')
        credit_score_data = cursor.fetchall()
        credit_scores = [row[0] for row in credit_score_data]
        
         # NEW: Query to determine credit score threshold
        cursor.execute('''
            SELECT MIN(CreditScore) as min_score_approved
            FROM LoanData
            WHERE `Default` = 0
        ''')
        threshold_data = cursor.fetchone()
        print('threshold_data--------------',threshold_data)
        credit_score_threshold = threshold_data[0] if threshold_data else 600  # Default fallback
        
        # NEW: Query for loan approval rates by income level
        cursor.execute('''
            SELECT 
                CASE 
                    WHEN Income < 30000 THEN 'Under 30K'
                    WHEN Income BETWEEN 30000 AND 49999 THEN '30K-50K'
                    WHEN Income BETWEEN 50000 AND 69999 THEN '50K-70K'
                    WHEN Income BETWEEN 70000 AND 89999 THEN '70K-90K'
                    WHEN Income >= 90000 THEN '90K+'
                END AS IncomeGroup,
                AVG(CASE WHEN `Default` = 0 THEN 1 ELSE 0 END) AS ApprovalRate
            FROM LoanData
            GROUP BY IncomeGroup
            ORDER BY ApprovalRate DESC
        ''')
        income_approval_data = cursor.fetchall()
        income_groups = [row[0] for row in income_approval_data]
        approval_rates = [float(row[1]) * 100 for row in income_approval_data]  # Convert to percentage
        
        # Combine all datasets into a single response
        return jsonify({
            "correlation_data": {
                "correlation_matrix": cor,
                "top_features": top_features
            },
            "borrower_segmentation": {
                "employment_types": employment_types,
                "counts": counts
            },
            "default_rates": {
                "employment_types": employment_types,
                "default_rates": default_rates
            },
            "default_probability_by_age_group": {
                "age_groups": age_groups,
                "default_probabilities": default_probabilities
            },
            "credit_score_distribution": {
                "credit_scores": credit_scores
            },
            "credit_score_threshold": credit_score_threshold,
            "credit_score_analysis": {
                "min_score": int(df['CreditScore'].min()),
                "max_score": int(df['CreditScore'].max()),
                "avg_score": float(df['CreditScore'].mean()),
                "threshold": credit_score_threshold
            },
            "income_vs_approval": {
            "income_groups": income_groups,
            "approval_rates": approval_rates
    }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/check_pan', methods=['GET'])
def check_pan():
    pannumber = request.args.get('pannumber')
    if not pannumber:
        return jsonify({"error": "PAN number is required"}), 400

    try:
        check_query = "SELECT COUNT(*) FROM PersonalLoanData WHERE pannumber = %s"
        cursor.execute(check_query, (pannumber,))
        result = cursor.fetchone()
        return jsonify({"exists": result[0] > 0})
    except mysql.connector.Error as err:
        return jsonify({"error": str(err)}), 500
    
    
# @app.route('/loan_analysis')
# @Cache.cached(timeout=CACHE_TIMEOUT, key_prefix='loan_analysis')
# def loan_analysis():
#     try:
        
#         cursor = conn.cursor(dictionary=True)  # This is crucial - returns dictionaries
#         print('111111111111111111111111')
#         # First verify the table exists
#         cursor.execute("SHOW TABLES LIKE 'LoanData'")
#         if not cursor.fetchone():
#             return jsonify({
#                 'error_message': "LoanData table not found",
#                 'loan_amounts': [],
#                 'default_status': []
#             })
#         print('222222222222222222222222222')
        
#         # Check if we have any data
#         cursor.execute("SELECT COUNT(*) as count FROM LoanData")
#         total_records = cursor.fetchone()['count']  # Now this will work
        
#         if total_records == 0:
#             return jsonify({
#                 'warning': "No loan records found in database",
#                 'loan_amounts': [],
#                 'default_status': []
#             })
#         print('33333333333333333333333333')
#         # Fetch the actual data
#         query = """
#             SELECT LoanAmount, `Default` as DefaultStatus 
#             FROM LoanData
#             WHERE LoanAmount IS NOT NULL
#             AND `Default` IS NOT NULL
#             ORDER BY RAND()
#             limit 125000
            
#         """
#         cursor.execute(query)
#         data = cursor.fetchall()
#         print(f'--------------length if the data{len(data)}')
#         print('44444444444444444444444444')
#         # Process data
#         loan_amounts = []
#         default_status = []
        
#         for row in data:
#             try:
#                 loan_amounts.append(float(row['LoanAmount']))  # Now this will work
#                 default_status.append(int(row['DefaultStatus']))  # And this too
#             except (ValueError, TypeError, KeyError) as e:
#                 print(f"Skipping row due to error: {e}")
#                 continue
        
#         if not loan_amounts:
#             return jsonify({
#                 'warning': "No valid loan data found after processing",
#                 'loan_amounts': [],
#                 'default_status': []
#             })
        
        
#         return jsonify({
#             'loan_amounts': loan_amounts,
#             'default_status': default_status,
#             'data_count': len(loan_amounts)
#         })
        
#     except mysql.connector.Error as err:
#         print(f"Database error: {err}")
#         return jsonify({
#             'error_message': "Database connection error",
#             'loan_amounts': [],
#             'default_status': []
#         })
#     except Exception as e:
#         print(f"Unexpected error: {e}")
#         return jsonify({
#             'error_message': f"System error occurred: {str(e)}",
#             'loan_amounts': [],
#             'default_status': []
#         })
#     finally:
#         if 'conn' in locals() and conn.is_connected():
#             cursor.close()
#             conn.close()


@app.route('/loan_analysis')
@Cache.cached(timeout=CACHE_TIMEOUT, key_prefix='loan_analysis')
def loan_analysis():
    try:
        cursor = conn.cursor(dictionary=True)
        
        # Verify table exists
        cursor.execute("SHOW TABLES LIKE 'LoanData'")
        if not cursor.fetchone():
            return jsonify({
                'error_message': "LoanData table not found",
                'loan_amounts': [],
                'default_status': []
            })
        
        # Get approximate count (faster for large tables)
        cursor.execute("SELECT TABLE_ROWS as count FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'LoanData'")
        total_records = cursor.fetchone()['count']
        
        if total_records == 0:
            return jsonify({
                'warning': "No loan records found in database",
                'loan_amounts': [],
                'default_status': []
            })
        
        # Use server-side cursor for large result sets
        cursor = conn.cursor(dictionary=True, buffered=False)
        
        # Fetch data in batches
        batch_size = 50000
        offset = 0
        loan_amounts = []
        default_status = []
        
        while True:
            query = f"""
                SELECT LoanAmount, `Default` as DefaultStatus 
                FROM LoanData
                WHERE LoanAmount IS NOT NULL
                AND `Default` IS NOT NULL
                LIMIT {batch_size} OFFSET {offset}
            """
            cursor.execute(query)
            batch = cursor.fetchall()
            
            if not batch:
                break
                
            for row in batch:
                try:
                    loan_amounts.append(float(row['LoanAmount']))
                    default_status.append(int(row['DefaultStatus']))
                except (ValueError, TypeError, KeyError) as e:
                    continue
            
            offset += batch_size
        
        return jsonify({
            'loan_amounts': loan_amounts,
            'default_status': default_status,
            'data_count': len(loan_amounts),
            'total_records': total_records
        })
        
    except mysql.connector.Error as err:
        return jsonify({
            'error_message': f"Database error: {str(err)}",
            'loan_amounts': [],
            'default_status': []
        })
    except Exception as e:
        return jsonify({
            'error_message': f"System error: {str(e)}",
            'loan_amounts': [],
            'default_status': []
        })
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

if __name__ == '__main__':
    app.run(debug=True)