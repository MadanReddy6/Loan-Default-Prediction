import mysql.connector
import pandas as pd

# Connect to MySQL database
conn = mysql.connector.connect(
    host='localhost',  # Use 'localhost' or '127.0.0.1' for local MySQL server
    user='root',
    password='Reddy@656',
    database='loan_db'
)

cursor = conn.cursor()

# Load the CSV file into a DataFrame
csv_file_path = r'C:\Users\K Madan Mohan Reddy\Desktop\Loan_default old.csv'
df = pd.read_csv(csv_file_path)

# Define the batch size (e.g., 1000 rows per insert)
batch_size = 1000

# Insert data in batches
for start in range(0, len(df), batch_size):
    end = start + batch_size
    batch_data = df.iloc[start:end].values  # Get the rows for the batch
    
    # Prepare the insert query
    insert_query = """
    INSERT INTO LoanData (Age, Income, LoanAmount, CreditScore, MonthsEmployed, 
                          NumCreditLines, InterestRate, LoanTerm, DTIRatio, Education, 
                          EmploymentType, MaritalStatus, HasMortgage, HasDependents, 
                          LoanPurpose, HasCoSigner, Default)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    # Execute the batch insert
    cursor.executemany(insert_query, batch_data)
    conn.commit()  # Commit after each batch

# Close the cursor and connection
cursor.close()
conn.close()

print("Data inserted successfully.")
