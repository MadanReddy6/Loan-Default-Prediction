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
csv_file_path = r'D:\Madan_MITRAz\MIT Project\Loan Default Prediction\data\Loan_default old.csv'
df = pd.read_csv(csv_file_path)
print(df.head())

# Remove all NaN values
df = df.dropna()

# Define the batch size (e.g., 100 rows per insert)
batch_size = 100
print(f"Total records: {len(df)}")

# Insert data in batches
for start in range(0, len(df), batch_size):
    end = start + batch_size
    batch_data = df.iloc[start:end].values.tolist()  # Get the rows for the batch
    
    # Prepare the insert query
    insert_query = """
    INSERT INTO LoanData (Age, Income, LoanAmount, CreditScore, MonthsEmployed, 
                          NumCreditLines, InterestRate, LoanTerm, DTIRatio, Education, 
                          EmploymentType, MaritalStatus, HasMortgage, HasDependents, 
                          LoanPurpose, HasCoSigner, `Default`)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    try:
        # Execute the batch insert
        cursor.executemany(insert_query, batch_data)
        conn.commit()  # Commit after each batch
        print(f"Inserted records {start} to {end}")
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        conn.rollback()  # Rollback in case of error

# Close the cursor and connection
cursor.close()
conn.close()

print("Data inserted successfully.")