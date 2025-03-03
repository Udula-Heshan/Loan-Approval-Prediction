Sfrom flask import Flask, request, render_template
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model
model = joblib.load('loan_approval_model.pkl')

# Define a route to show the form
@app.route('/')
def index():
    return render_template('index.html')

# Define a route to handle the form submission
@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    person_age = int(request.form['person_age'])
    person_income = float(request.form['person_income'])
    person_emp_length = float(request.form['person_emp_length'])
    loan_amnt = float(request.form['loan_amnt'])
    loan_int_rate = float(request.form['loan_int_rate'])
    loan_percent_income = float(request.form['loan_percent_income'])
    cb_person_cred_hist_length = int(request.form['cb_person_cred_hist_length'])
    cb_person_default_on_file = request.form['cb_person_default_on_file']

    # Prepare the input data as a DataFrame
    input_data = {
        "person_age": [person_age],
        "person_income": [person_income],
        "person_emp_length": [person_emp_length],
        "loan_amnt": [loan_amnt],
        "loan_int_rate": [loan_int_rate],
        "loan_percent_income": [loan_percent_income],
        "cb_person_cred_hist_length": [cb_person_cred_hist_length],
        "cb_person_default_on_file": [cb_person_default_on_file]
    }

    # Convert to DataFrame
    input_df = pd.DataFrame(input_data)

    # Handle categorical encoding (LabelEncoder for "cb_person_default_on_file")
    le = LabelEncoder()
    input_df["cb_person_default_on_file"] = le.fit_transform(input_df["cb_person_default_on_file"])

    # Ensure the input has the same number of features as the trained model
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    # Make prediction
    prediction = model.predict(input_df)

    # Return result as a string
    return f"Loan Approval Prediction: {'Loan Approved' if prediction[0] == 1 else 'Loan Not Approved'}"

if __name__ == '__main__':
    app.run(debug=True)
