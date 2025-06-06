from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Load trained model
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

# Load dataset to populate form dropdowns
car = pd.read_csv('Cleaned_Car_data.csv')

@app.route('/', methods=['GET'])
def index():
    companies = sorted(car['company'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = sorted(car['fuel_type'].unique())

    # Create dictionary: company -> list of models
    car_models_by_company = {}
    for company in companies:
        models = sorted(car[car['company'] == company]['name'].unique())
        car_models_by_company[company] = models

    # Pass all to template
    return render_template(
        'index.html',
        companies=['Select Company'] + companies,
        years=['Select Year'] + list(map(str, years)),
        fuel_types=['Select Fuel Type'] + fuel_types,
        # We'll pass empty list for models initially (template needs to handle this)
        car_models=[],
        car_models_by_company=car_models_by_company
    )


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    # Get data from form
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = request.form.get('year')
    fuel_type = request.form.get('fuel_type')
    driven = request.form.get('kilo_driven')

    # Validate input
    if not all([company, car_model, year, fuel_type, driven]) or \
       company == 'Select Company' or \
       car_model == '' or \
       year == 'Select Year' or \
       fuel_type == 'Select Fuel Type':
        return "Invalid input. Please fill in all fields correctly.", 400

    try:
        year = int(year)
        driven = int(driven)
    except ValueError:
        return "Year and Kilometers Driven must be numeric.", 400

    # Create DataFrame for prediction
    input_df = pd.DataFrame(
        data=[[car_model, company, year, driven, fuel_type]],
        columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']
    )

    # Predict
    try:
        prediction = model.predict(input_df)
    except Exception as e:
        return f"Prediction failed: {str(e)}", 500

     # Convert predicted price from INR to USD
    inr_price = np.round(prediction[0], 2)
    conversion_rate = 0.012  # Example conversion rate from ₹ to $
    usd_price = np.round(inr_price * conversion_rate, 2)

    return f"${usd_price}"  # Return price in dollars


if __name__ == '__main__':
    app.run(debug=True)
