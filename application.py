from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from pathlib import Path

app = Flask(__name__)
CORS(app)

# Conversion rate (update this with current exchange rate)
INR_TO_USD = 0.012

# Load trained model and data
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "LinearRegressionModel.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

try:
    # 2. Load the model using context manager
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully")
    
except FileNotFoundError:
    print(f"❌ Model file not found at {MODEL_PATH}")
    print("Please verify:")
    print(f"- The 'models' directory exists at {MODELS_DIR.absolute()}")
    print(f"- The file 'LinearRegressionModel.pkl' exists in that directory")
    
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")

car = pd.read_csv('Cleaned_Car_data.csv')

# Helper functions for validation
def validate_company(company):
    valid_companies = car['company'].unique()
    return company in valid_companies

def validate_model(company, model_name):
    if not validate_company(company):
        return False
    valid_models = car[car['company'] == company]['name'].unique()
    return model_name in valid_models

def validate_fuel_type(fuel_type):
    valid_fuels = car['fuel_type'].unique()
    return fuel_type in valid_fuels

def validate_year(year):
    current_year = datetime.now().year
    return 1900 <= year <= current_year + 1

def validate_kilometers(km):
    return km >= 0

@app.route('/', methods=['GET'])
def index():
    companies = sorted(car['company'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = sorted(car['fuel_type'].unique())

    car_models_by_company = {
        company: sorted(car[car['company'] == company]['name'].unique())
        for company in companies
    }

    # Load model metrics and convert to USD
    try:
        with open(METADATA_PATH, 'r') as f:
            metrics = json.load(f)['metrics']
            # Convert metrics to USD
            metrics['mae'] = round(metrics['mae'] * INR_TO_USD, 2)
            metrics['rmse'] = round(metrics['rmse'] * INR_TO_USD, 2)
    except Exception:
        metrics = {'r2_score': 'N/A', 'mae': 'N/A', 'rmse': 'N/A'}

    return render_template(
        'index.html',
        companies=['Select Company'] + companies,
        years=['Select Year'] + list(map(str, years)),
        fuel_types=['Select Fuel Type'] + fuel_types,
        car_models=[],
        car_models_by_company=car_models_by_company,
        model_metrics=metrics,
        conversion_rate=INR_TO_USD
    )

@app.route('/get-models/<company>', methods=['GET'])
@cross_origin()
def get_models(company):
    if not validate_company(company):
        return jsonify({'error': 'Invalid company', 'models': []}), 400
    
    models = sorted(car[car['company'] == company]['name'].unique())
    return jsonify({'models': models})

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        # Get and validate input data
        data = request.form
        company = data.get('company')
        car_model = data.get('car_models')
        year = data.get('year')
        fuel_type = data.get('fuel_type')
        driven = data.get('kilo_driven')

        # Basic validation
        if not all([company, car_model, year, fuel_type, driven]):
            return jsonify({
                'error': 'All fields are required',
                'status': 400
            }), 400

        if (company == 'Select Company' or 
            year == 'Select Year' or 
            fuel_type == 'Select Fuel Type'):
            return jsonify({
                'error': 'Please select valid options for all dropdowns',
                'status': 400
            }), 400

        # Convert and validate numeric fields
        try:
            year = int(year)
            driven = int(driven)
            
            if not validate_year(year):
                raise ValueError("Year must be between 1900 and current year")
                
            if not validate_kilometers(driven):
                raise ValueError("Kilometers must be positive")
                
        except ValueError as e:
            return jsonify({
                'error': f'Invalid input: {str(e)}',
                'status': 400
            }), 400

        # Validate categorical fields
        if not validate_company(company):
            return jsonify({
                'error': 'Invalid company selected',
                'status': 400
            }), 400

        if not validate_model(company, car_model):
            return jsonify({
                'error': 'Invalid model for selected company',
                'status': 400
            }), 400

        if not validate_fuel_type(fuel_type):
            return jsonify({
                'error': 'Invalid fuel type selected',
                'status': 400
            }), 400

        # Create input DataFrame
        input_df = pd.DataFrame([{
            'name': car_model,
            'company': company,
            'year': year,
            'kms_driven': driven,
            'fuel_type': fuel_type
        }])

        # Make prediction and convert to USD
        prediction = model.predict(input_df)[0]
        usd_price = np.round(prediction * INR_TO_USD, 2)

        return jsonify({
            'price': usd_price,
            'formatted_price': f"${usd_price:,.2f}",
            'status': 200
        })

    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': 'An unexpected error occurred',
            'status': 500
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)