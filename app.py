from flask import Flask, render_template, request
import pandas as pd
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load cleaned data and model
car = pd.read_csv("cleaned_car.csv")
model = pickle.load(open("LinearRegressionModel.pkl", "rb"))

@app.route('/')
def index():
    companies = sorted(car["company"].unique())
    car_models = sorted(car["name"].unique())
    year = sorted(car["year"].unique(), reverse=True)
    fuel_type = car["fuel_type"].unique()
    return render_template(
        'index.html',
        companies=companies,
        car_models=car_models,
        years=year,
        fuel_types=fuel_type
    )

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kms_driven'))

    # Create DataFrame in same format as training data
    input_df = pd.DataFrame(
        [[car_model, company, year, kms_driven, fuel_type]],
        columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']
    )

    # Predict
    prediction = model.predict(input_df)[0]
    prediction = round(prediction, 2)

    return render_template(
        'index.html',
        companies=sorted(car["company"].unique()),
        car_models=sorted(car["name"].unique()),
        years=sorted(car["year"].unique(), reverse=True),
        fuel_types=car["fuel_type"].unique(),
        prediction_text=f"Estimated Price: ₹{prediction:,.2f}"
    )

if __name__ == '__main__':
    app.run(debug=True)
