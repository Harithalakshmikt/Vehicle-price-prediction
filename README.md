# 🚗 Car Price Prediction Web App

A Flask-based web application that predicts the price of used cars based on features like company, model, year, kilometers driven, and fuel type using a trained Linear Regression model.

## 📂 Project Structure
```text
Vehicle-price-prediction/
├── app.py                     # Flask application
├── cleaned_car.csv            # Cleaned dataset
├── LinearRegressionModel.pkl  # Trained ML model
├── templates/
│   └── index.html             # HTML form for user input
└── README.md                  # Project documentation
```
⚙️ Features

Predicts car prices using a trained ML model.

Interactive web interface using Flask and HTML forms.

Handles categorical features (company, model, fuel type) via One-Hot Encoding.

Data cleaning includes:

Removing invalid year and price entries

Converting km driven to numeric

Handling missing fuel types

Removing extreme outliers

🛠 Technologies Used

Python 3

Flask – web framework

Pandas, NumPy – data processing

Scikit-learn – Linear Regression & One-Hot Encoding

Pickle – model serialization

🏃 How to Run Locally

Clone the repository: 
git clone https://github.com/Harithalakshmikt/Vehicle-price-prediction.git
cd Vehicle-price-prediction


Install dependencies:
pip install flask pandas scikit-learn


Run the Flask app:
python app.py


Open the app in your browser:
http://127.0.0.1:5000

If running in GitHub Codespaces, use the Port Preview feature.


🔹 Usage

Select the car company and model.

Enter year, kilometers driven, and fuel type.

Click Predict Price.

The app will display the estimated price of the car.

📈Model Details

Algorithm: Linear Regression

Features: name, company, year, kms_driven, fuel_type

Target: Price

Evaluation: R² score tested on 20% of dataset

⚡ Notes

Currently, the app is tested in GitHub Codespaces.

Public deployment URL can be added later using platforms like Render, AWS EC2, or Streamlit Cloud.

📝 Author

Haritha Lakshmi K.T

LinkedIn: https://www.linkedin.com/in/harithalakshmi-k-t-17459214b/

GitHub: https://github.com/Harithalakshmikt
