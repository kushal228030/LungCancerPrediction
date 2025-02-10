import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import joblib


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
loaded_model = joblib.load("C:/Users/User/Desktop/app/lung_cancer_log_reg.pkl")


# Define class names
CLASS_NAMES = ["NoCancer", "Cancer"]

@app.route('/', methods=['GET'])
def home():
    """Home route to check if API is running"""
    return "Logistic Model API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    """Predicts lung cancer based on input features"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        # Convert input data to NumPy array
        features = np.array(data['features']).reshape(1, -1)
        
        # Make prediction
        prediction = loaded_model.predict(features)
        predicted_class = CLASS_NAMES[int(prediction[0])]
        
        return jsonify({"prediction": predicted_class})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=5000)

    
