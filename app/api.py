from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import pandas as pd
from app.utils import (
    validate_csv,
    validate_data_columns,
    preprocess_data,
    save_model,
    load_model,
)
from app.model import ManufacturingModel

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "models"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Initialize global variables
current_model = None
data_columns = None

@app.route("/")
def home():
    return jsonify({"message": "Welcome to the Manufacturing Predictive Analysis API!"})


import pandas as pd

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            print(f"File saved at {file_path}")  # Add this log

            df = pd.read_csv(file_path)
            columns = df.columns.tolist()

            return jsonify({
                "message": "File uploaded and validated successfully.",
                "columns": columns
            }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/train", methods=["POST"])
def train_model():
    """
    Endpoint to train the ML model.
    """
    global current_model, data_columns

    if data_columns is None:
        return jsonify({"error": "No valid data uploaded. Please upload a CSV file first."}), 400

    # Load uploaded CSV
    uploaded_files = os.listdir(app.config["UPLOAD_FOLDER"])
    if not uploaded_files:
        return jsonify({"error": "No data file found in the upload folder."}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_files[-1])
    try:
        data = pd.read_csv(file_path)
        X, y = preprocess_data(data)

        # Train the model
        current_model = ManufacturingModel()
        metrics = current_model.train(X, y)

        # Save the trained model
        model_path = os.path.join(MODEL_FOLDER, "trained_model.pkl")
        save_model(current_model, model_path)

        return jsonify({"message": "Model trained successfully.", "metrics": metrics}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint to make predictions using the trained model.
    """
    global current_model

    if current_model is None or not current_model.is_trained:
        return jsonify({"error": "No trained model available. Please train a model first."}), 400

    # Get JSON input
    input_data = request.get_json()
    if not input_data:
        return jsonify({"error": "No input data provided."}), 400

    try:
        # Extract feature values and predict
        features = list(input_data.values())
        prediction, confidence = current_model.predict(features)

        return jsonify({"prediction": prediction, "confidence": round(confidence, 2)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
