import os
import pytest
import json
from flask import Flask
from app.api import app

# Configure test client
@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["UPLOAD_FOLDER"] = "test_uploads"
    app.config["MODEL_FOLDER"] = "test_models"
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["MODEL_FOLDER"], exist_ok=True)
    with app.test_client() as client:
        yield client
    # Cleanup test directories
    for folder in [app.config["UPLOAD_FOLDER"], app.config["MODEL_FOLDER"]]:
        for file in os.listdir(folder):
            os.remove(os.path.join(folder, file))
        os.rmdir(folder)


# Helper function to create a test CSV file
def create_test_csv(file_path):
    data = """Machine_ID,Temperature,Run_Time,Downtime_Flag
               1,75,100,0
               2,80,120,1
               3,85,150,0"""
    with open(file_path, "w") as file:
        file.write(data)


# Test: Home endpoint
def test_home(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json == {"message": "Welcome to the Manufacturing Predictive Analysis API!"}


# Test: Upload endpoint
def test_upload_file(client):
    test_file_path = os.path.join(app.config["UPLOAD_FOLDER"], "test_data.csv")
    create_test_csv(test_file_path)

    with open(test_file_path, "rb") as file:
        response = client.post("/upload", data={"file": file})

    assert response.status_code == 200
    assert "File uploaded and validated successfully." in response.json["message"]
    assert "columns" in response.json


# Test: Train endpoint
def test_train_model(client):
    test_file_path = os.path.join(app.config["UPLOAD_FOLDER"], "test_data.csv")
    create_test_csv(test_file_path)

    # Upload file first
    with open(test_file_path, "rb") as file:
        client.post("/upload", data={"file": file})

    # Train the model
    response = client.post("/train")
    print(response)
    assert response.status_code == 200
    assert "Model trained successfully." in response.json["message"]
    assert "metrics" in response.json


# Test: Predict endpoint
def test_predict(client):
    test_file_path = os.path.join(app.config["UPLOAD_FOLDER"], "test_data.csv")
    create_test_csv(test_file_path)

    # Upload file and train model first
    with open(test_file_path, "rb") as file:
        client.post("/upload", data={"file": file})
    client.post("/train")

    # Test prediction
    response = client.post(
        "/predict",
        data=json.dumps({"Temperature": 80, "Run_Time": 120}),
        content_type="application/json",
    )

    assert response.status_code == 200
    assert "prediction" in response.json
    assert "confidence" in response.json


# Test: Predict without training
def test_predict_without_training(client):
    response = client.post(
        "/predict",
        data=json.dumps({"Temperature": 80, "Run_Time": 120}),
        content_type="application/json",
    )

    assert response.status_code == 400
    assert "No trained model available" in response.json["error"]


# Test: Train without uploading data
def test_train_without_upload(client):
    response = client.post("/train")
    assert response.status_code == 400
    assert "No valid data uploaded" in response.json["error"]
