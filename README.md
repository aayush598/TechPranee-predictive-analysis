# Predictive Analysis API

This project provides an API to upload a CSV file containing data, train a machine learning model, and perform predictive analysis using the data.

## Table of Contents

- [Installation](#installation)
- [Setup](#setup)
- [API Endpoints](#api-endpoints)
  - [Upload File](#upload-file)
  - [Train Model](#train-model)
- [Example Requests](#example-requests)
  - [Upload File](#upload-file)
  - [Train Model](#train-model)
- [Error Handling](#error-handling)

## Installation

To set up and run the API, follow these steps:

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Flask
- pandas
- scikit-learn (or any required libraries)

### Step 1: Clone the repository

```bash
git clone https://github.com/aayush598/TechPranee-predictive-analysis.git
cd TechPranee-predictive-analysis
```

### Step 2: Create a virtual environment

```bash
python -m venv venv
```

### Step 3: Activate the virtual environment

- On Windows:

```bash
venv\Scripts\activate
```

- On macOS and Linux:

```bash
source venv/bin/activate
```

### Step 4: Install the required packages

```bash
pip install -r requirements.txt
```

### Step 5: Run the Flask application

```bash
flask run
```

The API will be available at http://localhost:5000.

## Setup

- Make sure the UPLOAD_FOLDER and MODEL_FOLDER directories are created in the project root to store uploaded files and the trained model, respectively.

- Create a .env file to configure the necessary environment variables (if required).

```bash
UPLOAD_FOLDER=./uploads
MODEL_FOLDER=./models
```

- Run the Flask application.

This will start the Flask server, and you can access the API endpoints at `http://localhost:5000`.

## API Endpoints

### Upload File

URL: /upload
Method: POST

Description
Uploads a CSV file containing data for the model training. The file must have a header row with column names.

Request

```
POST /train
```

Response

- 200 OK: If the model is trained successfully.

```
{
  "message": "Model trained successfully.",
  "metrics": {
    "accuracy": 0.95,
    "precision": 0.92,
    "recall": 0.93
  }
}
```

- 400 BAD REQUEST: If no file is uploaded or available for training.

```
{
  "error": "No data file found in the upload folder."
}
```

- 500 INTERNAL SERVER ERROR: If an error occurs during training.

```
{
  "error": "Error message describing the issue"
}
```

## Example Requests

### Example 1: Upload File

Using curl:

```
curl -X POST -F "file=@path/to/your/data.csv" http://localhost:5000/upload
```

Response:

```
{
  "message": "File uploaded and validated successfully.",
  "columns": ["Machine_ID", "Temperature", "Run_Time", "Downtime_Flag"]
}
```

### Example 2: Train Model

Using curl:

```
curl -X POST http://localhost:5000/train
```

Response:

```
{
  "message": "Model trained successfully.",
  "metrics": {
    "accuracy": 0.95,
    "precision": 0.92,
    "recall": 0.93
  }
}
```

## Error Handling

- 400 Bad Request: This error occurs when the request is missing data or contains invalid information. For example, a missing file or invalid CSV structure.
- 500 Internal Server Error: This error indicates that something went wrong on the server side, such as an issue with the model training or file processing.

## Conclusion

This API allows for uploading data, training machine learning models, and retrieving predictions. Make sure to provide the necessary files and data in the correct format to use the API successfully.

For further questions or issues, feel free to reach out!

### Key Sections in the README:

1. **Installation**: Explains how to clone the repository, set up the virtual environment, and install dependencies.
2. **Setup**: Details about configuring the environment and running the Flask app.
3. **API Endpoints**: Lists the available API routes, methods, and expected request/response formats.
4. **Example Requests**: Shows how to use the API with `curl`.
5. **Error Handling**: Describes possible errors and how they are handled by the API.

This README should provide a comprehensive guide to setting up and using the API. You can adjust the paths and other details based on your specific project structure.
