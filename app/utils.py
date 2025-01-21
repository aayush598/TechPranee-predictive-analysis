import pandas as pd
import joblib
from sklearn.exceptions import NotFittedError


def validate_csv(file):
    """
    Validates if the uploaded file is a valid CSV and returns a DataFrame.
    
    Args:
        file: File object uploaded via the API.
        
    Returns:
        A pandas DataFrame if the file is valid.
    
    Raises:
        ValueError: If the file is not a valid CSV.
    """
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        raise ValueError(f"Invalid CSV file: {e}")


def validate_data_columns(df, required_columns):
    """
    Validates that the required columns are present in the DataFrame.
    
    Args:
        df: The DataFrame to validate.
        required_columns: List of column names that must be present.
    
    Raises:
        ValueError: If any required column is missing.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")


def preprocess_data(df, feature_columns):
    """
    Extracts feature columns from the DataFrame for model training or prediction.
    
    Args:
        df: The DataFrame containing data.
        feature_columns: List of column names to use as features.
        
    Returns:
        A DataFrame with only the feature columns.
    """
    return df[feature_columns]


def save_model(model, filepath):
    """
    Saves a trained model to a file.
    
    Args:
        model: The trained model to save.
        filepath: The path where the model file should be saved.
    """
    joblib.dump(model, filepath)


def load_model(filepath):
    """
    Loads a trained model from a file.
    
    Args:
        filepath: The path to the model file.
        
    Returns:
        The loaded model.
    
    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    try:
        return joblib.load(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at {filepath}")


def get_prediction(model, input_data):
    """
    Makes a prediction using the trained model.
    
    Args:
        model: The trained model.
        input_data: A list or array of input features.
    
    Returns:
        A tuple containing the prediction and the confidence score.
    
    Raises:
        NotFittedError: If the model is not trained or loaded correctly.
    """
    try:
        prediction = model.predict(input_data)[0]
        confidence = max(model.predict_proba(input_data)[0])
        return prediction, confidence
    except NotFittedError:
        raise NotFittedError("The model is not trained or fitted yet.")
    except Exception as e:
        raise ValueError(f"Error during prediction: {e}")
