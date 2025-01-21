from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier
import numpy as np


class ManufacturingModel:
    """
    A class to handle the ML model for manufacturing operations.
    """

    def __init__(self, model=None, random_state=42):
        """
        Initializes the ManufacturingModel instance.

        Args:
            model: A machine learning model instance (default: DecisionTreeClassifier).
            random_state: Random seed for reproducibility.
        """
        self.model = model if model else DecisionTreeClassifier(random_state=random_state)
        self.is_trained = False

    def train(self, X, y, test_size=0.2):
        """
        Trains the model on the provided data.

        Args:
            X: Feature matrix.
            y: Target vector.
            test_size: Proportion of the dataset to include in the test split.

        Returns:
            dict: Training metrics including accuracy and F1-score.
        """
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Predict on the test set
        y_pred = self.model.predict(X_test)

        # Compute metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred, average="weighted"),
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }

        return metrics

    def predict(self, input_data):
        """
        Predicts the target value for the given input data.

        Args:
            input_data: A NumPy array or list of input features.

        Returns:
            tuple: Predicted class and confidence score.

        Raises:
            ValueError: If the model is not trained or input data is invalid.
        """
        if not self.is_trained:
            raise ValueError("The model has not been trained yet.")

        # Ensure input data is in the correct format
        if not isinstance(input_data, (np.ndarray, list)):
            raise ValueError("Input data must be a list or numpy array.")

        if isinstance(input_data, list):
            input_data = np.array(input_data).reshape(1, -1)

        # Make predictions
        prediction = self.model.predict(input_data)[0]
        confidence = max(self.model.predict_proba(input_data)[0])
        return prediction, confidence
