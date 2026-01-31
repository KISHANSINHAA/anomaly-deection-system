import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support
import joblib
import os
from .base_anomaly_detector import PointModel

class IsolationForestModel(PointModel):
    def __init__(self, contamination=0.1, random_state=42, n_estimators=100):
        """
        Isolation Forest for point anomaly detection
        
        Args:
            contamination: Expected proportion of anomalies (0.0 to 0.5)
            random_state: Random seed for reproducibility
            n_estimators: Number of base estimators in the ensemble
        """
        super().__init__(model_name="isolation_forest")
        self.contamination = contamination
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.feature_columns = None
        
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'IsolationForestModel':
        """
        Train the Isolation Forest model
        
        Args:
            X: Training data (2D array)
            y: Not used for unsupervised learning
            
        Returns:
            self: Trained model instance
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        self.feature_columns = [f'feature_{i}' for i in range(X.shape[1])]
        
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=self.n_estimators,
            max_samples='auto'
        )
        
        self.model.fit(X)
        self.is_trained = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies on the given data
        
        Args:
            X: Input data
            
        Returns:
            Binary predictions (0=normal, 1=anomaly)
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        predictions = self.model.predict(X)
        # Convert from sklearn format (-1=anomaly, 1=normal) to our format (0=normal, 1=anomaly)
        binary_predictions = (predictions == -1).astype(int)
        return binary_predictions
    
    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores for the given data
        
        Args:
            X: Input data
            
        Returns:
            Anomaly scores (higher = more anomalous, range [0,1])
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        # Get decision function scores (lower = more anomalous)
        scores = self.model.decision_function(X)
        # Convert to [0,1] range where 1 = most anomalous
        normalized_scores = (scores.max() - scores) / (scores.max() - scores.min() + 1e-8)
        return normalized_scores
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk
        
        Args:
            filepath: Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_columns': self.feature_columns,
            'contamination': self.contamination,
            'n_estimators': self.n_estimators,
            'random_state': self.random_state,
            'is_trained': self.is_trained
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> 'IsolationForestModel':
        """
        Load a trained model from disk
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            self: Loaded model instance
        """
        data = joblib.load(filepath)
        self.model = data['model']
        self.feature_columns = data['feature_columns']
        self.contamination = data['contamination']
        self.n_estimators = data.get('n_estimators', 100)
        self.random_state = data.get('random_state', 42)
        self.is_trained = data.get('is_trained', True)
        return self

def evaluate_iforest(iforest_detector, X_test, y_true_binary=None):
    """
    Evaluate Isolation Forest performance
    
    Args:
        iforest_detector: Trained IsolationForestDetector
        X_test: Test features
        y_true_binary: True binary labels (optional, for benchmarking)
    """
    # Get predictions and scores
    predictions = iforest_detector.predict(X_test)
    scores = iforest_detector.anomaly_scores(X_test)
    
    # Convert predictions to binary (1=anomaly, 0=normal)
    y_pred_binary = (predictions == -1).astype(int)
    
    results = {
        'predictions': predictions,
        'anomaly_scores': scores,
        'binary_predictions': y_pred_binary,
        'anomaly_rate': np.mean(y_pred_binary)
    }
    
    # If true labels provided, calculate metrics
    if y_true_binary is not None:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_binary, y_pred_binary, average='binary'
        )
        results.update({
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })
    
    return results

# Example usage
if __name__ == "__main__":
    # Sample usage (would be called from main pipeline)
    pass
