"""
Base Model Interface for Anomaly Detection
Defines the common interface for all anomaly detection models
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import joblib
import os


class BaseModel(ABC):
    """Abstract base class for all anomaly detection models"""
    
    def __init__(self, model_name: str):
        """
        Initialize base model
        
        Args:
            model_name: Name identifier for the model
        """
        self.model_name = model_name
        self.is_trained = False
        self.model = None
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'BaseModel':
        """
        Train the model on the given data
        
        Args:
            X: Training data
            y: Optional labels (for supervised methods)
            
        Returns:
            self: Trained model instance
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies on the given data
        
        Args:
            X: Input data
            
        Returns:
            Binary predictions (0=normal, 1=anomaly)
        """
        pass
    
    @abstractmethod
    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores for the given data
        
        Args:
            X: Input data
            
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        pass
    
    @abstractmethod
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk
        
        Args:
            filepath: Path to save the model
        """
        pass
    
    @abstractmethod
    def load_model(self, filepath: str) -> 'BaseModel':
        """
        Load a trained model from disk
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            self: Loaded model instance
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and metadata
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'model_type': self.__class__.__name__
        }


class SequenceModel(BaseModel):
    """Base class for sequence-based models (LSTM, GRU)"""
    
    def __init__(self, model_name: str, sequence_length: int = 7, n_features: int = 10):
        """
        Initialize sequence model
        
        Args:
            model_name: Name identifier for the model
            sequence_length: Length of input sequences
            n_features: Number of features per timestep
        """
        super().__init__(model_name)
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.history = None
    
    def create_sequences(self, data: np.ndarray) -> np.ndarray:
        """
        Create sequences from time series data
        
        Args:
            data: 2D array of shape (n_samples, n_features)
            
        Returns:
            3D array of shape (n_sequences, sequence_length, n_features)
        """
        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            sequences.append(data[i:(i + self.sequence_length)])
        return np.array(sequences)
    
    def sequences_to_points(self, sequence_predictions: np.ndarray) -> np.ndarray:
        """
        Convert sequence-level predictions to point-level predictions
        
        Args:
            sequence_predictions: Binary predictions for sequences
            
        Returns:
            Point-level binary predictions
        """
        n_points = len(sequence_predictions) + self.sequence_length - 1
        point_predictions = np.zeros(n_points)
        
        # For each sequence, mark all points in that sequence
        for i, is_anomaly in enumerate(sequence_predictions):
            if is_anomaly:
                start_idx = i
                end_idx = min(i + self.sequence_length, n_points)
                point_predictions[start_idx:end_idx] = 1
                
        return point_predictions


class PointModel(BaseModel):
    """Base class for point-based models"""
    
    def __init__(self, model_name: str):
        """
        Initialize point model
        
        Args:
            model_name: Name identifier for the model
        """
        super().__init__(model_name)
    
    def sequences_to_points(self, sequence_predictions: np.ndarray, 
                          sequence_length: int) -> np.ndarray:
        """
        Convert sequence-level predictions to point-level predictions
        
        Args:
            sequence_predictions: Binary predictions for sequences
            sequence_length: Length of sequences used
            
        Returns:
            Point-level binary predictions
        """
        # For point models, predictions are already at point level
        return sequence_predictions