"""
Model Factory for Anomaly Detection
Factory pattern for creating and managing different types of anomaly detection models
"""
from typing import Dict, Any, Optional
from .base_anomaly_detector import BaseModel
from .isolation_forest_detector import IsolationForestModel
from .lstm_anomaly_detector import LSTMAutoencoderModel
from .gru_anomaly_detector import GRUAutoencoderModel


class DetectorFactory:
    """Factory class for creating anomaly detection models"""
    
    # Registry of available models
    _models = {
        'isolation_forest': IsolationForestModel,
        'lstm_autoencoder': LSTMAutoencoderModel,
        'gru_autoencoder': GRUAutoencoderModel
    }
    
    @classmethod
    def create_model(cls, model_type: str, **kwargs) -> BaseModel:
        """
        Create a model instance
        
        Args:
            model_type: Type of model to create
            **kwargs: Model-specific parameters
            
        Returns:
            BaseModel instance
            
        Raises:
            ValueError: If model_type is not supported
        """
        if model_type not in cls._models:
            available_models = list(cls._models.keys())
            raise ValueError(f"Model type '{model_type}' not supported. "
                           f"Available models: {available_models}")
        
        model_class = cls._models[model_type]
        return model_class(**kwargs)
    
    @classmethod
    def get_available_models(cls) -> list:
        """
        Get list of available model types
        
        Returns:
            List of model type strings
        """
        return list(cls._models.keys())
    
    @classmethod
    def register_model(cls, model_type: str, model_class):
        """
        Register a new model type
        
        Args:
            model_type: String identifier for the model
            model_class: Model class that inherits from BaseModel
        """
        if not issubclass(model_class, BaseModel):
            raise ValueError("Model class must inherit from BaseModel")
        cls._models[model_type] = model_class
    
    @classmethod
    def get_model_config(cls, model_type: str) -> Dict[str, Any]:
        """
        Get default configuration for a model type
        
        Args:
            model_type: Type of model
            
        Returns:
            Dictionary with default configuration
        """
        configs = {
            'isolation_forest': {
                'n_estimators': 100,
                'contamination': 0.1,
                'random_state': 42
            },
            'lstm_autoencoder': {
                'sequence_length': 7,
                'n_features': 10,
                'encoding_dim': 5,
                'learning_rate': 0.001,
                'epochs': 50,
                'batch_size': 32
            },
            'gru_autoencoder': {
                'sequence_length': 7,
                'n_features': 10,
                'encoding_dim': 5,
                'learning_rate': 0.001,
                'epochs': 50,
                'batch_size': 32
            }
        }
        
        return configs.get(model_type, {})


def create_detector_from_config(detector_type: str, config: Optional[Dict[str, Any]] = None) -> BaseModel:
    """
    Create model from configuration dictionary
    
    Args:
        model_type: Type of model to create
        config: Configuration dictionary (optional)
        
    Returns:
        BaseModel instance
    """
    factory = DetectorFactory()
    
    if config is None:
        config = factory.get_model_config(detector_type)
    
    return factory.create_model(detector_type, **config)