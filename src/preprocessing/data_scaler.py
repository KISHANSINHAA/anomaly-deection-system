"""
Scaling Module for Time Series Data
Provides various scaling methods specifically designed for time series anomaly detection
"""
import numpy as np
from typing import Optional, Dict, Any
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

logger = logging.getLogger(__name__)


class TimeSeriesScaler:
    """Scaler specifically designed for time series data"""
    
    def __init__(self, method: str = 'minmax'):
        """
        Initialize scaler
        
        Args:
            method: Scaling method ('minmax', 'standard', 'robust')
        """
        self.method = method
        self.scaler = None
        self.is_fitted = False
        self.feature_ranges = {}
        self.feature_names = None
    
    def fit(self, X: np.ndarray, feature_names: Optional[list] = None) -> 'TimeSeriesScaler':
        """
        Fit scaler to training data
        
        Args:
            X: Training data (2D array: samples × features)
            feature_names: Optional list of feature names
            
        Returns:
            Fitted scaler instance
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        
        # Select appropriate scaler
        if self.method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.method == 'standard':
            self.scaler = StandardScaler()
        elif self.method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")
        
        # Fit the scaler
        self.scaler.fit(X)
        self.is_fitted = True
        
        # Store feature ranges and statistics
        if hasattr(self.scaler, 'data_min_'):
            self.feature_ranges = {
                'min': self.scaler.data_min_,
                'max': self.scaler.data_max_,
                'range': self.scaler.data_max_ - self.scaler.data_min_
            }
        elif hasattr(self.scaler, 'center_'):
            self.feature_ranges = {
                'center': self.scaler.center_,
                'scale': self.scaler.scale_
            }
        
        logger.info(f"Scaler fitted using {self.method} method on {X.shape[1]} features")
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted scaler
        
        Args:
            X: Data to transform (2D array)
            
        Returns:
            Scaled data
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        return self.scaler.transform(X)
    
    def fit_transform(self, X: np.ndarray, feature_names: Optional[list] = None) -> np.ndarray:
        """
        Fit scaler and transform data in one step
        
        Args:
            X: Data to fit and transform
            feature_names: Optional feature names
            
        Returns:
            Scaled data
        """
        return self.fit(X, feature_names).transform(X)
    
    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled data back to original scale
        
        Args:
            X_scaled: Scaled data
            
        Returns:
            Original scale data
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        return self.scaler.inverse_transform(X_scaled)
    
    def partial_fit(self, X: np.ndarray) -> 'TimeSeriesScaler':
        """
        Partially fit scaler (for online learning scenarios)
        
        Args:
            X: Batch of data to fit
            
        Returns:
            Updated scaler instance
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if self.scaler is None:
            self.fit(X)
        elif hasattr(self.scaler, 'partial_fit'):
            self.scaler.partial_fit(X)
        else:
            # For scalers without partial_fit, we need to refit
            # This is a limitation - would need to store all data for complete refitting
            logger.warning(f"Scaler {self.method} doesn't support partial fitting")
        
        return self
    
    def get_scaling_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the scaling parameters
        
        Returns:
            Dictionary with scaling information
        """
        if not self.is_fitted:
            return {}
        
        info = {
            'method': self.method,
            'n_features': self.scaler.n_features_in_,
            'feature_names': self.feature_names
        }
        
        # Add method-specific parameters
        if hasattr(self.scaler, 'scale_') and self.scaler.scale_ is not None:
            info['scale_factors'] = self.scaler.scale_.tolist()
        if hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None:
            info['means'] = self.scaler.mean_.tolist()
        if hasattr(self.scaler, 'var_') and self.scaler.var_ is not None:
            info['variances'] = self.scaler.var_.tolist()
            
        info.update(self.feature_ranges)
        return info
    
    def save_scaler(self, filepath: str) -> None:
        """
        Save fitted scaler to disk
        
        Args:
            filepath: Path to save scaler
        """
        import joblib
        import os
        
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted scaler")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Scaler saved to {filepath}")
    
    @classmethod
    def load_scaler(cls, filepath: str) -> 'TimeSeriesScaler':
        """
        Load scaler from disk
        
        Args:
            filepath: Path to load scaler from
            
        Returns:
            Loaded scaler instance
        """
        import joblib
        scaler = joblib.load(filepath)
        logger.info(f"Scaler loaded from {filepath}")
        return scaler


class AdaptiveScaler:
    """Adaptive scaler that can handle concept drift in time series"""
    
    def __init__(self, 
                 method: str = 'minmax',
                 window_size: int = 100,
                 update_frequency: int = 50):
        """
        Initialize adaptive scaler
        
        Args:
            method: Base scaling method
            window_size: Size of sliding window for statistics
            update_frequency: How often to update scaling parameters
        """
        self.base_scaler = TimeSeriesScaler(method)
        self.window_size = window_size
        self.update_frequency = update_frequency
        self.data_buffer = []
        self.update_counter = 0
        self.is_adapted = False
    
    def fit(self, X: np.ndarray) -> 'AdaptiveScaler':
        """
        Initial fit on historical data
        
        Args:
            X: Historical training data
            
        Returns:
            Fitted adaptive scaler
        """
        self.base_scaler.fit(X)
        self.data_buffer = X.tolist() if X.ndim > 1 else X.reshape(-1, 1).tolist()
        self.is_adapted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data with potential adaptation
        
        Args:
            X: New data to transform
            
        Returns:
            Scaled data
        """
        if not self.is_adapted:
            raise ValueError("Adaptive scaler not initialized. Call fit() first.")
        
        # Add new data to buffer
        if X.ndim == 1:
            new_data = X.reshape(-1, 1)
        else:
            new_data = X
            
        self.data_buffer.extend(new_data.tolist())
        
        # Keep only recent data in buffer
        if len(self.data_buffer) > self.window_size:
            self.data_buffer = self.data_buffer[-self.window_size:]
        
        # Update scaler periodically
        self.update_counter += 1
        if self.update_counter >= self.update_frequency:
            self._update_scaler()
            self.update_counter = 0
        
        return self.base_scaler.transform(X)
    
    def _update_scaler(self):
        """Update the base scaler with recent data"""
        if len(self.data_buffer) >= 10:  # Minimum data for reliable statistics
            recent_data = np.array(self.data_buffer)
            try:
                self.base_scaler.fit(recent_data)
                logger.debug("Adaptive scaler updated with recent data statistics")
            except Exception as e:
                logger.warning(f"Failed to update adaptive scaler: {e}")


def create_scaler(method: str = 'minmax', adaptive: bool = False, **kwargs) -> TimeSeriesScaler:
    """
    Factory function to create appropriate scaler
    
    Args:
        method: Scaling method
        adaptive: Whether to use adaptive scaling
        **kwargs: Additional parameters for scaler
        
    Returns:
        Scaler instance
    """
    if adaptive:
        return AdaptiveScaler(method=method, **kwargs)
    else:
        return TimeSeriesScaler(method=method)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample data
    np.random.seed(42)
    sample_data = np.random.randn(1000, 5) * 100 + 500  # Fare-like data
    
    try:
        # Test different scalers
        for method in ['minmax', 'standard', 'robust']:
            print(f"\nTesting {method} scaler:")
            
            scaler = TimeSeriesScaler(method)
            scaled_data = scaler.fit_transform(sample_data)
            
            print(f"  Original range: [{sample_data.min():.2f}, {sample_data.max():.2f}]")
            print(f"  Scaled range: [{scaled_data.min():.2f}, {scaled_data.max():.2f}]")
            
            # Test inverse transform
            original_data = scaler.inverse_transform(scaled_data)
            max_diff = np.max(np.abs(sample_data - original_data))
            print(f"  Max reconstruction error: {max_diff:.6f}")
            
            # Print scaling info
            info = scaler.get_scaling_info()
            print(f"  Features: {info['n_features']}")
    
    except Exception as e:
        print(f"Error: {e}")