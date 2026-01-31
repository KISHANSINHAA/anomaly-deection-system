"""
Feature Engineering Module
Advanced feature extraction and engineering for time series anomaly detection
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Advanced feature engineering for time series data"""
    
    def __init__(self):
        """Initialize feature engineer"""
        self.scalers = {}
        self.feature_stats = {}
    
    def create_lag_features(self, 
                          data: np.ndarray, 
                          lags: List[int] = [1, 2, 3, 7, 14]) -> np.ndarray:
        """
        Create lag features for time series data
        
        Args:
            data: 1D or 2D array of time series data
            lags: List of lag periods to create
            
        Returns:
            Array with original + lag features
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        n_samples, n_features = data.shape
        lag_features = []
        
        # Add original features
        lag_features.append(data)
        
        # Create lag features for each lag period
        for lag in lags:
            lagged = np.zeros_like(data)
            if lag < n_samples:
                lagged[lag:] = data[:-lag]
                # Fill initial values with the first available value
                lagged[:lag] = data[0]
            lag_features.append(lagged)
        
        return np.concatenate(lag_features, axis=1)
    
    def create_rolling_features(self, 
                              data: np.ndarray, 
                              windows: List[int] = [3, 7, 14, 30]) -> np.ndarray:
        """
        Create rolling window statistics
        
        Args:
            data: 1D or 2D array of time series data
            windows: List of window sizes
            
        Returns:
            Array with rolling statistics features
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        n_samples, n_features = data.shape
        rolling_features = []
        
        # Add original features
        rolling_features.append(data)
        
        # Create rolling features for each window size
        for window in windows:
            for feature_idx in range(n_features):
                feature_data = data[:, feature_idx]
                
                # Rolling mean
                rolling_mean = self._rolling_operation(feature_data, window, np.mean)
                rolling_features.append(rolling_mean.reshape(-1, 1))
                
                # Rolling std
                rolling_std = self._rolling_operation(feature_data, window, np.std)
                rolling_features.append(rolling_std.reshape(-1, 1))
                
                # Rolling min
                rolling_min = self._rolling_operation(feature_data, window, np.min)
                rolling_features.append(rolling_min.reshape(-1, 1))
                
                # Rolling max
                rolling_max = self._rolling_operation(feature_data, window, np.max)
                rolling_features.append(rolling_max.reshape(-1, 1))
        
        return np.concatenate(rolling_features, axis=1)
    
    def _rolling_operation(self, data: np.ndarray, window: int, operation) -> np.ndarray:
        """Apply rolling operation to 1D array"""
        result = np.zeros_like(data)
        for i in range(len(data)):
            start_idx = max(0, i - window + 1)
            window_data = data[start_idx:i+1]
            result[i] = operation(window_data)
        return result
    
    def create_statistical_features(self, data: np.ndarray) -> np.ndarray:
        """
        Create statistical features including trend, seasonality indicators
        
        Args:
            data: 1D or 2D array of time series data
            
        Returns:
            Array with statistical features
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        n_samples, n_features = data.shape
        stat_features = []
        
        # Add original features
        stat_features.append(data)
        
        # Create statistical features for each feature
        for feature_idx in range(n_features):
            feature_data = data[:, feature_idx]
            
            # First and second order differences
            diff1 = np.diff(feature_data, prepend=feature_data[0])
            diff2 = np.diff(diff1, prepend=diff1[0])
            
            stat_features.append(diff1.reshape(-1, 1))
            stat_features.append(diff2.reshape(-1, 1))
            
            # Cumulative statistics
            cumsum = np.cumsum(feature_data)
            cummean = cumsum / np.arange(1, len(feature_data) + 1)
            
            stat_features.append(cumsum.reshape(-1, 1))
            stat_features.append(cummean.reshape(-1, 1))
            
            # Z-scores (standardized values)
            if np.std(feature_data) > 0:
                z_scores = (feature_data - np.mean(feature_data)) / np.std(feature_data)
            else:
                z_scores = np.zeros_like(feature_data)
            stat_features.append(z_scores.reshape(-1, 1))
        
        return np.concatenate(stat_features, axis=1)
    
    def create_cyclical_features(self, 
                               timestamps: np.ndarray,
                               period: str = 'daily') -> np.ndarray:
        """
        Create cyclical features for temporal patterns
        
        Args:
            timestamps: Array of timestamps or date indices
            period: Period type ('daily', 'weekly', 'monthly', 'yearly')
            
        Returns:
            Array with cyclical features (sin, cos components)
        """
        if period == 'daily':
            # Assuming timestamps represent days
            cycle_length = 7  # weekly pattern
        elif period == 'weekly':
            cycle_length = 52  # yearly pattern
        elif period == 'monthly':
            cycle_length = 12  # yearly pattern
        elif period == 'yearly':
            cycle_length = 1  # No cycle for yearly
        else:
            cycle_length = 7
        
        # Normalize to [0, 1] range
        normalized = (timestamps % cycle_length) / cycle_length
        
        # Create sin and cos components
        sin_component = np.sin(2 * np.pi * normalized)
        cos_component = np.cos(2 * np.pi * normalized)
        
        return np.column_stack([sin_component, cos_component])
    
    def engineer_features(self, 
                         data: np.ndarray,
                         feature_types: Optional[List[str]] = None,
                         timestamps: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Comprehensive feature engineering pipeline
        
        Args:
            data: Input time series data (2D array)
            feature_types: List of feature types to create
            timestamps: Optional timestamps for cyclical features
            
        Returns:
            Engineered feature matrix
        """
        if feature_types is None:
            feature_types = ['lag', 'rolling', 'statistical']
        
        engineered_features = [data]  # Start with original features
        
        # Apply different feature engineering techniques
        if 'lag' in feature_types:
            lag_features = self.create_lag_features(data)
            engineered_features.append(lag_features[:, data.shape[1]:])  # Exclude original
            
        if 'rolling' in feature_types:
            rolling_features = self.create_rolling_features(data)
            engineered_features.append(rolling_features[:, data.shape[1]:])  # Exclude original
            
        if 'statistical' in feature_types:
            stat_features = self.create_statistical_features(data)
            engineered_features.append(stat_features[:, data.shape[1]:])  # Exclude original
            
        if 'cyclical' in feature_types and timestamps is not None:
            cyclical_features = self.create_cyclical_features(timestamps)
            engineered_features.append(cyclical_features)
        
        # Combine all features
        final_features = np.concatenate(engineered_features, axis=1)
        
        logger.info(f"Feature engineering completed. Original shape: {data.shape}, "
                   f"Engineered shape: {final_features.shape}")
        
        return final_features


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
    
    def fit(self, X: np.ndarray) -> 'TimeSeriesScaler':
        """
        Fit scaler to training data
        
        Args:
            X: Training data (2D array)
            
        Returns:
            Fitted scaler instance
        """
        if self.method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.method == 'standard':
            self.scaler = StandardScaler()
        elif self.method == 'robust':
            from sklearn.preprocessing import RobustScaler
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")
        
        # Fit the scaler
        self.scaler.fit(X)
        self.is_fitted = True
        
        # Store feature ranges for reference
        if hasattr(self.scaler, 'data_min_'):
            self.feature_ranges = {
                'min': self.scaler.data_min_,
                'max': self.scaler.data_max_,
                'range': self.scaler.data_max_ - self.scaler.data_min_
            }
        
        logger.info(f"Scaler fitted using {self.method} method")
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted scaler
        
        Args:
            X: Data to transform
            
        Returns:
            Scaled data
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        return self.scaler.transform(X)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit scaler and transform data in one step
        
        Args:
            X: Data to fit and transform
            
        Returns:
            Scaled data
        """
        return self.fit(X).transform(X)
    
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
    
    def get_scaling_info(self) -> Dict[str, Any]:
        """
        Get information about the scaling parameters
        
        Returns:
            Dictionary with scaling information
        """
        if not self.is_fitted:
            return {}
        
        info = {
            'method': self.method,
            'n_features': self.scaler.n_features_in_
        }
        
        if hasattr(self.scaler, 'scale_'):
            info['scale_factors'] = self.scaler.scale_.tolist()
        if hasattr(self.scaler, 'mean_'):
            info['means'] = self.scaler.mean_.tolist()
            
        info.update(self.feature_ranges)
        return info


class TimeSeriesWindower:
    """Create time series windows for sequence models"""
    
    def __init__(self, sequence_length: int = 7):
        """
        Initialize windower
        
        Args:
            sequence_length: Length of sequences to create
        """
        self.sequence_length = sequence_length
    
    def create_windows(self, 
                      data: np.ndarray, 
                      target: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sliding windows from time series data
        
        Args:
            data: Input time series data (2D array: samples × features)
            target: Optional target values for supervised learning
            
        Returns:
            Tuple of (windowed_data, windowed_targets)
        """
        n_samples, n_features = data.shape
        
        if n_samples < self.sequence_length:
            raise ValueError(f"Insufficient data. Need at least {self.sequence_length} samples, "
                           f"got {n_samples}")
        
        # Create sequences
        sequences = []
        targets = [] if target is not None else None
        
        for i in range(n_samples - self.sequence_length + 1):
            # Extract sequence
            sequence = data[i:(i + self.sequence_length)]
            sequences.append(sequence)
            
            # Extract corresponding target if provided
            if target is not None:
                target_value = target[i + self.sequence_length - 1]  # Target for last point in sequence
                targets.append(target_value)
        
        windowed_data = np.array(sequences)
        
        if targets is not None:
            windowed_targets = np.array(targets)
        else:
            windowed_targets = None
        
        logger.info(f"Created {len(sequences)} sequences of length {self.sequence_length}")
        
        return windowed_data, windowed_targets
    
    def windows_to_points(self, 
                         window_predictions: np.ndarray,
                         prediction_type: str = 'last') -> np.ndarray:
        """
        Convert window-level predictions to point-level predictions
        
        Args:
            window_predictions: Predictions for each window
            prediction_type: How to aggregate ('last', 'any', 'all', 'majority')
            
        Returns:
            Point-level predictions
        """
        n_windows = len(window_predictions)
        n_points = n_windows + self.sequence_length - 1
        point_predictions = np.zeros(n_points)
        
        for i, window_pred in enumerate(window_predictions):
            if prediction_type == 'last':
                # Only the last point in each window gets the prediction
                point_idx = i + self.sequence_length - 1
                if point_idx < n_points:
                    point_predictions[point_idx] = window_pred
            elif prediction_type == 'any':
                # If any point in window is anomalous, mark all points
                if window_pred == 1:
                    start_idx = i
                    end_idx = min(i + self.sequence_length, n_points)
                    point_predictions[start_idx:end_idx] = 1
            elif prediction_type == 'all':
                # Only mark if all points in window are anomalous
                if window_pred == 1:
                    # This would require more complex logic for multi-point windows
                    point_predictions[i + self.sequence_length - 1] = 1
            elif prediction_type == 'majority':
                # Mark based on majority vote in overlapping windows
                start_idx = i
                end_idx = min(i + self.sequence_length, n_points)
                point_predictions[start_idx:end_idx] += window_pred
        
        # For majority voting, convert to binary (threshold at 0.5)
        if prediction_type == 'majority':
            point_predictions = (point_predictions > 0).astype(int)
        
        return point_predictions


# Convenience functions
def engineer_and_scale_features(data: np.ndarray,
                              feature_types: Optional[List[str]] = None,
                              scaling_method: str = 'minmax',
                              sequence_length: int = 7) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Complete pipeline: feature engineering + scaling + windowing
    
    Args:
        data: Input time series data
        feature_types: Types of features to engineer
        scaling_method: Scaling method to use
        sequence_length: Sequence length for windowing
        
    Returns:
        Tuple of (processed_sequences, pipeline_info)
    """
    # Step 1: Feature Engineering
    engineer = FeatureEngineer()
    engineered_data = engineer.engineer_features(data, feature_types)
    
    # Step 2: Scaling
    scaler = TimeSeriesScaler(scaling_method)
    scaled_data = scaler.fit_transform(engineered_data)
    
    # Step 3: Windowing
    windower = TimeSeriesWindower(sequence_length)
    windowed_data, _ = windower.create_windows(scaled_data)
    
    # Return processed data and pipeline information
    pipeline_info = {
        'original_shape': data.shape,
        'engineered_shape': engineered_data.shape,
        'scaled_shape': scaled_data.shape,
        'windowed_shape': windowed_data.shape,
        'scaler_info': scaler.get_scaling_info(),
        'sequence_length': sequence_length
    }
    
    return windowed_data, pipeline_info


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    n_features = 3
    
    sample_data = np.random.randn(n_samples, n_features)
    sample_data = np.abs(sample_data)  # Make it positive for fare-like data
    
    try:
        # Complete pipeline
        processed_data, info = engineer_and_scale_features(
            sample_data, 
            feature_types=['lag', 'rolling'],
            scaling_method='minmax',
            sequence_length=5
        )
        
        print("Pipeline completed successfully:")
        print(f"  Original shape: {info['original_shape']}")
        print(f"  Final shape: {info['windowed_shape']}")
        print(f"  Sequence length: {info['sequence_length']}")
        
    except Exception as e:
        print(f"Error: {e}")