"""
Dynamic Rolling Percentile Thresholding
Adaptive thresholding based on local error distributions for anomaly detection
"""
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class DynamicThreshold:
    """Dynamic rolling percentile threshold for anomaly detection"""
    
    def __init__(self, 
                 percentile: float = 95.0,
                 window_size: int = 100,
                 min_window_size: int = 10,
                 adaptive: bool = True):
        """
        Initialize dynamic threshold
        
        Args:
            percentile: Percentile threshold (e.g., 95.0 for 95th percentile)
            window_size: Size of rolling window for threshold calculation
            min_window_size: Minimum window size for reliable statistics
            adaptive: Whether to adapt threshold based on recent data
        """
        self.percentile = percentile
        self.window_size = window_size
        self.min_window_size = min_window_size
        self.adaptive = adaptive
        
        # Store historical errors for threshold calculation
        self.error_history = []
        self.threshold_history = []
        self.is_initialized = False
    
    def calculate_threshold(self, errors: np.ndarray) -> float:
        """
        Calculate dynamic threshold based on error distribution
        
        Args:
            errors: Array of reconstruction errors or anomaly scores
            
        Returns:
            Calculated threshold value
        """
        if len(errors) == 0:
            raise ValueError("Cannot calculate threshold with empty error array")
        
        # Add new errors to history
        self.error_history.extend(errors.tolist())
        
        # Keep only recent errors based on window size
        if len(self.error_history) > self.window_size:
            self.error_history = self.error_history[-self.window_size:]
        
        # Calculate threshold based on current error distribution
        current_errors = np.array(self.error_history)
        
        if len(current_errors) < self.min_window_size:
            # Not enough data, use global statistics
            threshold = np.percentile(errors, self.percentile)
            logger.warning(f"Insufficient data ({len(current_errors)} < {self.min_window_size}), "
                          f"using global percentile: {threshold:.4f}")
        else:
            # Use local statistics from recent window
            threshold = np.percentile(current_errors, self.percentile)
            
            # Apply adaptive adjustment if enabled
            if self.adaptive and len(self.threshold_history) > 0:
                threshold = self._adaptive_adjustment(threshold, current_errors)
        
        # Store threshold history
        self.threshold_history.append(threshold)
        
        if not self.is_initialized and len(self.error_history) >= self.min_window_size:
            self.is_initialized = True
            logger.info(f"Dynamic threshold initialized with {len(self.error_history)} samples")
        
        return threshold
    
    def _adaptive_adjustment(self, threshold: float, current_errors: np.ndarray) -> float:
        """
        Apply adaptive adjustment to threshold based on recent trends
        
        Args:
            threshold: Base threshold calculated from percentiles
            current_errors: Current error window
            
        Returns:
            Adjusted threshold
        """
        if len(self.threshold_history) < 2:
            return threshold
        
        # Calculate recent trend in thresholds
        recent_thresholds = np.array(self.threshold_history[-10:]) if len(self.threshold_history) >= 10 else np.array(self.threshold_history)
        threshold_trend = np.mean(np.diff(recent_thresholds)) if len(recent_thresholds) > 1 else 0
        
        # Calculate error volatility
        error_std = np.std(current_errors)
        error_mean = np.mean(current_errors)
        
        # Adjust threshold based on trend and volatility
        if error_std > 0 and error_mean > 0:
            volatility_factor = min(error_std / error_mean, 1.0)  # Cap at 100%
            trend_factor = 0.1 * volatility_factor  # Limited trend influence
            adjustment = threshold_trend * trend_factor
            adjusted_threshold = threshold + adjustment
        else:
            adjusted_threshold = threshold
        
        return adjusted_threshold
    
    def predict_anomalies(self, errors: np.ndarray) -> np.ndarray:
        """
        Predict anomalies based on dynamic thresholding
        
        Args:
            errors: Array of reconstruction errors or anomaly scores
            
        Returns:
            Binary anomaly predictions (1 = anomaly, 0 = normal)
        """
        threshold = self.calculate_threshold(errors)
        predictions = (errors > threshold).astype(int)
        
        anomaly_rate = np.mean(predictions)
        logger.debug(f"Threshold: {threshold:.4f}, Anomaly rate: {anomaly_rate:.2%}")
        
        return predictions
    
    def get_threshold_info(self) -> Dict[str, Any]:
        """
        Get information about current threshold state
        
        Returns:
            Dictionary with threshold information
        """
        if not self.threshold_history:
            return {}
        
        return {
            'current_threshold': self.threshold_history[-1] if self.threshold_history else None,
            'threshold_history_length': len(self.threshold_history),
            'error_history_length': len(self.error_history),
            'percentile': self.percentile,
            'window_size': self.window_size,
            'is_initialized': self.is_initialized,
            'mean_threshold': np.mean(self.threshold_history) if self.threshold_history else 0,
            'std_threshold': np.std(self.threshold_history) if self.threshold_history else 0
        }
    
    def reset(self):
        """Reset the threshold calculator"""
        self.error_history = []
        self.threshold_history = []
        self.is_initialized = False
        logger.info("Dynamic threshold reset")


class MultiScaleThreshold:
    """Apply thresholding at multiple scales for robust detection"""
    
    def __init__(self, percentiles: List[float] = [90.0, 95.0, 99.0]):
        """
        Initialize multi-scale thresholding
        
        Args:
            percentiles: List of percentiles to use for thresholding
        """
        self.percentiles = percentiles
        self.thresholds = [DynamicThreshold(p) for p in percentiles]
    
    def predict_anomalies(self, errors: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Predict anomalies using multiple thresholds
        
        Args:
            errors: Array of reconstruction errors
            
        Returns:
            Tuple of (consensus_predictions, threshold_details)
        """
        predictions_by_threshold = []
        threshold_details = {}
        
        # Get predictions from each threshold
        for i, (percentile, threshold_calc) in enumerate(zip(self.percentiles, self.thresholds)):
            predictions = threshold_calc.predict_anomalies(errors)
            predictions_by_threshold.append(predictions)
            
            # Store threshold info
            threshold_info = threshold_calc.get_threshold_info()
            threshold_details[f'p{int(percentile)}'] = {
                'threshold': threshold_info.get('current_threshold', 0),
                'anomaly_rate': np.mean(predictions),
                'predictions': predictions.tolist()
            }
        
        # Combine predictions (majority vote)
        predictions_array = np.array(predictions_by_threshold)
        consensus_predictions = (np.mean(predictions_array, axis=0) > 0.5).astype(int)
        
        # Add consensus info
        threshold_details['consensus'] = {
            'anomaly_rate': np.mean(consensus_predictions),
            'predictions': consensus_predictions.tolist()
        }
        
        return consensus_predictions, threshold_details


class AdaptivePercentileThreshold:
    """Adaptive percentile threshold that adjusts based on data characteristics"""
    
    def __init__(self, 
                 initial_percentile: float = 95.0,
                 sensitivity: float = 1.0,
                 max_percentile: float = 99.5,
                 min_percentile: float = 80.0):
        """
        Initialize adaptive percentile threshold
        
        Args:
            initial_percentile: Starting percentile
            sensitivity: Sensitivity to adjust percentile (higher = more responsive)
            max_percentile: Maximum allowed percentile
            min_percentile: Minimum allowed percentile
        """
        self.current_percentile = initial_percentile
        self.sensitivity = sensitivity
        self.max_percentile = max_percentile
        self.min_percentile = min_percentile
        self.error_history = []
        self.anomaly_history = []
    
    def predict_anomalies(self, errors: np.ndarray) -> np.ndarray:
        """
        Predict anomalies with adaptive percentile adjustment
        
        Args:
            errors: Array of reconstruction errors
            
        Returns:
            Binary anomaly predictions
        """
        # Calculate threshold with current percentile
        threshold = np.percentile(errors, self.current_percentile)
        
        # Make predictions
        predictions = (errors > threshold).astype(int)
        
        # Update adaptation based on recent performance
        self._adapt_percentile(predictions, errors)
        
        # Store history
        self.error_history.extend(errors.tolist())
        self.anomaly_history.extend(predictions.tolist())
        
        # Keep limited history
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]
            self.anomaly_history = self.anomaly_history[-1000:]
        
        return predictions
    
    def _adapt_percentile(self, predictions: np.ndarray, errors: np.ndarray):
        """
        Adapt percentile based on recent prediction performance
        
        Args:
            predictions: Recent predictions
            errors: Recent errors
        """
        if len(self.anomaly_history) < 50:  # Need sufficient history
            return
        
        # Calculate recent anomaly rate
        recent_anomalies = np.array(self.anomaly_history[-50:])
        recent_rate = np.mean(recent_anomalies)
        
        # Desired anomaly rate (around 2-5% typically)
        target_rate = 0.03
        
        # Adjust percentile based on deviation from target
        rate_error = recent_rate - target_rate
        
        # Update percentile (with bounds checking)
        adjustment = self.sensitivity * rate_error * 10  # Scale factor
        new_percentile = self.current_percentile - adjustment
        
        # Apply bounds
        new_percentile = max(self.min_percentile, min(self.max_percentile, new_percentile))
        
        # Only update if significant change
        if abs(new_percentile - self.current_percentile) > 0.1:
            self.current_percentile = new_percentile
            logger.debug(f"Adaptive threshold adjusted: {self.current_percentile:.1f}th percentile")


def apply_dynamic_thresholding(errors: np.ndarray,
                             percentile: float = 95.0,
                             window_size: int = 100,
                             method: str = 'simple') -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convenience function to apply dynamic thresholding
    
    Args:
        errors: Reconstruction errors or anomaly scores
        percentile: Percentile threshold
        window_size: Rolling window size
        method: Thresholding method ('simple', 'multi_scale', 'adaptive')
        
    Returns:
        Tuple of (predictions, threshold_info)
    """
    if method == 'simple':
        threshold_calc = DynamicThreshold(percentile, window_size)
        predictions = threshold_calc.predict_anomalies(errors)
        threshold_info = threshold_calc.get_threshold_info()
        
    elif method == 'multi_scale':
        threshold_calc = MultiScaleThreshold([90.0, 95.0, 99.0])
        predictions, threshold_info = threshold_calc.predict_anomalies(errors)
        
    elif method == 'adaptive':
        threshold_calc = AdaptivePercentileThreshold(percentile)
        predictions = threshold_calc.predict_anomalies(errors)
        threshold_info = {
            'current_percentile': threshold_calc.current_percentile,
            'method': 'adaptive'
        }
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return predictions, threshold_info


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample errors (simulating reconstruction errors)
    np.random.seed(42)
    n_samples = 200
    normal_errors = np.random.exponential(1.0, n_samples)
    # Inject some anomalous errors
    anomaly_indices = np.random.choice(n_samples, size=20, replace=False)
    normal_errors[anomaly_indices] = np.random.exponential(5.0, 20)
    
    try:
        # Test different thresholding methods
        methods = ['simple', 'multi_scale', 'adaptive']
        
        for method in methods:
            print(f"\nTesting {method} thresholding:")
            
            predictions, info = apply_dynamic_thresholding(
                normal_errors, 
                percentile=95.0, 
                method=method
            )
            
            anomaly_rate = np.mean(predictions)
            detected_anomalies = np.sum(predictions)
            
            print(f"  Anomaly rate: {anomaly_rate:.2%}")
            print(f"  Detected anomalies: {detected_anomalies}/{len(predictions)}")
            
            if method == 'simple':
                print(f"  Final threshold: {info.get('current_threshold', 0):.4f}")
            elif method == 'adaptive':
                print(f"  Final percentile: {info.get('current_percentile', 0):.1f}")
    
    except Exception as e:
        print(f"Error: {e}")