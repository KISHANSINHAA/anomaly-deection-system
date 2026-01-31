"""
Time Series Windowing Module
Create and manage time series windows for sequence-based models
"""
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TimeSeriesWindower:
    """Create time series windows for sequence models"""
    
    def __init__(self, sequence_length: int = 7, stride: int = 1):
        """
        Initialize windower
        
        Args:
            sequence_length: Length of sequences to create
            stride: Step size between consecutive windows
        """
        self.sequence_length = sequence_length
        self.stride = stride
        self.n_features = None
    
    def create_windows(self, 
                      data: np.ndarray, 
                      targets: Optional[np.ndarray] = None,
                      include_targets: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sliding windows from time series data
        
        Args:
            data: Input time series data (2D array: samples × features)
            targets: Optional target values
            include_targets: Whether to create target windows
            
        Returns:
            Tuple of (windowed_data, windowed_targets)
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        n_samples, self.n_features = data.shape
        
        if n_samples < self.sequence_length:
            raise ValueError(f"Insufficient data. Need at least {self.sequence_length} samples, "
                           f"got {n_samples}")
        
        # Create sequences
        sequences = []
        target_sequences = [] if targets is not None and include_targets else None
        
        # Generate window indices
        window_starts = range(0, n_samples - self.sequence_length + 1, self.stride)
        
        for start_idx in window_starts:
            # Extract sequence
            sequence = data[start_idx:(start_idx + self.sequence_length)]
            sequences.append(sequence)
            
            # Extract corresponding targets if provided
            if targets is not None and include_targets:
                if targets.ndim == 1:
                    target_seq = targets[start_idx:(start_idx + self.sequence_length)]
                else:
                    target_seq = targets[start_idx:(start_idx + self.sequence_length)]
                target_sequences.append(target_seq)
        
        windowed_data = np.array(sequences)
        
        if target_sequences is not None:
            windowed_targets = np.array(target_sequences)
        else:
            windowed_targets = None
        
        logger.info(f"Created {len(sequences)} sequences of length {self.sequence_length} "
                   f"with stride {self.stride}")
        
        return windowed_data, windowed_targets
    
    def windows_to_points(self, 
                         window_predictions: np.ndarray,
                         aggregation_method: str = 'any') -> np.ndarray:
        """
        Convert window-level predictions to point-level predictions
        
        Args:
            window_predictions: Predictions for each window (1D array)
            aggregation_method: How to aggregate ('any', 'all', 'majority', 'last')
            
        Returns:
            Point-level predictions (1D array)
        """
        if window_predictions.ndim > 1:
            # Handle multi-dimensional predictions (e.g., one-hot encoded)
            window_predictions = np.argmax(window_predictions, axis=1)
        
        n_windows = len(window_predictions)
        n_points = (n_windows - 1) * self.stride + self.sequence_length
        point_scores = np.zeros(n_points)
        point_counts = np.zeros(n_points)
        
        # Aggregate predictions from all overlapping windows
        for i, pred in enumerate(window_predictions):
            start_idx = i * self.stride
            end_idx = start_idx + self.sequence_length
            
            if end_idx <= n_points:
                if aggregation_method == 'any':
                    # If any window predicts anomaly, mark point
                    point_scores[start_idx:end_idx] = np.maximum(
                        point_scores[start_idx:end_idx], pred
                    )
                elif aggregation_method == 'all':
                    # Only mark if ALL overlapping windows predict anomaly
                    point_scores[start_idx:end_idx] += pred
                    point_counts[start_idx:end_idx] += 1
                elif aggregation_method == 'majority':
                    # Majority vote among overlapping windows
                    point_scores[start_idx:end_idx] += pred
                    point_counts[start_idx:end_idx] += 1
                elif aggregation_method == 'last':
                    # Only the last point in each window gets the prediction
                    point_scores[end_idx - 1] = pred
                elif aggregation_method == 'first':
                    # Only the first point in each window gets the prediction
                    point_scores[start_idx] = pred
                elif aggregation_method == 'mean':
                    # Average of all overlapping window predictions
                    point_scores[start_idx:end_idx] += pred
                    point_counts[start_idx:end_idx] += 1
        
        # Final aggregation for methods that require it
        if aggregation_method == 'all':
            # All overlapping windows must predict anomaly
            point_predictions = (point_scores == point_counts).astype(int)
        elif aggregation_method == 'majority':
            # Majority vote (more than 50% of windows)
            point_predictions = (point_scores > point_counts / 2).astype(int)
        elif aggregation_method == 'mean':
            # Average prediction
            with np.errstate(divide='ignore', invalid='ignore'):
                avg_scores = np.divide(point_scores, point_counts, 
                                     out=np.zeros_like(point_scores), where=point_counts!=0)
            point_predictions = avg_scores
        else:
            # For 'any', 'last', 'first' - scores are already binary predictions
            point_predictions = point_scores.astype(int)
        
        return point_predictions
    
    def get_window_info(self) -> Dict[str, Any]:
        """
        Get information about the windowing configuration
        
        Returns:
            Dictionary with windowing information
        """
        return {
            'sequence_length': self.sequence_length,
            'stride': self.stride,
            'n_features': self.n_features,
            'overlap_ratio': (self.sequence_length - self.stride) / self.sequence_length
        }
    
    def create_overlapping_windows(self, data: np.ndarray) -> np.ndarray:
        """
        Create maximally overlapping windows (stride=1)
        
        Args:
            data: Input time series data
            
        Returns:
            Windowed data with maximum overlap
        """
        original_stride = self.stride
        self.stride = 1
        windowed_data, _ = self.create_windows(data)
        self.stride = original_stride
        return windowed_data


class MultiScaleWindower:
    """Create windows at multiple scales for multi-resolution analysis"""
    
    def __init__(self, sequence_lengths: List[int], stride_ratio: float = 0.5):
        """
        Initialize multi-scale windower
        
        Args:
            sequence_lengths: List of different sequence lengths
            stride_ratio: Ratio of stride to sequence length
        """
        self.sequence_lengths = sorted(sequence_lengths, reverse=True)
        self.stride_ratio = stride_ratio
        self.windowers = [
            TimeSeriesWindower(seq_len, max(1, int(seq_len * stride_ratio)))
            for seq_len in sequence_lengths
        ]
    
    def create_multi_scale_windows(self, data: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Create windows at multiple scales
        
        Args:
            data: Input time series data
            
        Returns:
            Dictionary mapping sequence_length -> windowed_data
        """
        multi_scale_windows = {}
        
        for seq_len, windower in zip(self.sequence_lengths, self.windowers):
            windows, _ = windower.create_windows(data)
            multi_scale_windows[seq_len] = windows
            
        return multi_scale_windows
    
    def get_scale_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all scales
        
        Returns:
            List of windowing information for each scale
        """
        return [windower.get_window_info() for windower in self.windowers]


def create_sequences(data: np.ndarray, 
                    sequence_length: int = 7, 
                    stride: int = 1,
                    targets: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Convenience function to create time series sequences
    
    Args:
        data: Input time series data
        sequence_length: Length of sequences
        stride: Step between sequences
        targets: Optional target values
        
    Returns:
        Tuple of (sequences, target_sequences)
    """
    windower = TimeSeriesWindower(sequence_length, stride)
    return windower.create_windows(data, targets)


def aggregate_predictions(window_predictions: np.ndarray,
                         sequence_length: int,
                         stride: int = 1,
                         method: str = 'any') -> np.ndarray:
    """
    Convenience function to aggregate window predictions to point predictions
    
    Args:
        window_predictions: Window-level predictions
        sequence_length: Length of sequences used
        stride: Stride used for windowing
        method: Aggregation method
        
    Returns:
        Point-level predictions
    """
    windower = TimeSeriesWindower(sequence_length, stride)
    return windower.windows_to_points(window_predictions, method)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 50
    n_features = 3
    sample_data = np.random.randn(n_samples, n_features)
    
    try:
        # Test basic windowing
        windower = TimeSeriesWindower(sequence_length=5, stride=2)
        windows, _ = windower.create_windows(sample_data)
        
        print("Basic windowing test:")
        print(f"  Original data shape: {sample_data.shape}")
        print(f"  Windowed data shape: {windows.shape}")
        print(f"  Number of windows: {len(windows)}")
        
        # Test point aggregation
        window_predictions = np.random.choice([0, 1], size=len(windows))
        point_predictions = windower.windows_to_points(window_predictions, 'any')
        
        print(f"  Window predictions shape: {window_predictions.shape}")
        print(f"  Point predictions shape: {point_predictions.shape}")
        
        # Test multi-scale windowing
        multi_windower = MultiScaleWindower([3, 5, 7])
        multi_windows = multi_windower.create_multi_scale_windows(sample_data)
        
        print("\nMulti-scale windowing test:")
        for seq_len, windows in multi_windows.items():
            print(f"  Sequence length {seq_len}: {windows.shape}")
    
    except Exception as e:
        print(f"Error: {e}")