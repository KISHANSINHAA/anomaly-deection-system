"""
Anomaly Injection Module
Inject various types of anomalies into time series data for testing and validation
"""
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class AnomalyInjector:
    """Inject various types of anomalies into time series data"""
    
    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize anomaly injector
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
    
    def inject_point_anomalies(self, 
                             data: np.ndarray,
                             contamination_rate: float = 0.05,
                             anomaly_multiplier: float = 3.0,
                             feature_indices: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject point anomalies (individual anomalous points)
        
        Args:
            data: Input time series data (2D array)
            contamination_rate: Proportion of points to make anomalous
            anomaly_multiplier: How much to amplify/diminish anomalous points
            feature_indices: Which features to inject anomalies into (None = all)
            
        Returns:
            Tuple of (anomalous_data, anomaly_labels)
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        n_samples, n_features = data.shape
        anomalous_data = data.copy()
        anomaly_labels = np.zeros(n_samples)
        
        # Select features to inject anomalies into
        if feature_indices is None:
            feature_indices = list(range(n_features))
        
        # Calculate number of anomalies per feature
        n_anomalies_per_feature = max(1, int(contamination_rate * n_samples))
        
        for feature_idx in feature_indices:
            feature_data = data[:, feature_idx]
            
            # Select random indices for anomalies
            anomaly_indices = np.random.choice(
                n_samples, size=n_anomalies_per_feature, replace=False
            )
            
            # Inject anomalies (amplify or diminish values)
            for idx in anomaly_indices:
                # Determine if we amplify or diminish (random)
                if np.random.random() > 0.5:
                    # Amplify the value
                    anomalous_data[idx, feature_idx] = (
                        feature_data[idx] * anomaly_multiplier
                    )
                else:
                    # Diminish the value (but keep it positive if original was positive)
                    if feature_data[idx] > 0:
                        anomalous_data[idx, feature_idx] = max(
                            feature_data[idx] / anomaly_multiplier, 
                            feature_data[idx] * 0.1
                        )
                    else:
                        anomalous_data[idx, feature_idx] = (
                            feature_data[idx] * anomaly_multiplier
                        )
                
                anomaly_labels[idx] = 1
        
        logger.info(f"Injected {np.sum(anomaly_labels)} point anomalies "
                   f"({contamination_rate * 100:.1f}% contamination)")
        
        return anomalous_data, anomaly_labels.astype(int)
    
    def inject_collective_anomalies(self,
                                  data: np.ndarray,
                                  n_events: int = 3,
                                  event_duration: int = 10,
                                  amplitude_change: float = 2.0,
                                  feature_indices: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject collective anomalies (sustained anomalous behavior)
        
        Args:
            data: Input time series data
            n_events: Number of collective anomaly events
            event_duration: Duration of each anomalous event
            amplitude_change: Factor by which to change amplitude
            feature_indices: Which features to affect
            
        Returns:
            Tuple of (anomalous_data, anomaly_labels)
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        n_samples, n_features = data.shape
        anomalous_data = data.copy()
        anomaly_labels = np.zeros(n_samples)
        
        # Select features to inject anomalies into
        if feature_indices is None:
            feature_indices = list(range(n_features))
        
        # Generate collective anomalies
        for _ in range(n_events):
            # Random start position (ensuring enough data for the event)
            max_start = n_samples - event_duration
            if max_start <= 0:
                continue
                
            start_idx = np.random.randint(0, max_start)
            end_idx = start_idx + event_duration
            
            # Apply anomaly to selected features
            for feature_idx in feature_indices:
                # Choose anomaly type (increase or decrease)
                if np.random.random() > 0.5:
                    # Sudden increase
                    anomalous_data[start_idx:end_idx, feature_idx] *= amplitude_change
                else:
                    # Sudden decrease
                    anomalous_data[start_idx:end_idx, feature_idx] /= amplitude_change
                
                # Mark all points in event as anomalous
                anomaly_labels[start_idx:end_idx] = 1
        
        logger.info(f"Injected {n_events} collective anomalies of duration {event_duration}")
        
        return anomalous_data, anomaly_labels.astype(int)
    
    def inject_trend_anomalies(self,
                             data: np.ndarray,
                             n_events: int = 2,
                             trend_duration: int = 20,
                             trend_slope: float = 0.1,
                             feature_indices: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject trend anomalies (gradual drift in data pattern)
        
        Args:
            data: Input time series data
            n_events: Number of trend anomalies
            trend_duration: Duration of each trend
            trend_slope: Slope of the trend (positive or negative)
            feature_indices: Which features to affect
            
        Returns:
            Tuple of (anomalous_data, anomaly_labels)
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        n_samples, n_features = data.shape
        anomalous_data = data.copy()
        anomaly_labels = np.zeros(n_samples)
        
        # Select features to inject anomalies into
        if feature_indices is None:
            feature_indices = list(range(n_features))
        
        # Generate trend anomalies
        for _ in range(n_events):
            # Random start position
            max_start = n_samples - trend_duration
            if max_start <= 0:
                continue
                
            start_idx = np.random.randint(0, max_start)
            end_idx = start_idx + trend_duration
            
            # Apply trend to selected features
            for feature_idx in feature_indices:
                # Create linear trend
                time_points = np.arange(trend_duration)
                trend_values = data[start_idx, feature_idx] + trend_slope * time_points
                
                # Apply trend
                anomalous_data[start_idx:end_idx, feature_idx] = trend_values
                
                # Mark all points in trend as anomalous
                anomaly_labels[start_idx:end_idx] = 1
        
        logger.info(f"Injected {n_events} trend anomalies of duration {trend_duration}")
        
        return anomalous_data, anomaly_labels.astype(int)
    
    def inject_seasonal_anomalies(self,
                                data: np.ndarray,
                                period: int = 7,
                                amplitude_change: float = 2.0,
                                n_cycles: int = 3,
                                feature_indices: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject seasonal pattern anomalies
        
        Args:
            data: Input time series data
            period: Seasonal period
            amplitude_change: Factor to change seasonal amplitude
            n_cycles: Number of seasonal cycles to affect
            feature_indices: Which features to affect
            
        Returns:
            Tuple of (anomalous_data, anomaly_labels)
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        n_samples, n_features = data.shape
        anomalous_data = data.copy()
        anomaly_labels = np.zeros(n_samples)
        
        # Select features to inject anomalies into
        if feature_indices is None:
            feature_indices = list(range(n_features))
        
        # Generate seasonal anomalies
        for feature_idx in feature_indices:
            feature_data = data[:, feature_idx]
            
            # Find seasonal components
            for cycle in range(n_cycles):
                # Random start position for seasonal anomaly
                start_idx = np.random.randint(0, n_samples - period)
                end_idx = min(start_idx + period, n_samples)
                
                # Amplify or diminish the seasonal pattern
                if np.random.random() > 0.5:
                    anomalous_data[start_idx:end_idx, feature_idx] *= amplitude_change
                else:
                    anomalous_data[start_idx:end_idx, feature_idx] /= amplitude_change
                
                # Mark affected points as anomalous
                anomaly_labels[start_idx:end_idx] = 1
        
        logger.info(f"Injected seasonal anomalies with period {period}")
        
        return anomalous_data, anomaly_labels.astype(int)
    
    def inject_mixed_anomalies(self,
                             data: np.ndarray,
                             contamination_config: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject multiple types of anomalies with configurable parameters
        
        Args:
            data: Input time series data
            contamination_config: Configuration for different anomaly types
            
        Returns:
            Tuple of (anomalous_data, anomaly_labels)
        """
        if contamination_config is None:
            contamination_config = {
                'point': {'rate': 0.03, 'multiplier': 3.0},
                'collective': {'n_events': 2, 'duration': 15, 'amplitude': 2.5},
                'trend': {'n_events': 1, 'duration': 25, 'slope': 0.15},
                'seasonal': {'period': 7, 'amplitude': 2.0, 'cycles': 2}
            }
        
        anomalous_data = data.copy()
        all_anomaly_labels = np.zeros(len(data))
        
        # Inject point anomalies
        if 'point' in contamination_config:
            point_config = contamination_config['point']
            anomalous_data, point_labels = self.inject_point_anomalies(
                anomalous_data,
                contamination_rate=point_config.get('rate', 0.05),
                anomaly_multiplier=point_config.get('multiplier', 3.0)
            )
            all_anomaly_labels = np.maximum(all_anomaly_labels, point_labels)
        
        # Inject collective anomalies
        if 'collective' in contamination_config:
            collective_config = contamination_config['collective']
            anomalous_data, collective_labels = self.inject_collective_anomalies(
                anomalous_data,
                n_events=collective_config.get('n_events', 3),
                event_duration=collective_config.get('duration', 10),
                amplitude_change=collective_config.get('amplitude', 2.0)
            )
            all_anomaly_labels = np.maximum(all_anomaly_labels, collective_labels)
        
        # Inject trend anomalies
        if 'trend' in contamination_config:
            trend_config = contamination_config['trend']
            anomalous_data, trend_labels = self.inject_trend_anomalies(
                anomalous_data,
                n_events=trend_config.get('n_events', 2),
                trend_duration=trend_config.get('duration', 20),
                trend_slope=trend_config.get('slope', 0.1)
            )
            all_anomaly_labels = np.maximum(all_anomaly_labels, trend_labels)
        
        # Inject seasonal anomalies
        if 'seasonal' in contamination_config:
            seasonal_config = contamination_config['seasonal']
            anomalous_data, seasonal_labels = self.inject_seasonal_anomalies(
                anomalous_data,
                period=seasonal_config.get('period', 7),
                amplitude_change=seasonal_config.get('amplitude', 2.0),
                n_cycles=seasonal_config.get('cycles', 3)
            )
            all_anomaly_labels = np.maximum(all_anomaly_labels, seasonal_labels)
        
        total_anomalies = np.sum(all_anomaly_labels)
        contamination_rate = total_anomalies / len(data)
        
        logger.info(f"Injected mixed anomalies. Total: {total_anomalies} "
                   f"({contamination_rate * 100:.1f}% contamination)")
        
        return anomalous_data, all_anomaly_labels.astype(int)
    
    def get_anomaly_summary(self, anomaly_labels: np.ndarray) -> Dict[str, Any]:
        """
        Get summary statistics of injected anomalies
        
        Args:
            anomaly_labels: Binary anomaly labels
            
        Returns:
            Dictionary with anomaly statistics
        """
        total_points = len(anomaly_labels)
        n_anomalies = np.sum(anomaly_labels)
        contamination_rate = n_anomalies / total_points
        
        # Find consecutive anomaly segments
        anomaly_segments = []
        in_anomaly = False
        segment_start = 0
        
        for i, is_anomaly in enumerate(anomaly_labels):
            if is_anomaly and not in_anomaly:
                # Start of anomaly segment
                segment_start = i
                in_anomaly = True
            elif not is_anomaly and in_anomaly:
                # End of anomaly segment
                anomaly_segments.append(i - segment_start)
                in_anomaly = False
        
        # Handle case where last point is anomalous
        if in_anomaly:
            anomaly_segments.append(len(anomaly_labels) - segment_start)
        
        return {
            'total_points': total_points,
            'n_anomalies': n_anomalies,
            'contamination_rate': contamination_rate,
            'n_anomaly_segments': len(anomaly_segments),
            'avg_segment_length': np.mean(anomaly_segments) if anomaly_segments else 0,
            'max_segment_length': np.max(anomaly_segments) if anomaly_segments else 0
        }


def create_contaminated_dataset(data: np.ndarray,
                              contamination_type: str = 'mixed',
                              **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to create contaminated dataset
    
    Args:
        data: Original clean data
        contamination_type: Type of contamination ('point', 'collective', 'trend', 'seasonal', 'mixed')
        **kwargs: Parameters for specific contamination type
        
    Returns:
        Tuple of (contaminated_data, anomaly_labels)
    """
    injector = AnomalyInjector()
    
    if contamination_type == 'point':
        return injector.inject_point_anomalies(data, **kwargs)
    elif contamination_type == 'collective':
        return injector.inject_collective_anomalies(data, **kwargs)
    elif contamination_type == 'trend':
        return injector.inject_trend_anomalies(data, **kwargs)
    elif contamination_type == 'seasonal':
        return injector.inject_seasonal_anomalies(data, **kwargs)
    elif contamination_type == 'mixed':
        return injector.inject_mixed_anomalies(data, **kwargs)
    else:
        raise ValueError(f"Unknown contamination type: {contamination_type}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 200
    sample_data = np.random.randn(n_samples, 2) * 100 + 500  # Fare-like data
    
    try:
        injector = AnomalyInjector(random_state=42)
        
        # Test different anomaly types
        anomaly_types = ['point', 'collective', 'trend', 'mixed']
        
        for anomaly_type in anomaly_types:
            print(f"\nTesting {anomaly_type} anomalies:")
            
            if anomaly_type == 'point':
                contaminated_data, labels = injector.inject_point_anomalies(
                    sample_data, contamination_rate=0.05
                )
            elif anomaly_type == 'collective':
                contaminated_data, labels = injector.inject_collective_anomalies(
                    sample_data, n_events=3, event_duration=10
                )
            elif anomaly_type == 'trend':
                contaminated_data, labels = injector.inject_trend_anomalies(
                    sample_data, n_events=2, trend_duration=15
                )
            else:  # mixed
                contaminated_data, labels = injector.inject_mixed_anomalies(sample_data)
            
            # Print summary
            summary = injector.get_anomaly_summary(labels)
            print(f"  Contamination rate: {summary['contamination_rate']:.1%}")
            print(f"  Anomaly segments: {summary['n_anomaly_segments']}")
            print(f"  Avg segment length: {summary['avg_segment_length']:.1f}")
    
    except Exception as e:
        print(f"Error: {e}")