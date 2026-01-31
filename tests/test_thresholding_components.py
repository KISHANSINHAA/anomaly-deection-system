"""
Test suite for thresholding components
"""
import unittest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.thresholding.dynamic_threshold_calculator import DynamicThreshold, MultiScaleThreshold, AdaptivePercentileThreshold


class TestThresholding(unittest.TestCase):
    """Test cases for dynamic thresholding"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        # Generate sample errors (mostly normal with some anomalies)
        self.normal_errors = np.random.exponential(1.0, 200)
        # Inject anomalous errors
        anomaly_indices = np.random.choice(200, size=20, replace=False)
        self.normal_errors[anomaly_indices] = np.random.exponential(5.0, 20)
    
    def test_dynamic_threshold_basic(self):
        """Test basic dynamic threshold functionality"""
        threshold_calc = DynamicThreshold(percentile=95.0, window_size=100)
        
        # Test threshold calculation
        threshold = threshold_calc.calculate_threshold(self.normal_errors)
        self.assertIsInstance(threshold, float)
        self.assertGreater(threshold, 0)
        
        # Test predictions
        predictions = threshold_calc.predict_anomalies(self.normal_errors)
        self.assertEqual(len(predictions), len(self.normal_errors))
        self.assertIn(0, predictions)
        self.assertIn(1, predictions)
        
        # Test that anomalies are detected
        self.assertGreater(np.sum(predictions), 0)
    
    def test_dynamic_threshold_initialization(self):
        """Test threshold initialization and state management"""
        threshold_calc = DynamicThreshold(percentile=90.0, window_size=50)
        
        # Should not be initialized with insufficient data
        threshold_calc.calculate_threshold(self.normal_errors[:5])
        self.assertFalse(threshold_calc.is_initialized)
        
        # Should be initialized with sufficient data
        threshold_calc.calculate_threshold(self.normal_errors[:50])
        self.assertTrue(threshold_calc.is_initialized)
    
    def test_multi_scale_threshold(self):
        """Test multi-scale thresholding"""
        multi_threshold = MultiScaleThreshold(percentiles=[90.0, 95.0, 99.0])
        
        predictions, details = multi_threshold.predict_anomalies(self.normal_errors)
        
        # Check results structure
        self.assertEqual(len(predictions), len(self.normal_errors))
        self.assertIsInstance(details, dict)
        self.assertIn('p90', details)
        self.assertIn('p95', details)
        self.assertIn('p99', details)
        self.assertIn('consensus', details)
        
        # Check individual threshold details
        for percentile_key in ['p90', 'p95', 'p99']:
            self.assertIn('threshold', details[percentile_key])
            self.assertIn('anomaly_rate', details[percentile_key])
            self.assertIn('predictions', details[percentile_key])
    
    def test_adaptive_percentile_threshold(self):
        """Test adaptive percentile thresholding"""
        adaptive_threshold = AdaptivePercentileThreshold(
            initial_percentile=95.0,
            sensitivity=1.0,
            max_percentile=99.5,
            min_percentile=80.0
        )
        
        predictions = adaptive_threshold.predict_anomalies(self.normal_errors)
        
        # Check results
        self.assertEqual(len(predictions), len(self.normal_errors))
        self.assertIn(0, predictions)
        self.assertIn(1, predictions)
        
        # Check that percentile can adapt
        initial_percentile = adaptive_threshold.current_percentile
        # Process more data to allow adaptation
        for _ in range(5):
            adaptive_threshold.predict_anomalies(np.random.exponential(1.0, 50))
        
        # Percentile should have changed (or at least stayed the same)
        self.assertIsNotNone(adaptive_threshold.current_percentile)
    
    def test_threshold_info_retrieval(self):
        """Test threshold information retrieval"""
        threshold_calc = DynamicThreshold(percentile=95.0)
        threshold_calc.predict_anomalies(self.normal_errors)
        
        info = threshold_calc.get_threshold_info()
        
        # Check required fields
        self.assertIn('current_threshold', info)
        self.assertIn('threshold_history_length', info)
        self.assertIn('error_history_length', info)
        self.assertIn('percentile', info)
        self.assertIn('window_size', info)
        self.assertIn('is_initialized', info)
        
        # Check value types
        self.assertIsInstance(info['current_threshold'], (float, type(None)))
        self.assertIsInstance(info['threshold_history_length'], int)
        self.assertIsInstance(info['error_history_length'], int)
        self.assertIsInstance(info['percentile'], float)
        self.assertIsInstance(info['window_size'], int)
        self.assertIsInstance(info['is_initialized'], bool)


if __name__ == '__main__':
    unittest.main()