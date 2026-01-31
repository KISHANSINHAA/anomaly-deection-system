"""
Test suite for anomaly detection models
"""
import unittest
import numpy as np
import os
import tempfile
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.models.detector_factory import DetectorFactory
from src.models.isolation_forest_detector import IsolationForestModel
from src.models.lstm_anomaly_detector import LSTMAutoencoderModel
from src.models.gru_anomaly_detector import GRUAutoencoderModel


class TestModels(unittest.TestCase):
    """Test cases for anomaly detection models"""
    
    def setUp(self):
        """Set up test data"""
        # Generate sample data
        np.random.seed(42)
        self.n_samples = 100
        self.n_features = 5
        self.sample_data = np.random.randn(self.n_samples, self.n_features)
        self.sample_data = np.abs(self.sample_data)  # Make positive
        
        # Create sequences for sequence models
        self.sequence_length = 5
        self.sequences = []
        for i in range(len(self.sample_data) - self.sequence_length + 1):
            self.sequences.append(self.sample_data[i:(i + self.sequence_length)])
        self.sequence_data = np.array(self.sequences)
    
    def test_detector_factory(self):
        """Test model factory functionality"""
        # Test available models
        available_models = DetectorFactory.get_available_models()
        expected_models = ['isolation_forest', 'lstm_autoencoder', 'gru_autoencoder']
        self.assertEqual(set(available_models), set(expected_models))
        
        # Test model creation
        for model_type in expected_models:
            model = DetectorFactory.create_model(model_type)
            self.assertIsNotNone(model)
            self.assertFalse(model.is_trained)
    
    def test_isolation_forest_model(self):
        """Test Isolation Forest model"""
        # Create model
        model = IsolationForestModel(contamination=0.1, n_estimators=50)
        
        # Test fitting
        model.fit(self.sample_data)
        self.assertTrue(model.is_trained)
        
        # Test predictions
        predictions = model.predict(self.sample_data)
        self.assertEqual(len(predictions), len(self.sample_data))
        self.assertIn(0, predictions)
        self.assertIn(1, predictions)
        
        # Test anomaly scores
        scores = model.anomaly_scores(self.sample_data)
        self.assertEqual(len(scores), len(self.sample_data))
        self.assertTrue(np.all(scores >= 0))
        self.assertTrue(np.all(scores <= 1))
        
        # Test serialization
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, 'iforest_model.joblib')
            model.save_model(model_path)
            self.assertTrue(os.path.exists(model_path))
            
            # Test loading
            loaded_model = IsolationForestModel()
            loaded_model.load_model(model_path)
            self.assertTrue(loaded_model.is_trained)
    
    def test_lstm_autoencoder_model(self):
        """Test LSTM Autoencoder model"""
        # Create model
        model = LSTMAutoencoderModel(
            sequence_length=self.sequence_length,
            n_features=self.n_features,
            encoding_dim=3,
            epochs=5,  # Few epochs for testing
            batch_size=16
        )
        
        # Test fitting
        model.fit(self.sequence_data)
        self.assertTrue(model.is_trained)
        
        # Test predictions
        predictions = model.predict(self.sequence_data[:10])  # Test subset
        self.assertEqual(predictions.shape[0], 10)
        self.assertEqual(predictions.shape[1], self.sequence_length)
        self.assertEqual(predictions.shape[2], self.n_features)
        
        # Test reconstruction errors
        errors = model.reconstruction_errors(self.sequence_data[:10])
        self.assertEqual(len(errors), 10)
        self.assertTrue(np.all(errors >= 0))
        
        # Test anomaly scores
        scores = model.anomaly_scores(self.sequence_data[:10])
        self.assertEqual(len(scores), 10)
        self.assertTrue(np.all(scores >= 0))
        self.assertTrue(np.all(scores <= 1))
        
        # Test serialization
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, 'lstm_model.h5')
            model.save_model(model_path)
            self.assertTrue(os.path.exists(model_path))
            
            # Test loading
            loaded_model = LSTMAutoencoderModel(
                sequence_length=self.sequence_length,
                n_features=self.n_features,
                encoding_dim=3
            )
            loaded_model.load_model(model_path)
            self.assertTrue(loaded_model.is_trained)
    
    def test_gru_autoencoder_model(self):
        """Test GRU Autoencoder model"""
        # Create model
        model = GRUAutoencoderModel(
            sequence_length=self.sequence_length,
            n_features=self.n_features,
            encoding_dim=3,
            epochs=5,  # Few epochs for testing
            batch_size=16
        )
        
        # Test fitting
        model.fit(self.sequence_data)
        self.assertTrue(model.is_trained)
        
        # Test predictions
        predictions = model.predict(self.sequence_data[:10])  # Test subset
        self.assertEqual(predictions.shape[0], 10)
        self.assertEqual(predictions.shape[1], self.sequence_length)
        self.assertEqual(predictions.shape[2], self.n_features)
        
        # Test reconstruction errors
        errors = model.reconstruction_errors(self.sequence_data[:10])
        self.assertEqual(len(errors), 10)
        self.assertTrue(np.all(errors >= 0))
        
        # Test anomaly scores
        scores = model.anomaly_scores(self.sequence_data[:10])
        self.assertEqual(len(scores), 10)
        self.assertTrue(np.all(scores >= 0))
        self.assertTrue(np.all(scores <= 1))
        
        # Test serialization
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, 'gru_model.h5')
            model.save_model(model_path)
            self.assertTrue(os.path.exists(model_path))
            
            # Test loading
            loaded_model = GRUAutoencoderModel(
                sequence_length=self.sequence_length,
                n_features=self.n_features,
                encoding_dim=3
            )
            loaded_model.load_model(model_path)
            self.assertTrue(loaded_model.is_trained)
    
    def test_model_interface_compliance(self):
        """Test that all models follow the base interface"""
        models = [
            IsolationForestModel(),
            LSTMAutoencoderModel(sequence_length=5, n_features=5),
            GRUAutoencoderModel(sequence_length=5, n_features=5)
        ]
        
        for model in models:
            # Check required methods exist
            self.assertTrue(hasattr(model, 'fit'))
            self.assertTrue(hasattr(model, 'predict'))
            self.assertTrue(hasattr(model, 'anomaly_scores'))
            self.assertTrue(hasattr(model, 'save_model'))
            self.assertTrue(hasattr(model, 'load_model'))
            self.assertTrue(hasattr(model, 'is_trained'))
            
            # Check attributes exist
            self.assertTrue(hasattr(model, 'model_name'))
            self.assertTrue(hasattr(model, 'model'))


if __name__ == '__main__':
    unittest.main()