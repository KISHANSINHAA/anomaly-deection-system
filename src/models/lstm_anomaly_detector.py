import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam
import joblib
import os
from .base_anomaly_detector import SequenceModel

class LSTMAutoencoderModel(SequenceModel):
    def __init__(self, sequence_length=7, n_features=10, encoding_dim=5, learning_rate=0.001, epochs=50, batch_size=32):
        """
        LSTM Autoencoder for temporal anomaly detection
        
        Args:
            sequence_length: Length of input sequences
            n_features: Number of features per timestep
            encoding_dim: Size of encoded representation
            learning_rate: Learning rate for optimizer
            epochs: Number of training epochs
            batch_size: Training batch size
        """
        super().__init__(model_name="lstm_autoencoder", sequence_length=sequence_length, n_features=n_features)
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.encoder = None
        
    def build_model(self):
        """Build LSTM Autoencoder architecture"""
        # Encoder
        inputs = Input(shape=(self.sequence_length, self.n_features))
        encoded = LSTM(self.encoding_dim, activation='relu')(inputs)
        
        # Decoder
        decoded = RepeatVector(self.sequence_length)(encoded)
        decoded = LSTM(self.n_features, activation='relu', return_sequences=True)(decoded)
        
        # Output
        outputs = TimeDistributed(Dense(self.n_features))(decoded)
        
        # Build model
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        # Build encoder for inference
        self.encoder = Model(inputs=inputs, outputs=encoded)
        
        return self
    
    def fit(self, X: np.ndarray, y: np.ndarray = None, validation_split: float = 0.2, verbose: int = 1) -> 'LSTMAutoencoderModel':
        """
        Train the autoencoder
        
        Args:
            X: 3D array of shape (samples, sequence_length, features)
            y: Not used for autoencoder (unsupervised)
            validation_split: Fraction of data for validation
            verbose: Verbosity level
            
        Returns:
            self: Trained model instance
        """
        # Ensure X is 3D
        if X.ndim != 3:
            raise ValueError(f"Expected 3D input, got {X.ndim}D")
        
        if self.model is None:
            self.build_model()
            
        self.history = self.model.fit(
            X, X,  # Autoencoder: input = output
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            verbose=verbose
        )
        
        self.is_trained = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Reconstruct input sequences
        
        Args:
            X: Input sequences (3D array)
            
        Returns:
            Reconstructed sequences
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Ensure X is 3D
        if X.ndim != 3:
            raise ValueError(f"Expected 3D input, got {X.ndim}D")
            
        return self.model.predict(X, verbose=0)
    
    def reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate reconstruction errors for anomaly scoring
        Higher error = more anomalous
        
        Args:
            X: Input sequences (3D array)
            
        Returns:
            Reconstruction errors per sequence
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        reconstructed = self.predict(X)
        # Mean squared error per sample
        errors = np.mean(np.square(X - reconstructed), axis=(1, 2))
        return errors
    
    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Get normalized anomaly scores [0,1]
        1 = most anomalous
        
        Args:
            X: Input sequences (3D array)
            
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        errors = self.reconstruction_errors(X)
        # Normalize to [0,1]
        scores = (errors - np.min(errors)) / (np.max(errors) - np.min(errors) + 1e-8)
        return scores
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained model
        
        Args:
            filepath: Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> 'LSTMAutoencoderModel':
        """
        Load trained model
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            self: Loaded model instance
        """
        self.model = tf.keras.models.load_model(filepath)
        # Rebuild encoder
        inputs = self.model.input
        encoded_layer = self.model.get_layer(index=1).output  # Assuming LSTM layer is at index 1
        self.encoder = Model(inputs=inputs, outputs=encoded_layer)
        self.is_trained = True
        return self

def evaluate_lstm_ae(lstm_ae, X_test, y_true_binary=None):
    """
    Evaluate LSTM Autoencoder performance
    
    Args:
        lstm_ae: Trained LSTMAutoencoder
        X_test: Test sequences
        y_true_binary: True binary labels (optional)
    """
    # Get reconstruction errors and scores
    errors = lstm_ae.reconstruction_errors(X_test)
    scores = lstm_ae.anomaly_scores(X_test)
    
    results = {
        'reconstruction_errors': errors,
        'anomaly_scores': scores,
        'mean_error': np.mean(errors),
        'std_error': np.std(errors)
    }
    
    return results

# Example usage
if __name__ == "__main__":
    pass
