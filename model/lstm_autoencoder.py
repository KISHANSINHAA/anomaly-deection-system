from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, RepeatVector, TimeDistributed

def build_lstm_autoencoder(timesteps, features):
    inputs = Input(shape=(timesteps, features))

    # Encoder
    encoded = LSTM(64, activation="relu", return_sequences=False)(inputs)

    # Bottleneck → Repeat
    repeated = RepeatVector(timesteps)(encoded)

    # Decoder
    decoded = LSTM(64, activation="relu", return_sequences=True)(repeated)
    outputs = TimeDistributed(Dense(features))(decoded)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")

    return model
