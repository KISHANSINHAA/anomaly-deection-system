from sklearn.preprocessing import MinMaxScaler

def scale_series(values):
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(values.reshape(-1, 1))
    return scaled_values, scaler
