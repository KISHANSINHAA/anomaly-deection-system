import numpy as np

def inject_spike(data, spike_value=0.5, count=5):
    """
    Inject sudden spikes at random positions
    """
    data = data.copy()
    indices = np.random.choice(len(data), count, replace=False)
    for i in indices:
        data[i] = data[i] + spike_value
    return data, indices


def inject_freeze(data, length=10):
    """
    Simulate sensor freeze (constant value)
    """
    data = data.copy()
    start = np.random.randint(0, len(data) - length)
    data[start:start+length] = data[start]
    return data, list(range(start, start+length))
