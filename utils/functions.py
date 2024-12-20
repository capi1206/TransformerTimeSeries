import numpy as np

def norm_mts_with_window(data, window_size=30):
    """
    Normalize a multivariate time series so that each component of the vector has
    a standard deviation of 1 within a sliding window lokking back in time.
    If window size is 1 does nothing.

    Returns np.array normalized time series data with the same shape as input.
    """
    t_steps, _ = data.shape
    normalized_data = np.zeros_like(data)

    for t in range(t_steps):
        start_i = max(0, t - window_size + 1)
        end_i = t + 1
        window = data[start_i:end_i]
        std = np.std(window, axis=0)
        mean = np.mean(window, axis=0)
        std[std == 0] = 1
        normalized_data[t] = (data[t])/ std

    return normalized_data


#function to create tensor with seg_length of backward steps
def create_seq_from_ts(data, seq_length):
    """
    Function that returns the vectors with corresponding seq_length,
    of observations in the past.

    Output should have shape (data.shape[0] - seq_length, seq_length, num_features)

    """
    sequences = []

    for i in range(len(data) - seq_length + 1):
        # Extract the sequence of features
        sequence = data[i:i+seq_length]
        sequences.append(sequence)

    sequences = np.array(sequences)

    return sequences