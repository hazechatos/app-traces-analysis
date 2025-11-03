import pandas as pd
import re

def compute_time_features(actions: pd.DataFrame, sequence_lengths: pd.DataFrame):
    durations = []
    speeds = []

    # Traverse each action sequence to get last time value
    for sequence_idx in range(len(actions)):
        action = actions.iloc[sequence_idx]
        length = sequence_lengths.iloc[sequence_idx]
        last_time = 5

        # Traverse sequence in descending order to get last time value
        for i in range(length-1, 0, -1):
            if re.fullmatch(r'^t\d+$', action.iloc[i]):
                last_time_string = action.iloc[i]
                last_time = int(last_time_string[1:])
                break

        # Compute session speed
        n_time_cells = last_time/5 
        n_actions = length - n_time_cells
        speed = round(n_actions / last_time, 6)

        # Append values
        durations.append(last_time)
        speeds.append(speed)

    return pd.DataFrame({'duration': durations, 'speed': speeds})


def bucketize_time_features(time_features: pd.DataFrame, n_buckets: int = 8):
    """
    Bucketize time features (duration and speed) into discrete bins for tokenization.
    
    Args:
        time_features: DataFrame with 'duration' and 'speed' columns
        n_buckets: Number of buckets to create for each feature (default: 20)
    
    Returns:
        DataFrame with 'duration_bucket' and 'speed_bucket' columns containing bucket indices
    """
    result = pd.DataFrame()
    
    # Bucketize duration using quantile-based binning (handles skewness well)
    result['duration_bucket'] = pd.qcut(
        time_features['duration'], 
        q=n_buckets, 
        labels=False, 
        duplicates='drop'
    )
    
    # Bucketize speed using quantile-based binning
    result['speed_bucket'] = pd.qcut(
        time_features['speed'], 
        q=n_buckets, 
        labels=False, 
        duplicates='drop'
    )
    
    # Fill any NaN values (shouldn't happen with qcut, but just in case)
    result = result.fillna(0).astype(int)
    
    return result