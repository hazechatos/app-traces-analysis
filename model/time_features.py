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