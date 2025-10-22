import pandas as pd
import os
import csv

def reader(ds_name: str) -> pd.DataFrame:
    """Robust CSV loader for files where rows have variable field counts.
    Uses Python's csv.reader to parse lines, pads rows to the maximum column count with empty strings,
    and returns a pandas DataFrame with missing values replaced by empty strings.
    Accepts either 'train' or 'train.csv' as input.
    """
    filename = ds_name if ds_name.endswith('.csv') else ds_name + '.csv'
    path = os.path.join('data', filename)
    # Read using csv.reader which avoids pandas C-engine tokenization errors on malformed rows
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
    if not rows:
        print(f"No rows read from {path}")
        return pd.DataFrame()
    max_cols = max(len(r) for r in rows)
    padded = [r + [''] * (max_cols - len(r)) for r in rows]
    df = pd.DataFrame(padded)
    df = df.fillna('')
    rename_map = {col: ('util' if col == 0 else 'browser' if col == 1 else f'action_{col-1}') for col in df.columns}
    df.rename(columns=rename_map, inplace=True)
    return df