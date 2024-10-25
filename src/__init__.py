import sys
import pandas as pd
import os
sys.path.append('../src')

def get_dataframe(dataset_path: str) -> pd.DataFrame:
    """Read the dataset from the given path."""
    try:
        file_extension = os.path.splitext(dataset_path)[1].lower()
        if file_extension == '.csv':
            return pd.read_csv(dataset_path)
        elif file_extension == '.json':
            return pd.read_json(dataset_path)
        else:
            try:
                return pd.read_csv(dataset_path)
            except Exception as exc:
                raise ValueError("Unsupported file format. Only .csv and .json are supported.") from exc
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error