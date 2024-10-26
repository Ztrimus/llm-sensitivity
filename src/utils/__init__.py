import os
import time
import pandas as pd


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

def split_string_into_list(text: str) -> list:
    return [i.strip() for i in text.split(',')]

def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        func_run_log = f"Function {func.__name__} took {execution_time:.4f} seconds to execute"
        print(func_run_log)

        return result

    return wrapper