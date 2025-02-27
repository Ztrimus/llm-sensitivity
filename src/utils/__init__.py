import os
import time
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def format_execution_time(total_seconds: float) -> str:
    # Calculate days, hours, minutes, and seconds
    days = int(total_seconds // 86400)
    total_seconds %= 86400
    hours = int(total_seconds // 3600)
    total_seconds %= 3600
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60

    # Build output string based on the largest applicable unit
    time_parts = []
    if days > 0:
        time_parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours > 0 or days > 0:  # Include hours if there are any days or hours
        time_parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if (
        minutes > 0 or hours > 0 or days > 0
    ):  # Include minutes if any higher unit exists
        time_parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    # Always include seconds for precision
    time_parts.append(f"{seconds:.4f} second{'s' if seconds != 1 else ''}")

    return ", ".join(time_parts)


def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        func_run_log = f"Function {func.__name__} took {format_execution_time(execution_time)} to execute"
        print(func_run_log)

        return result

    return wrapper


def print_log(*args, is_print=True, is_log=True, is_error=False):
    message = " ".join(str(arg) for arg in args)
    if is_print:
        print(message)
    if is_log:
        logger.info(message)
    if is_error:
        logger.error(message)


def get_dataframe(dataset_path: str) -> pd.DataFrame:
    """Read the dataset from the given path."""
    try:
        file_extension = os.path.splitext(dataset_path)[1].lower()
        if file_extension == ".csv":
            return pd.read_csv(dataset_path)
        elif file_extension == ".json":
            return pd.read_json(dataset_path)
        else:
            try:
                return pd.read_csv(dataset_path)
            except Exception as exc:
                raise ValueError(
                    "Unsupported file format. Only .csv and .json are supported."
                ) from exc
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error


def split_string_into_list(text: str) -> list:
    return [i.strip() for i in text.split(",")]


def filter_safety_response(label):
    if label.strip():
        return label.strip().split()[0].lower()
    else:
        return label