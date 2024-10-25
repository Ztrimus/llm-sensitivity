import time

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