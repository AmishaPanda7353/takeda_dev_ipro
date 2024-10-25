import os
import time
from functools import wraps
from io import BytesIO, StringIO

import boto3
import pandas as pd

# Global variable to store application start time
app_start_time = None
current_task_label = None


def initialize_app_start_time():
    global app_start_time
    if app_start_time is None:
        app_start_time = time.time()


# Global dictionary to store timing information
timing_data = {}


def set_task_label(label):
    global current_task_label
    current_task_label = label


def timing_decorator(track_app_start=True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            global timing_data, current_task_label
            method_start_time = time.time()
            result = func(*args, **kwargs)
            method_end_time = time.time()
            elapsed_time = method_end_time - method_start_time

            if track_app_start:
                if app_start_time is None:
                    raise RuntimeError(
                        "Application start time not initialized. Call `initialize_app_start_time` before using this decorator."
                    )
                app_to_method_duration = method_start_time - app_start_time
            else:
                app_to_method_duration = None

            # Use the global task label if set
            function_name = (
                f"{func.__name__} {current_task_label}"
                if current_task_label
                else func.__name__
            )

            update_timing_data(function_name, elapsed_time, app_to_method_duration)
            # if function_name not in timing_data:
            #     timing_data[function_name] = pd.DataFrame(
            #         columns=["Function_Name", "time_taken", "app_to_method_duration"]
            #     )

            # # Append the timing information to the DataFrame
            # timing_data[function_name] = timing_data[function_name].append(
            #     {
            #         "Function_Name": function_name,
            #         "time_taken": elapsed_time,
            #         "app_to_method_duration": app_to_method_duration,
            #     },
            #     ignore_index=True,
            # )

            # Reset task label after each call to avoid affecting subsequent calls
            current_task_label = None

            print(
                f"{function_name} took {elapsed_time:.4f} seconds"
                + (
                    f" (App to Method: {app_to_method_duration:.4f} seconds)"
                    if track_app_start
                    else ""
                )
            )
            return result

        return wrapper

    return decorator


def update_timing_data(function_name, elapsed_time, app_to_method_duration):
    """
    Updates the latest timing data for the given function in the global timing_data.
    If the function doesn't exist in timing_data, it initializes it.
    """
    global timing_data

    # Ensure the DataFrame exists in timing_data
    if function_name not in timing_data:
        timing_data[function_name] = pd.DataFrame(
            columns=["Function_Name", "time_taken", "app_to_method_duration"]
        )

    # If the function exists, update the latest row or replace it
    if not timing_data[function_name].empty:
        # Update the latest row with new timing data
        timing_data[function_name].iloc[-1] = {
            "Function_Name": function_name,
            "time_taken": elapsed_time,
            "app_to_method_duration": app_to_method_duration,
        }
    else:
        # If no previous data exists, add a new row
        timing_data[function_name] = timing_data[function_name].append(
            {
                "Function_Name": function_name,
                "time_taken": elapsed_time,
                "app_to_method_duration": app_to_method_duration,
            },
            ignore_index=True,
        )


def save_timing_info_and_merge(s3_bucket, s3_path, fs, path, file_name, cloud_provider):
    global timing_data

    # Combine all timing DataFrames into a single DataFrame
    combined_df = pd.concat(timing_data.values(), ignore_index=True)

    # Create an S3 client
    s3 = boto3.client("s3")

    # Extract file name from s3_path if necessary
    s3_key = s3_path.lstrip("/")  # Adjust this if your path format is different

    # Read the Excel file from S3
    obj = s3.get_object(Bucket=s3_bucket, Key=s3_key)
    excel_data = obj["Body"].read()

    # Load the Excel DataFrame
    excel_df = pd.read_excel(BytesIO(excel_data))

    # Check if the 'function_name' column is in the Excel DataFrame
    if "Function_Name" not in excel_df.columns:
        raise ValueError("Excel file must contain a 'Function_Name' column.")

    # Merge the DataFrames
    merged_df = pd.merge(excel_df, combined_df, on="Function_Name", how="left")

    # Convert the merged DataFrame to CSV format
    csv_data = merged_df.to_csv(index=False)

    # Save the CSV data to the specified file path
    if cloud_provider != "s3":
        with fs.open(os.path.join(path, file_name), "w") as file:
            file.write(csv_data)
    else:
        s3_client = boto3.client("s3")
        file_path = f"{path}/{file_name}"
        s3_client.put_object(
            Bucket="mcd-ipro",
            Key=file_path,
            Body=csv_data,
        )

    # timing_data = {}
    print(f"Timing information saved to {path}")
