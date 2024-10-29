import datetime
import functools
import io
import json
import logging
import os
import posixpath as pp
import re
import shutil
import signal
import sqlite3
import sys
import tempfile
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout, suppress
from functools import partial
from io import BytesIO, StringIO
from typing import Callable, List, Tuple, Union

import boto3
import dateutil
import fsspec
import nltk
import openai
import pandas as pd
import s3fs
import spacy
import tiktoken
import wrapt
import wrapt_timeout_decorator
import yaml
from core.utils.read_config import cloud_config, process_guidelines
from fs.subfs import SubFS
from fs_s3fs import S3FS

MYLOGGERNAME = "QueryInsights"


class CloudStorageHandler(logging.Handler):
    """
    Logging FileHandler for cloud storage
    """

    def __init__(self, fs, log_file_path: str):
        """
        Parameters
        ----------
        fs : fsspec.filesystem, optional
            Filesystem of the url
        log_file_path : str
            File path to save the logs
        """
        super().__init__()
        self._fs = fs
        self._log_file_path = log_file_path

    def emit(self, record):
        log_message = self.format(record)
        try:
            # Load previous log messages
            if cloud_config.cloud_provider != "s3":
                with self._fs.open(self._log_file_path, "r") as f:
                    _history = f.read()
            else:
                _history = read_data(
                    self._log_file_path, cloud_config.domain_storage.account_name
                )
        except FileNotFoundError:
            # If file doesn't exist yet, history is empty
            _history = ""
        # Write old log messages and new log messages
        if cloud_config.cloud_provider != "s3":
            with self._fs.open(self._log_file_path, "w") as f:
                f.write(_history)
                f.write(log_message + "\n")
        else:
            meassage = _history + "\n" + log_message + "\n"
            write_data(file_path=self._log_file_path, content=meassage)


def create_logger(
    logger_name: str = "QueryInsights",
    level: str = "WARNING",
    log_file_path: str = None,
    verbose: bool = True,
    fs=None,
) -> None:
    """Creates logger object. By default, logger objects have global namespace.

    Parameters
    ----------
    logger_name : str, optional
        Name of the logger. Used by other scripts in this package, by default "QueryInsights"
    level : str, optional
        Level or severity of the events they are used to track. Acceptable values are ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], by default "WARNING", by default "WARNING"
    log_file_path : str, optional
        File path to save the logs, by default None
    verbose : bool, optional
        If `True` logs will be printed to console, by default True
    fs : fsspec.filesystem, optional
        Filesystem of the url, by default ``None``
    """
    # Create logger
    logger = logging.getLogger(logger_name)

    # Set level
    all_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if level in all_levels:
        logger.setLevel(level)
    else:
        print(
            f"""{level} is not part of supported levels i.e.('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
            Setting WARNING as default level"""
        )
        logger.setLevel("WARNING")

    # Set handler
    logger.handlers.clear()
    if log_file_path is not None and cloud_config.cloud_provider != "s3":
        # TODO: fsspec: Check for other cloud storage providers like S3
        fs = fs or fsspec.filesystem("file")
        if fs.protocol == "file":
            if not fs.exists(log_file_path) and not fs.exists(
                os.path.dirname(log_file_path)
            ):
                fs.makedirs(os.path.dirname(log_file_path))
            fh = logging.FileHandler(log_file_path, mode="w")
        else:
            if not fs.exists(log_file_path) and not fs.exists(
                os.path.dirname(log_file_path)
            ):
                # fs.makedirs is not creating a directory in blob storage sometimes
                # Create a dummy file if it doesn't exist
                with fs.open(log_file_path, mode="w") as _:
                    pass
            fh = CloudStorageHandler(fs, log_file_path)
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(module)s - %(funcName)s - %(levelname)s - %(message)s"
            )
        )
        logger.addHandler(fh)
    elif log_file_path is not None:
        # TODO: fsspec: Check for other cloud storage providers like S3
        fs = fs or fsspec.filesystem("file")
        if not fs.exists(log_file_path):
            # fs.makedirs is not creating a directory in blob storage sometimes
            # Create a dummy file if it doesn't exist
            # fs.makedirs(os.path.dirname(log_file_path), recreate=True)
            write_data(file_path=log_file_path, content="")
            # with fs.open(log_file_path, mode="w") as _:
            #     pass
        fh = CloudStorageHandler(fs, log_file_path)
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(module)s - %(funcName)s - %(levelname)s - %(message)s"
            )
        )
        logger.addHandler(fh)

    if verbose:
        sh = logging.StreamHandler()  # (sys.stdout)
        sh.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(module)s - %(funcName)s - %(levelname)s - %(message)s"
            )
        )
        logger.addHandler(sh)
    if log_file_path is None and not verbose:
        logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return


def copy_folders(source_path, source_folders, source_files, destination_path, fs):
    """
    Copy folders and files from source_path to destination_path.

    Parameters
    ----------
    source_path : str
        The path to the source folder.
    source_folders : list of str
        The list of folders to copy.
    source_files : list of str
        The list of files to copy.
    destination_path : str
        The path to the destination folder.
    fs : filesystem object
        The filesystem object to use.

    Returns
    -------
    None

    """

    # Copy each folder
    for folder in source_folders:
        if fs.exists(pp.join(source_path, folder)):
            source_folder_path = pp.join(source_path, folder)
            destination_folder_path = pp.join(destination_path, folder)
            fs.cp(source_folder_path, destination_folder_path, recursive=True)

    # Copy the files
    for file in source_files:
        if fs.exists(pp.join(source_path, file)):
            if "runtime" in file:
                source_file_path = pp.join(source_path, file)
                destination_file_path = pp.join(destination_path, file)
                fs.move(source_file_path, destination_file_path)

            else:
                source_file_path = pp.join(source_path, file)
                destination_file_path = pp.join(destination_path, file)
                fs.cp(source_file_path, destination_file_path)

    return None


def copy_folder_s3(source_path, source_folders, source_files, destination_path, fs):

    # Ensure destination folder exists
    if not fs.exists(destination_path):
        fs.makedirs(destination_path)

    def copy_recursive(src_folder, dst_folder):
        # Ensure destination sub-folder exists
        if not fs.exists(dst_folder):
            fs.makedirs(dst_folder)

        # Copy files and folders in the current directory
        for item in fs.listdir(src_folder):
            src_item_path = os.path.join(src_folder, item)
            dst_item_path = os.path.join(dst_folder, item)

            if fs.isdir(src_item_path):
                # Recursively copy sub-folders
                copy_recursive(src_item_path, dst_item_path)
            elif fs.isfile(src_item_path):
                # Copy files
                with fs.open(src_item_path, "rb") as src_file:
                    with fs.open(dst_item_path, "wb") as dst_file:
                        dst_file.write(src_file.read())
            else:
                print(f"{src_item_path} is not a file or folder, skipping.")

    # Copy only the selected folders
    for folder in source_folders:
        src_folder_path = os.path.join(source_path, folder)
        dst_folder_path = os.path.join(destination_path, folder)

        if fs.exists(src_folder_path) and fs.isdir(src_folder_path):
            # Copy the specified folder and its contents
            copy_recursive(src_folder_path, dst_folder_path)
        else:
            print(f"Source folder {src_folder_path} does not exist or is not a folder")

    # Copy the specified files that are not inside any selected folders
    for file_name in source_files:
        src_file_path = os.path.join(source_path, file_name)
        dst_file_path = os.path.join(destination_path, file_name)

        if fs.exists(src_file_path) and fs.isfile(src_file_path):
            # Copy files manually
            with fs.open(src_file_path, "rb") as src_file:
                with fs.open(dst_file_path, "wb") as dst_file:
                    dst_file.write(src_file.read())
        else:
            print(f"File {src_file_path} does not exist or is not a file")

    return None


class TokenLimitError(Exception):
    """custom exception class to indicate GPT token limit is exceeded."""

    pass


class SensitiveContentError(Exception):
    """custom exception class to indicate openAI model's flag the content as sensitive."""

    pass


class TimeoutError(Exception):
    """Custom exception class to indicate we got timeout error even after retry."""

    pass


class DotifyDict(dict):
    """
    DotifyDict makes a dict accessable by dot(.)
    for a dictionary ex = {'aaa': 1, 'bbb': {'ccc': 2}}, ex['bbb']['ccc'] can be accessed using ex.bbb.ccc

    Returns
    -------
    a custom dict type accessable by dot(.)
    """

    MARKER = object()

    def __init__(self, value=None):
        if value is None:
            pass
        elif isinstance(value, dict):
            for key in value:
                self.__setitem__(key, value[key])
        else:
            raise TypeError("expected dict")

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, DotifyDict):
            value = DotifyDict(value)
        super(DotifyDict, self).__setitem__(key, value)

    def __getitem__(self, key):
        found = self.get(key, DotifyDict.MARKER)
        if found is DotifyDict.MARKER:
            found = DotifyDict()
            super(DotifyDict, self).__setitem__(key, found)
        return found

    __setattr__, __getattr__ = __setitem__, __getitem__


# TODO: this needs to be changed as it signal alarm is for Posix only.
def timeout_depricated(func):
    """
    Decorator to timeout a function call.

    Parameters
    ----------
    func : function
        function to be decorated

    Returns
    -------
    function
        decorated function

    Raises
    ------
    TimeoutError
        if the function call times out.
    """

    def decorator(*args, **kwargs):
        """_decorator_

        Parameters
        ----------
        *args : list
            list of arguments
        **kwargs : dict
            dictionary of keyword arguments

        Returns
        -------
        function
            decorated function

        Raises
        ------
        TimeoutError
            if the function call times out.
        """
        model_param_dict = kwargs.get("model_param_dict", {})
        seconds = model_param_dict.get("timeout")
        max_tries = model_param_dict.get("max_tries", 1)
        error_message = "Function call timed out"
        if os.name == "nt":
            # Its windows, skip this handling
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                raise e

            return result
        else:
            for i in range(max_tries):
                if seconds is None:
                    return func(*args, **kwargs)
                else:

                    def _handle_timeout(signum, frame):
                        raise TimeoutError(error_message)

                    def _handle_exception(exc_type, exc_value, traceback):
                        if i == max_tries - 1:
                            raise exc_type(exc_value).with_traceback(traceback)

                    signal.signal(signal.SIGALRM, _handle_timeout)
                    signal.alarm(seconds)
                    try:
                        result = func(*args, **kwargs)
                    except Exception as e:
                        _handle_exception(type(e), e, e.__traceback__)
                    finally:
                        signal.alarm(0)
            return result
        # raise TimeoutError(error_message)

    return decorator


def timeout(
    timeout_seconds: int = 2,
    max_tries: int = 2,
    use_signals: bool = False,
    logger_name: str = "QueryInsights",
):
    """
    Decorator to timeout a function call.

    Parameters
    ----------
    timeout_seconds : int
        Seconds to timeout in, by default 2
    max_tries : int
        Number retires after timeout, by default 2
    use_signals : bool
        If signals module is to be used,
        will raise error if not run in main thread,
        defaults to multiprocessing if False, by default False
    logger_name : str
        Logger name, by default ``QueryInsights``

    Returns
    -------
    function
        decorated function

    Raises
    ------
    TimeoutError
        if the function call times out.
    """
    logger = logging.getLogger(logger_name)

    @wrapt.decorator
    def decorator(wrapped, instance, args, kwargs):
        """Decorator function

        Parameters
        ----------
        wrapped :
            Function that is being wrapped in the decorator
        instance :
            Instance object
        args : list
            List of arguments
        kwargs : dict
            Dictionary of keyword arguments

        Returns
        -------
        function
            Decorated function

        Raises
        ------
        TimeoutError
            If the function call times out.
        """

        if os.name == "nt":
            # TODO: Needs more debugging to make it work in windows
            # Its windows, skip this handling
            try:
                result = wrapped(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"The following error occured when running function {wrapped.__name__}"
                )
                logger.error(traceback.format_exc())
                raise e

            return result

        model_param_dict = kwargs.get("model_param_dict", {}) or (
            args[0] if args else {}
        )
        _seconds = model_param_dict.get("timeout", None)
        _max_tries = model_param_dict.get("max_tries", None)

        # Track number of retries
        retries = 0
        # Give preference to timeout provided in config file over default timeout
        timeout_seconds_ = _seconds if _seconds is not None else timeout_seconds
        # Give preference to max tries provided in config file over default max tries
        max_tries_ = _max_tries if _max_tries is not None else max_tries
        # While number of retires is lesser than value provided, the function is run max_tries_ times
        while retries < (max_tries_):
            try:
                # Timeout wrapper on the specified function
                result = wrapt_timeout_decorator.timeout(
                    timeout_seconds_,
                    use_signals=use_signals,
                    timeout_exception=TimeoutError,
                )(wrapped)(*args, **kwargs)
                return result
            except TimeoutError:
                # Except TimeoutError add to number of retries
                logger.info(
                    f"Function {wrapped.__name__} timed out. Retries {retries+1}/{max_tries_}"
                )
                # print(traceback.format_exc())
                retries += 1

            except Exception as e:
                # Code errored out due to issues in function
                retries = max_tries_
                logger.error(
                    f"The following error occured when running function {wrapped.__name__}"
                )
                logger.error(traceback.format_exc())
                raise e

        raise TimeoutError(
            f"Function {wrapped.__name__} timed out after {max_tries_} retries."
        )

    return decorator


def fs_connection(fs_connection_dict=None, fs_key=None):
    """Returns the prefix url and storage options by reading the connection dictionary and the account key

    Parameters
    ----------
    fs_connection_dict : dict, optional
        Dictionary containing configuration settings to connect to the cloud, by default None
    fs_key : str, optional
        Account key to make connection to the specified platform (in fs_connection_dict). If platform is not None, it will look for the path in the data_config and read the key from there. Can be left as None for using local File storage (Windows, Linux) (when platform in None), by default None

    Returns
    -------
    str
        Prefix URL for connecting to the file storage. None for normal FS (Linux, Windows etc)
    str
        Storage options for connecting to the file storage. None for normal FS (Linux, Windows etc)

    Raises
    ------
    ValueError
        When platform is not None and appropriate account key is not specified to make the connection
    """
    platform = fs_connection_dict.platform
    if platform is None:
        prefix_url = None
        storage_options = None
    elif platform == "azure":
        # TODO: fsspec: Add validations to make sure that this path exists and add some logger messages?
        # load Azure account key
        if fs_key is not None:
            load_key_to_env(secret_key=fs_key, env_var="BLOB_ACCOUNT_KEY")
        else:
            load_key_to_env(
                secret_key=fs_connection_dict.account_key_path,
                env_var="BLOB_ACCOUNT_KEY",
                fs=None,
            )
        account_key = os.getenv("BLOB_ACCOUNT_KEY")
        if account_key is None:
            raise ValueError(
                "Failed loading the Azure account key into environment variable. Please use `fs_key` parameter pass account key as a string or specify the path to the account key in the data configuration cloud_config.account_key_path)"
            )

        prefix_url = fs_connection_dict.prefix_url
        fs_connection_string = f"DefaultEndpointsProtocol={fs_connection_dict.DefaultEndpointsProtocol};AccountName={fs_connection_dict.AccountName};AccountKey={account_key};EndpointSuffix={fs_connection_dict.EndpointSuffix}"
        storage_options = {
            "connection_string": fs_connection_string,
            "account_key": account_key,
        }
    return prefix_url, storage_options


def get_fs_and_abs_path(path, storage_options):
    """Get the Filesystem and paths from a urlpath and options.

    Parameters
    ----------
    path : string or iterable
        Absolute or relative filepath, URL (may include protocols like
        ``s3://``), or globstring pointing to data. If None is provided, this
        parameter will be defaulted to an empty string to avoid errors.
    storage_options : dict, optional
        Additional keywords to pass to the filesystem class.

    Returns
    -------
    fsspec.FileSystem
       Filesystem Object
    list(str)
        List of paths in the input path.
    """
    path = path if path is not None else ""
    if cloud_config.cloud_provider != "s3":
        fs, _, paths = fsspec.core.get_fs_token_paths(
            path, storage_options=storage_options
        )
        if len(paths) == 1:
            return fs, paths[0]
        else:
            return fs, paths
    else:
        s3_fs = S3FS(cloud_config.domain_storage.account_name, **storage_options)
        sub_fs = SubFS(s3_fs, path)
        return sub_fs, None


def load_yml(path, *, fs=None, **kwargs):
    """Load a yml file from the input `path`.

    Parameters
    ----------
    path : string
        Absolute or relative filepath, URL (may include protocols like
        ``s3://``).
    fs : fsspec.filesystem, optional
        Filesystem of the url, by default ``None``

    Returns
    -------
    dict
        dictionary of the loaded yml file
    """
    fs = fs or fsspec.filesystem("file")
    with fs.open(path, mode="r") as fp:
        return yaml.safe_load(fp, **kwargs)


def load_key_to_env(secret_key, env_var, fs=None):
    """
    Loads the the secret key into specified environment variables

    Parameters
    ----------
    secret_key : str
        This can be a secret key as a string or a path to a file that contains the secret key
    env_var : str
        Environment variable to which the secret key will be loaded
    fs : fsspec.filesystem, optional
        Filesystem of the url, by default ``None``

    Returns
    -------
    None
    """
    fs = fs or fsspec.filesystem("file")
    if fs.exists(secret_key):
        # If we received file path, read the file, else directly pass in the key.
        with fs.open(secret_key, "r") as f:
            key = f.read()
    else:
        key = secret_key

    os.environ[env_var] = key


def read_data(
    path,
    bucket=None,
    logger_name: str = "QueryInsights",
    fs=None,
):
    """
    Reads the excel/csv file and formats the column names.

    Parameters
    ----------
    path : str
        path to the csv or xlsx file
    logger_name : str
        Logger name, by default ``QueryInsights``
    fs : fsspec.filesystem, optional
        Filesystem of the url, by default ``None``

    Returns
    -------
    pd.DataFrame
        formatted data frame that can be loaded to SQL DB

    Raises
    ------
    ValueError
        if the path is not a valid csv or xlsx file.
    """
    # Create logger
    # TODO: All file reads should happen using this function. Please add modifications for other file formats as required.
    logger = logging.getLogger(logger_name)

    try:
        if cloud_config.cloud_provider != "s3":
            fs = fs or fsspec.filesystem("file")
            if path.endswith("csv"):
                with fs.open(path, mode="rb") as fp:
                    data = pd.read_csv(fp)
            if path.endswith("xlsx"):
                with fs.open(path, mode="rb") as fp:
                    data = pd.read_excel(fp, engine="openpyxl")
        else:
            try:
                s3_client = boto3.resource("s3")
                content_object = s3_client.Object(bucket, path)
                if path.endswith("csv"):
                    file_content = content_object.get()["Body"].read().decode("utf-8")
                    data = pd.read_csv(StringIO(file_content))
                if path.endswith("xlsx"):
                    data = pd.read_excel(
                        BytesIO(content_object.get()["Body"].read()), engine="openpyxl"
                    )
                if path.endswith("txt") or path.endswith("py") or path.endswith("log"):
                    data = content_object.get()["Body"].read().decode("utf-8")
            except Exception as e:
                print(f"error {e} occured while reading the data from s3")

    except Exception as e:
        logger.error(f"Error:{e}")
        raise ValueError(f"{path} is not a valid csv or xlsx file.")

    return data


def write_data(file_path, content):
    s3_client = boto3.client("s3")
    base_name = os.path.basename(file_path)
    if ".csv" in base_name:
        csv_buffer = StringIO()
        content.to_csv(csv_buffer, index=False)
        s3_client.put_object(
            Bucket=cloud_config.domain_storage.account_name,
            Key=file_path,
            Body=csv_buffer.getvalue(),
        )
    # elif "data_dictionary" in base_name:
    #     # buffer = StringIO()
    #     txt_data = json.dumps(content, indent=4)
    #     # buffer.seek(0)
    #     # buffer = BytesIO(self.output_table_dict.encode("utf-8"))
    #     s3_client.put_object(
    #         Bucket=cloud_config.domain_storage.account_name,
    #         Key=file_path,
    #         Body=txt_data,
    #     )
    else:
        if ".json" in base_name or "data_dictionary" in base_name:
            content = json.dumps(content, indent=4)
        # buffer = BytesIO(content.encode("utf-8"))
        s3_client.put_object(
            Bucket=cloud_config.domain_storage.account_name, Key=file_path, Body=content
        )


def process_historical_guidelines(guideline_file, guidelines):

    historical_guide = read_data(
        guideline_file, cloud_config.domain_storage.account_name
    )
    modified_guideline = re.sub(
        r"Guideline 1:.*?Athena_guide_sep",
        f"Guideline 1: {historical_guide}Athena_guide_sep",
        guidelines,
        flags=re.DOTALL,
    )
    modified_guideline = modified_guideline.replace("Athena_guide_sep", "")
    modified_guideline = process_guidelines(modified_guideline)

    return modified_guideline


def read_and_process_data(
    path,
    logger_name: str = "QueryInsights",
    fs=None,
    **kwargs,
):
    """
    Reads the excel/csv file and formats the column names and the date columns to load it to a SQL database

    Parameters
    ----------
    path : str
        path to the csv or xlsx file
    logger_name : str
        Logger name, by default ``QueryInsights``
    fs : fsspec.filesystem, optional
        Filesystem of the url, by default ``None``

    Returns
    -------
    pd.DataFrame
        formatted data frame that can be loaded to SQL DB

    Raises
    ------
    ValueError
        if the path is not a valid csv or xlsx file.
    """

    # Create logger
    logger = logging.getLogger(logger_name)

    try:
        fs = fs or fsspec.filesystem("file")
        if path.endswith("csv"):
            # TODO: Changed this back since it isn't working in Windows. Not sure if this will work with Azure blob or not
            # data = pd.read_csv(prefix_url+path, storage_options=storage_options)
            with fs.open(path, mode="rb") as fp:
                data = pd.read_csv(fp, **kwargs)
        if path.endswith("xlsx"):
            with fs.open(path, mode="rb") as fp:
                data = pd.read_excel(fp, engine="openpyxl", **kwargs)
    except Exception as e:
        logger.error(f"Error:{e}")
        raise ValueError(f"{path} is not a valid csv or xlsx file.")

    return data


def read_and_process_data_dictionary(data_dict):
    """
    edit the column names in the data dictionary so that they match with the formatted data frame

    Parameters
    ----------
    data_dict : dict
        contains data dictionary with column names and their descriptions

    Returns
    -------
    dict
        data dictionary with updated column names
    """
    pattern = re.compile(r"[^\w]")
    for column in data_dict["columns"]:
        column["name"] = pattern.sub("_", column["name"].lower().replace(" ", "_"))
    return data_dict


def load_data_dictionary(path, fs=None, **kwargs):
    """
    read the data dictionary JSON files

    Parameters
    ----------
    path : str
        path to the JSON file
    fs : fsspec.filesystem, optional
        Filesystem of the url, by default ``None``

    Returns
    -------
    dict
        data dictionary that can be used in prompts

    Raises
    ------
    ValueError
        if the path is not a valid JSON file.
    """
    try:
        fs = fs or fsspec.filesystem("file")
        with fs.open(path, "r") as f:
            data_dictionary = json.load(f, **kwargs)
    except Exception as e:
        print(f"error {e} occured while reading the data dictionary")
    return read_and_process_data_dictionary(data_dictionary)


def load_data_dictionary_from_s3(bucket, path):
    """
    read data dictionary JSON files from S3.
    Parameters
    ----------
    path : str
        path to the JSON file
    Returns
    -------
    dict
        data dictionary that can be used in prompts

    Raises
    ------
    ValueError
        if the path is not a valid JSON file.
    """
    try:
        s3_client = boto3.resource("s3")
        content_object = s3_client.Object(bucket, path)
        file_content = content_object.get()["Body"].read().decode("utf-8")
        json_content = json.loads(file_content)
        return json_content
    except Exception as e:
        print(f"error {e} occured while reading the data dictionary")


def load_to_in_memory_db(df, table_name, conn):
    """
    Loads the data frame to an in memory SQLite database

    Parameters
    ----------
    df : pd.DataFrame
        data frame to be loaded
    table_name : str
        name of the table to be created
    conn : sqlite3.Connection
        connection to the database

    Returns
    -------
    None
    """
    df.fillna("None", inplace=True)

    column_names = df.columns.tolist()

    sql_data_types = {
        "object": "TEXT",
        "int64": "INTEGER",
        "float64": "REAL",
        "datetime64[ns]": "TEXT",
        "bool": "INTEGER",
    }

    columns = ",\n".join(
        [
            f"{col_name} {sql_data_types[str(df[col_name].dtype)]} DEFAULT NULL"
            for col_name in column_names
        ]
    )
    create_table_query = f"CREATE TABLE {table_name} (\n{columns}\n);"
    conn.execute(create_table_query)

    insert_values = ", ".join(
        [
            f"({', '.join([f'{repr(val)}' for val in row.values])})"
            for _, row in df.iterrows()
        ]
    )
    insert_query = (
        f"INSERT INTO {table_name} ({', '.join(column_names)}) VALUES {insert_values};"
    )
    conn.execute(insert_query)
    return None


def generate_paraphrases(original_sentence, num_paraphrases=5, temperature=0):
    """Generates paraphrases for the input sentence using OpenAI's GPT-3 API.

    Parameters
    ----------
    original_sentence : str
        input sentence for which paraphrases are to be generated
    num_paraphrases : int, optional
        number of paraphrases to be generated, by default 5
    temperature : int, optional
        temperature parameter for the GPT-3 API, by default 0

    Returns
    -------
    list
        list of paraphrases
    """
    prompt = f"""Generate {num_paraphrases} different paraphrased sentences split by ';'
                 for the following sentence: '{original_sentence}'\n"""

    model_engine = "text-davinci-003"
    temperature = temperature
    max_tokens = 1000
    stop_sequence = None

    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop_sequence,
    )
    paraphrased_sentence = response.choices[0].text.strip()

    paraphrased_sentence = paraphrased_sentence.split(";")
    paraphrased_sentence = [p.replace("\n", "").strip() for p in paraphrased_sentence]
    return paraphrased_sentence


def convert_df_to_csv_string(input_df: pd.DataFrame) -> str:
    """Converts input dataframe to csv as a string variable.

    Parameters
    ----------
    input_df : pd.DataFrame
        Input dataframe

    Returns
    -------
    str
        csv string variable
    """
    table_string = input_df.to_csv(index=False)
    return table_string


def get_gpt_token_count(input_data: str, model: str):
    """Returns the number of tokens in the input data.

    Parameters
    ----------
    input_data : str
        Input data
    model : str
        Model name

    Returns
    -------
    int
        Number of tokens in the input data
    """
    enc = tiktoken.encoding_for_model(model)
    num_tokens = len(enc.encode(input_data))
    return num_tokens


def download_nltk_data():
    """Function to download nltk resources automatically.

    Returns
    -------
    bool
        True, if there is an error in downloading nltk data. False, otherwise.
    """
    # TODO: fsspec: Integrate fsspec in nltk data find? Or it can read from local storage?
    error_flag = False
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError as e:
        print(f"Error:{e}")
        download_flag = nltk.download("punkt", quiet=True)
        if not download_flag:
            error_flag = True

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError as e:
        print(f"Error:{e}")
        download_flag = nltk.download("stopwords", quiet=True)
        if not download_flag:
            error_flag = True
    return error_flag


def download_spacy_data(en_core_web_model: str = "en_core_web_lg") -> bool:
    """Function to download spacy automatically.

    Parameters
    ----------
    en_core_web_model: str, optional
        Specify which Spacy web model is to be downloaded.
        For example "en_core_web_sm", "en_core_web_md", "en_core_web_lg", by default "en_core_web_lg"

    Returns
    -------
    bool
        True, if there is an error in downloading spacy data. False, otherwise.
    """
    # TODO: fsspec: Integrate fsspec in nltk data find? Or it can read from local storage?
    error_flag = False

    # Check for spacy en_core_web download
    available_flag = spacy.util.is_package(en_core_web_model)
    if not available_flag:
        try:
            spacy.cli.download(en_core_web_model)
        except (SystemExit, Exception) as e:
            print(f"Error:{e}")
            error_flag = True
    return error_flag


def rate_limit_error_handler(
    time_delay: int,
    max_retries: int,
    logger: logging.Logger = None,
    errors: tuple = (openai.RateLimitError,),
):
    """Retry request after sleeping for x seconds.

    Parameters
    ----------
    logger : logging.Logger
        Logger object passed from QueryInsights.
    time_delay : int
        time in seconds to sleep.
    max_retries : int
        Maximum number of retries whenever we face Ratelimiterror
    errors : tuple, optional
        tuple of openAI errors, by default (openai.error.RateLimitError,)

    Returns
    -------
    function
        decorator function

    Raises
    ------
    Exception
        Maximum number of retries exceeded.
    """
    logger = logging.getLogger(MYLOGGERNAME)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Initialize variables
            num_retries = 0

            # Loop until a successful response or max_retries is hit or an exception is raised
            while True:
                try:
                    return func(*args, **kwargs)

                # Retry on specified errors
                except errors:
                    # Increment retries
                    num_retries += 1
                    logger.info(f"Trial number: {num_retries} failed.")

                    # Check if max retries has been reached
                    if num_retries > max_retries:
                        logger.error(
                            f"Maximum number of retries ({max_retries}) exceeded."
                        )
                        raise Exception(
                            f"Maximum number of retries ({max_retries}) exceeded."
                        )

                    # Sleep for the delay
                    logger.info(
                        f"Request will be retried after sleeping for {time_delay} seconds."
                    )
                    time.sleep(time_delay)

                # Raise exceptions for any errors not specified
                except Exception as e:
                    raise e

        return wrapper

    return decorator


def read_text_file(filename, fs=None):
    """Reads the text file and returns the contents.

    Parameters
    ----------
    filename : str
        Path to the text file
    fs : fsspec.filesystem, optional
        Filesystem of the url, by default ``None``

    Returns
    -------
    str
        Contents of the text file
    """
    fs = fs or fsspec.filesystem("file")
    with fs.open(filename, "r") as f:
        contents = f.read()
    return contents


def convert_data_dictionary_to_pandas_df(data_dictionary: dict):
    """
    Converts the raw data dictionary to a pandas dataframe.

    Parameters
    ----------
    data_dictionary : dict
        Raw data dictionary with table_name and columns as Keys and Description/ID as values

    Returns
    -------
    data_dictionary_df
        Pandas dataframe
    """
    data_dictionary_df = pd.DataFrame()
    for key in data_dictionary.keys():
        dd_table = data_dictionary[key]["columns"]
        df = pd.DataFrame(dd_table)
        df["table_name"] = key
        data_dictionary_df = pd.concat([data_dictionary_df, df], axis=0)

    return data_dictionary_df


def capture_stdout_to_var(func: Callable[[str], None], kwargs: dict) -> Tuple[str, str]:
    """Captures stdout and stderr when we run code blocks to two variables and returns them.

    Parameters
    ----------
    func : Callable[[str], None]
        Function for which we are capturing the stdout and stderr
    kwargs : dict
        keyword arguments for the function.

    Returns
    -------
    Tuple[str, str]
        stdout and stderr.
    """
    f = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(f), redirect_stderr(err):
        func(**kwargs)

    # Get the stdout that gets populated when we run `func` into var `f`.
    out = f.getvalue()
    err_out = err.getvalue()

    return out, err_out


def save_results_as_json(
    question: str,
    predicted_answer: str = "",
    file_save_path: str = None,
    fs=None,
    **kwargs,
) -> None:
    """Saves Track 3 results in JSON format as below::

        {"id": id, "input": {"text": question}, "output": {"text": predicted_answer}}

    Parameters
    ----------
    question : str
        User question
    predicted_answer : str, optional
        Track 3 result, by default ""
    file_save_path : str, optional
        JSON file save path, by default None
    fs : fsspec.filesystem, optional
        Filesystem of the url, by default ``None``

    Raises
    ------
    ValueError
        if any of the argument is missing.
    """
    if file_save_path is None:
        raise ValueError("Save path of json must be given.")

    fs = fs or fsspec.filesystem("file")
    id_ = "1"

    result = {
        "id": id_,
        "input": {"text": question},
        "output": {"text": predicted_answer},
    }

    with fs.open(file_save_path, "w") as f:
        json.dump(result, f, **kwargs)

    return


def upload_data(exp):
    s3 = s3fs.S3FileSystem()
    s3_path = f"takeda-ipro/results/experiments/{exp}"
    local_path = f"../../data/output_folder/{exp}/"
    s3.put(local_path, s3_path, recursive=True)


def multiple_json_to_jsonl(json_folder_path: str, jsonl_file_path: str = None, fs=None):
    """Reads all JSON files in given folder and joins them together in a JSONL file.

    Parameters
    ----------
    json_folder_path : str
        Folder which will be traversed to search for all json files.
    jsonl_file_path : str, optional
        save path for resulting jsonl, by default None
    fs : fsspec.filesystem, optional
        Filesystem of the url, by default ``None``

    Returns
    -------
    None
    """
    if jsonl_file_path is None:
        jsonl_file_path = pp.join(json_folder_path, "all_user_query.jsonl")

    all_query_data = []
    fs = fs or fsspec.filesystem("file")
    # Get all json saved already in the path.
    for dirpath, _, files in fs.walk(json_folder_path):
        for file in files:
            if file.endswith(".json"):
                print("Json file found at", pp.join(dirpath, file))
                with fs.open(pp.join(dirpath, file), "r") as json_file:
                    json_data = json.load(json_file)
                    all_query_data.append(json_data)

    # Save these multiple json to jsonl
    with fs.open(jsonl_file_path, "w") as fout:
        for each_json_data in all_query_data:
            json.dump(each_json_data, fout)
            fout.write("\n")
    print("jsonl saved at ", jsonl_file_path)

    return


def get_table_names(conn, schema="sqlite_schema") -> List[str]:
    """
    Get table names in sql database

    Parameters
    ----------
    conn : sqlite3.Connection
        connection to the database

    Returns
    -------
    List[str]:
        List of table names present in sql database
    """
    cursor = conn.cursor()
    query = f"""SELECT name FROM {schema} WHERE type='table';"""
    cursor.execute(query)

    table_names = cursor.fetchall()
    table_names = [table[0] for table in table_names]
    return table_names


def get_dummy_data_dictionary(conn, table_name: str) -> dict:
    """
    Get dummy data dictionary for a given table

    Parameters
    ----------
    conn : sqlite3.Connection
        Connection to the database
    table_name : str
        Table name for the data dictionary

    Returns
    -------
    dict:
        Dummy data dictionary containing table name and column names.
        For example
        {
            "table_name": "table_name",
            "columns": [
                {
                    "name":"column1",
                    "description":"",
                },
                {
                    "name":"column2",
                    "description":"",
                },
            ]
        }
        In case table name is not present in the database, an empty dictionary will be returned as follows
        {
            "table_name" : "table_name",
            "columns": [
                {
                    "name": "",
                    "description": "",
                }
            ],
        }

    """
    cursor = conn.cursor()
    query = f"""SELECT * from {table_name}"""
    try:
        cursor.execute(query)
        column_names = list(map(lambda x: x[0], cursor.description))

        return {
            "table_name": table_name,
            "columns": [{"name": column, "description": ""} for column in column_names],
        }
    except sqlite3.OperationalError:
        return {
            "table_name": table_name,
            "columns": [
                {
                    "name": "",
                    "description": "",
                }
            ],
        }


def format_dataframe(df):
    """
    converts data frame into a front end usable format

    Parameters
    ----------
    df : pd.DataFrame
        data frame to be sent to front end

    Returns
    -------
    list
    converted rows from a dataframe into a list
    """

    # if the index is just the pandas default index, it can be dropped. Else, it will be added to the data frame as a column
    df = df.reset_index()
    if "index" in df.columns:
        del df["index"]

    # converting any date columns to string format before sending it in to reponse json(else a huge number is sent to tables instead of dates)
    date_columns = df.select_dtypes(include=["datetime64"]).columns.tolist()
    df[date_columns] = df[date_columns].astype(str)

    return df.to_json(orient="records")


def parse_today_date(
    date_str: str = None, format: str = "%d %B %Y", logger_name: str = "QueryInsights"
) -> str:
    """
    Given a date string in any format, parse it to a specified date format.

    Parameters
    ----------
    date_str : str
        Date string to be parsed
    format : str, optional
        Format to which the provided date needs to be formatted to, by default "%-d %B %Y"
    logger_name : str, optional
        Name of the logger. Used by other scripts in this package, by default "QueryInsights"

    Returns
    -------
    str
        Formatted date

    Raises
    ------
    ValueError
        If date format provided is invalid

    Examples
    --------
    >>> from query_insights.utils import parse_today_date

    >>> parse_today_date(None) # If None is provided, today's date is returned in Day Month Year format
        '6 August 2023'

    >>> parse_today_date("2023-08-04") # The function can also parse different formats of date automatically
        '4 August 2023'

    >>> parse_today_date("08/09/2023")
        '9 August 2023'

    >>> parse_today_date("Aug 15, 2023")
        '15 August 2023'

    >>> parse_today_date("2023-08-19T12:34:56")
        '19 August 2023'

    """
    logger = logging.getLogger(logger_name)
    try:
        if date_str is None:
            logger.debug("date_str is None, parsing current date")
            todays_date = datetime.datetime.now()
            formatted_date = todays_date.strftime(format)
        else:
            logger.debug(f"Received {date_str} as date to parse")
            parsed_date = dateutil.parser.parse(date_str)
            formatted_date = parsed_date.strftime(format)
        logger.debug(f"Formatted date: {formatted_date}")
        return formatted_date
    except Exception as e:
        logger.error("Error occured when parsing date string")
        logger.error(traceback.format_exc())
        raise ValueError("Invalid date format") from e


def get_word_chunks(doc, stop_words):
    """
    Extracts chunks of words from a given document 'doc' excluding stop words.
    Example:
    For document input of this sentence -
        How much quantity of FlavorCrave brand was shipped each week?
    Result - ['week', 'shipped', 'How much quantity', 'FlavorCrave brand']

    Parameters
    ----------
    doc : spacy.tokens.Doc
        The input document from which chunks will be extracted.
    stop_words : list
        A list of stop words to be excluded from the chunks.

    Returns
    -------
    list
        A list of chunks containing nouns, verbs, proper nouns, adjectives, and adverbs,
        excluding stop words and punctuation.
    """
    # doc = nlp(question)
    noun_chunks = [chunk.text for chunk in doc.noun_chunks]
    noun_chunks = [n for n in noun_chunks if n not in stop_words]

    allowed_postags = ["NOUN", "VERB", "PROPN", "ADJ", "ADV"]
    tokens = [
        token.text
        for token in doc
        if not token.is_stop and not token.is_punct and token.pos_ in allowed_postags
    ]
    all_chunks = list(set(noun_chunks + tokens))

    result = []
    for word in all_chunks:
        is_subset = any(
            word in other_word and word != other_word for other_word in all_chunks
        )
        if not is_subset:
            result.append(word)
    return result


class CustomExceptionHook:
    """
    This class redirects exceptions to predefined functions for both python and jupyter.
    The respective functions in the class is invoked in case of an error and the error message is logged.
    """

    def __init__(self, logger: logging.Logger):
        """
        Parameters
        ----------
        logger : logging.Logger
            Logger object passed from QueryInsights.
        """
        self.logger = logger

    def custom_excepthook(self, exc_type: type, exc_value, exc_traceback: traceback):
        """
        This function is invoked in case of exceptions when code is run in python shell
        Logs given errors to logger

        Parameters
        ----------
        exc_type : type
            Type of error
        exc_value : class
            Error class and message
        exc_traceback : traceback
            Traceback object with the error information
        """
        tb_formatted = traceback.format_exception(exc_type, exc_value, exc_traceback)
        self.logger.error(
            "The following uncaught error occurred"
            + "\n"
            + "".join(tb_formatted).strip()
        )

    def custom_excepthook_jupyter(
        self, shell, exc_type, exc_value, exc_traceback, tb_offset=None
    ):
        """
        This function is invoked in case of exceptions when code is run in jupyter notebooks
        Logs given errors to logger

        Parameters
        ----------
        shell : IPython.shell type
            Ipython shell object
        exc_type : type
            Type of error
        exc_value : class
            Error class and message
        exc_traceback : traceback
            Traceback object with the error information
        tb_offset : IPython object
            IPython offset object
        """
        tb_formatted = traceback.format_exception(exc_type, exc_value, exc_traceback)
        self.logger.error(
            "The following uncaught error occurred"
            + "\n"
            + "".join(tb_formatted).strip()
        )


def log_uncaught_errors(logger: logging.Logger):
    """
    Catches uncaught errors in code and logs to logger before exiting

    Parameters
    ----------
    logger : logging.Logger
        Logger object passed from QueryInsights.
    """
    customExceptionHook = CustomExceptionHook(logger)

    if is_jupyter():
        with suppress(Exception):
            # This will error out. Only added to avoid import error message in vscode
            from IPython import get_ipython

        get_ipython().set_custom_exc(
            (Exception,),
            customExceptionHook.custom_excepthook_jupyter,
        )
    else:
        sys.excepthook = customExceptionHook.custom_excepthook


def is_jupyter() -> bool:
    """Check if code is running in jupyter notebook or python interpreter

    Returns
    -------
    bool
        True if code is running in jupyter, False otherwise
    """
    try:
        with suppress(Exception):
            # This will error out. Only added to avoid import error message in vscode
            from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def generate_env_dict(
    cloud_storage_dict: DotifyDict = None, account_key: str = None
) -> Union[dict, None]:
    """
    From given cloud storage parameters, generate an environment dictionary which can be passed to subprocess.run as environment variables

    Parameters
    ----------
    cloud_storage_dict : DotifyDict, optional
        Dictionary containing cloud connection parameters, by default None
    account_key : str, optional
        Account key of the cloud storage provider, by default None

    Returns
    -------
    Union[dict, None]
        If cloud_storage_dict isn't None, return a dictionary containing environment variables, else return None
    """
    if (
        cloud_storage_dict["cloud_provider"] == "local"
        or cloud_storage_dict["cloud_provider"] == "s3"
    ):
        env = None
    else:
        env = {}
        env["AccountKey"] = "" if account_key is None else account_key
        # Below env variables are only required in Windows os for subprocess.run to work
        env["SYSTEMROOT"] = os.getenv("SYSTEMROOT", "")
        env["APPDATA"] = os.getenv("APPDATA", "")
    return env


def load_db_credentials(password_path, fs=None):
    """
    Loads the database password into environment variables

    Parameters
    ----------
    password_path : str
        this can be the path to a txt file that contains password, or the password itself
    fs : fsspec.filesystem, optional
        Filesystem of the url, by default ``None``

    Returns
    -------
    None
    """
    fs = fs or fsspec.filesystem("file")
    if fs.exists(password_path):
        # If we received file path of password, read the file, else directly pass in the password.
        with fs.open(password_path, "r") as f:
            password = f.read()
    else:
        password = password_path
    return password


class DataConverter:
    """
    This class have functionalities to convert data from one format to another format. e.g. dictionary to dataframe, dataframe to dictionary, dictionary to list of tuples
    """

    @staticmethod
    def json_to_dataframe(data_dictionary: dict) -> pd.DataFrame:
        """
        Json file(having column details for each table) will be converted to a dataframe where each row will
        have information related to one unique column, unique id will be assigned to each unique column.

        Parameters
        ----------
        json_data : dict
            Dictionary having details for each table

        Returns
        -------
        pd.DataFrame
            Returns dataframe having unique index for unique colum name and other information related to that column,
            e.g. column description, table name etc.

        """

        logger = logging.getLogger("dataframe_with_column_details")
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            stream=sys.stdout,
        )
        try:
            rows = []
            unique_id_counter = 1

            # Iterate over each table in the JSON
            for table_key, table_value in data_dictionary.items():
                table_name = table_value["table_name"]
                for column in table_value["columns"]:
                    # Extract column data
                    column_name = column["name"]
                    column_description = column.get("description", "")
                    id_flag = column.get("id", "No")

                    # Create a dictionary for each row
                    row = {
                        "unique_id": unique_id_counter,
                        "column_name": column_name,
                        "column_description": column_description,
                        "table_name": table_name,
                        "id": id_flag,
                    }
                    rows.append(row)
                    unique_id_counter += 1

            created_dataframe = pd.DataFrame(rows)

            # concatenate the rows column name, column description and table name into a new column called combined
            created_dataframe["combined"] = (
                created_dataframe["column_name"]
                + " "
                + created_dataframe["column_description"]
                + " "
                + created_dataframe["table_name"]
            )
            created_dataframe.to_csv("dataframe.csv")
            # Create DataFrame
            logger.info(f"DataFrame having column details created")
            return created_dataframe
        except Exception as e:
            logger.error(f"DataFrame having column details not created. Error :\n {e}")

    @staticmethod
    def dataframe_to_json(df, top_result_ids: list) -> dict:
        """
        A dictionary will be created from the column indexes after mapping these indexes with the dataframe
        that was created from data dictionary

        Parameters
        ----------
        top_result_ids : list
            list of column indexes

        Returns
        -------
        dict
            This dictionary consists table, column name, column description.

        """
        logger = logging.getLogger("data_dict_selected_tables")
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            stream=sys.stdout,
        )

        try:
            json_data = {}
            df = df[df["unique_id"].isin(top_result_ids)]
            for _, row in df.iterrows():
                table_name = row["table_name"]
                column_info = {
                    "name": row["column_name"],
                    "description": row["column_description"],
                }
                if pd.notna(row["id"]) and row["id"] != "No":
                    column_info["id"] = row["id"]

                if table_name not in json_data:
                    json_data[table_name] = {"table_name": table_name, "columns": []}

                json_data[table_name]["columns"].append(column_info)
            logger.info(f"data dictionary for selected tables generated")
            return json_data

        except Exception as e:
            logger.error(
                f"data dictionary not generated for selected tables. Error :\n {e}"
            )


def get_s3_client():
    """Initialize and return the S3 client."""
    return boto3.client("s3")


def upload_to_s3(s3_client, local_file_path, bucket_name, s3_key):
    """Upload file to an S3 bucket"""
    s3_client.upload_file(local_file_path, bucket_name, s3_key)


def download_from_s3(s3_client, bucket_name, s3_key, local_file_path):
    """Download a file from an S3 bucket."""
    s3_client.download_file(bucket_name, s3_key, local_file_path)


def fetch_s3_details(question_folder):
    # fteching details of s3 bucket
    bucket_name = cloud_config.domain_storage.account_name
    s3_folder = question_folder
    pickle_file = "track_dict.pkl"
    s3_key = f"{s3_folder}/{pickle_file}"
    local_pickle_path = "/tmp/track_dict.pkl"
    return bucket_name, s3_key, local_pickle_path
