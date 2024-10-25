import logging
import os
import shutil
import sqlite3
import tempfile
import traceback
from typing import Dict, Optional

import fsspec
import pandas as pd
import mysql.connector
from mysql.connector import errorcode
from sqlalchemy import create_engine, text
from src.query_insights.utils.utils import load_db_credentials
# from ../../../../configs.database import database
from core.database.database_client import DatabaseClient
# from core.utils.read_config import config

MYLOGGERNAME = "QueryInsights"


class AthenaClient(DatabaseClient):
    def __init__(self):
        pass

    def create_database_connection(self, config, fs=None, update_table=False):
        """
        Creates a connection to a SQLite database.

        Args:
            config (dict): The configuration details for the database connection.
            fs (fsspec.AbstractFileSystem, optional): The filesystem object used for file operations. Defaults to None.

        Returns:
            sqlite3.Connection: The connection object to the SQLite database.

        Raises:
            ValueError: If there is an error loading the SQLite database.

        """
        self.logger = logging.getLogger(MYLOGGERNAME)
        self.fs = fs or fsspec.filesystem("file")

        logging.info(f"Running from AthenaClient: {config.athena_conn_string}")
        # MySql DB Connection
        logging.info("Establishing AthenaClient DB connection ...")
        try:
            conn = create_engine(config.athena_conn_string)
            connect = conn.connect()
            results = connect.execute(text("SHOW SCHEMAS;"))
            results.fetchall()
            return connect
        except Exception as err:
            self.logger.error(f"Error connecting to Athena DB : {err}")

    def execute_query(
        self,
        connection,
        query,
        return_value: bool = False,
        content: Optional[Dict] = None,
    ):
        """
        Executes the given SQL query on the provided database connection.

        Args:
            connection: The database connection object.
            query: The SQL query to be executed.
            return_value: A boolean indicating whether to return the query results or not.
            content: Optional dictionary containing additional content.

        Returns:
            If return_value is True, returns the query results as a pandas DataFrame.
            Otherwise, returns None.
        """
        self.output_table = pd.read_sql_query(text(query), connection)
        return self.output_table

    def read_query(
        self, connection, query: str, fetch_columns: bool = False, table_name: str = ""
    ):
        pass
