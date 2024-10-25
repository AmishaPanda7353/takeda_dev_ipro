import logging
import os
import shutil
import sqlite3
import tempfile
import traceback
from typing import Dict, Optional

import fsspec
import mysql.connector
import pandas as pd

# from ../../../../configs.database import database
from core.database.database_client import DatabaseClient
from mysql.connector import errorcode
from src.query_insights.utils.utils import load_db_credentials

# from core.utils.read_config import config

MYLOGGERNAME = "QueryInsights"


class MySqlClient(DatabaseClient):
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

        logging.info(f"Running from MySqlClient: {config.host}")
        # MySql DB Connection
        logging.info("Establishing MySqlClient DB connection ...")
        try:
            conn = mysql.connector.connect(
                host=config.host,
                user=config.username,
                password=load_db_credentials(config.password_path),
                database=config.database_name,
                port=config.port,
            )
            mycursor = conn.cursor()
            mycursor.execute("show databases")
            all_tables = mycursor.fetchall()
            if config.database_name is None:
                self.logger.error("Database name isn't given. Expected a string")
                raise ValueError("Database name isn't given. Expected a string")
            if (config.database_name,) in all_tables:
                self.logger.info(f"Connected to MySQL database: {config.database_name}")
                conn = mysql.connector.connect(
                    host=config.host,
                    user=config.username,
                    password=load_db_credentials(config.password_path),
                    database=config.database_name,
                    port=config.port,
                )
            else:
                mycursor.execute(f"CREATE DATABASE {config.database_name}")
                self.logger.info(f"Created new database: {config.database_name}")
                conn = mysql.connector.connect(
                    host=config.host,
                    user=config.username,
                    password=load_db_credentials(config.password_path),
                    database=config.database_name,
                    port=config.port,
                )
            return conn

        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                self.logger.error("Something is wrong with your user name or password")
            else:
                self.logger.error(f"Error connecting to MySQL: {err}")

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

        self.output_table = pd.read_sql_query(query, connection)

        # cursor = connection.cursor()

        # cursor.execute(query)

        # results = cursor.fetchall()
        print(self.output_table)
        return self.output_table

    def read_query(
        self, connection, query: str, fetch_columns: bool = False, table_name: str = ""
    ):
        pass
