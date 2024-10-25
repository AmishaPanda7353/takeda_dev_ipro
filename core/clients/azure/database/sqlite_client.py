import logging
import os
import shutil
import sqlite3
import tempfile
import traceback
from typing import Dict, Optional

import fsspec
import pandas as pd

from core.database.database_client import DatabaseClient
from core.utils.read_config import config

MYLOGGERNAME = "QueryInsights"


class SqliteClient(DatabaseClient):
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

        logging.info(f"Running from SqliteClient: {config}")
        domain_db_path = config.db_file_path

        # Sqlite DB Connection
        logging.info("Establishing SqliteClient DB connection ...")
        if self.fs.protocol == "file":
            try:
                conn = sqlite3.connect(domain_db_path, check_same_thread=False)
            except Exception as e:
                self.logger.error(f"Error occurred when loading sqlite3 database: {e}")
                self.logger.error(traceback.format_exc())
                raise ValueError("Unable to load sqlite3 db")
            self.logger.info("Sqlite3 database is loaded")
        else:
            tmp_path = os.path.join(os.getcwd(), "tmp")
            if not os.path.exists(tmp_path):
                os.mkdir(tmp_path)

            if os.name == "nt":
                temp_path = tempfile.TemporaryDirectory()
                # TODO: fsspec: fs.copy not working in windows, so using fs.get (which is working in Windows and Linux) which we need to test fs.get upon deploying to azure functions
                # https://filesystem-spec.readthedocs.io/en/latest/copying.html
                self.fs.get(
                    domain_db_path,
                    os.path.join(temp_path.name, os.path.basename(temp_path.name)),
                )
                try:
                    _conn = sqlite3.connect(
                        os.path.join(temp_path.name, os.path.basename(temp_path.name)),
                        check_same_thread=False,
                    )
                    conn = sqlite3.connect(":memory:", check_same_thread=False)
                    _conn.backup(conn)
                    _conn.close()
                except Exception:
                    self.logger.error(
                        "Error occurred when loading sqlite3 database in Windows"
                    )
                    self.logger.error(traceback.format_exc())
                    shutil.rmtree(temp_path.name)
                    raise ValueError("Unable to load sqlite3 db")
                self.logger.info("Sqlite3 database is loaded")
                shutil.rmtree(temp_path.name)
            else:
                with self.fs.open(domain_db_path, "rb") as f:
                    # TODO: Do more research on tempfiles and why the below commented code isn't working in windows. If we can make this work, we can use the same in track 3 also
                    with tempfile.NamedTemporaryFile(suffix=".db") as fp:
                        fp.write(f.read())
                        try:
                            conn = sqlite3.connect(fp.name, check_same_thread=False)
                            logging.info("established connection->sql cloud")
                        except Exception:
                            self.logger.error(
                                "Error occurred when loading sqlite3 database"
                            )
                            self.logger.error(traceback.format_exc())
                            raise ValueError("Unable to load sqlite3 db")
                self.logger.info("Sqlite3 database is loaded")

        return conn

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

        return self.output_table

    def read_query(
        self, connection, query: str, fetch_columns: bool = False, table_name: str = ""
    ):
        pass
