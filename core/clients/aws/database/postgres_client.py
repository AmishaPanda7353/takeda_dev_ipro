import json
import logging
from typing import Dict, Optional

import psycopg2

from core.database.database_client import DatabaseClient
# from core.utils.read_config import db_secrets, secrets_config

logging.getLogger().setLevel(logging.INFO)


class AzurePostgresClient(DatabaseClient):
    def create_database_connection(self, config, fs=None, update_table=False):
        """
        Creates a connection to a PostgreSQL database using the provided configuration.

        Args:
            config (Config): The configuration object containing the database connection details.
            fs (FileSystem): The file system object to be used for file operations (optional).

        Returns:
            connection (psycopg2.extensions.connection): The connection object representing the PostgreSQL database connection.

        Raises:
            Exception: If the database connection fails.
        """
        # logging.info(f"Instantiated PostgresClient")
        try:
            # Postgres DB Connection
            logging.info("Establishing PostgreSQL DB connection ...")
            connection = psycopg2.connect(
                database=config.database_name,
                user=config.username,
                password=config.password,
                host=config.host,
                port=config.port,
            )
            # logging.info("Connection Established!!")
            return connection

        except Exception as err:
            # logging.error(f"Error: '{err}'")
            raise Exception("Database Connection Failed.")

    def execute_query(
        self,
        connection,
        query,
        return_value: bool = False,
        content: Optional[Dict] = None,
    ):
        """
        Execute DML queries.

        Args:
            connection: The database connection object.
            query: The SQL query to be executed.
            return_value: A boolean indicating whether to return a value from the query execution.
            content: Optional dictionary containing additional content for the query.

        Returns:
            If `return_value` is True, the primary key value returned from the query execution.
            Otherwise, None.

        Raises:
            Exception: If the database query execution fails.
        """
        cursor = connection.cursor()
        try:
            if content is not None:
                # Change starts
                c = (
                    json.dumps(content)
                    .replace("\n", " ")
                    .replace("'", " ")
                )
                print("******************************")
                print(c)
                cursor.execute(query, (content["category"], c))
                # Change ends
            else:
                cursor.execute(query)
            if return_value:
                primary_key_value = cursor.fetchone()[0]
                connection.commit()
                return primary_key_value
            else:
                connection.commit()
        except Exception as err:
            logging.error(f"Query: ", query)
            logging.error(f"Error: '{err}'")
            raise Exception("Database Query Execution Failed.")

    def read_query(
        self, connection, query: str, fetch_columns: bool = False, table_name: str = ""
    ):
        """
        Args:
            connection: The database connection object.
            query (str): The SQL query to execute.
            fetch_columns (bool, optional): Whether to fetch the column names. Defaults to False.
            table_name (str, optional): The name of the table. Required if fetch_columns is True.

        Returns:
            tuple or list: The result of the query execution. If fetch_columns is True, returns a tuple
            containing the query result and the column names as a list. Otherwise, returns just the query result.

        Raises:
            Exception: If there is an error executing the query.

        """
        cursor = connection.cursor()
        result = None
        try:
            cursor.execute(query)
            result = cursor.fetchall()
            if fetch_columns:
                columns = list(
                    [row.column_name for row in cursor.columns(table=table_name)]
                )
                return result, columns
            else:
                return result
        except Exception as err:
            logging.error("Query: ", query)
            logging.error(f"Error: '{err}'")
            raise Exception("Database Query Retrieval Failed.")
