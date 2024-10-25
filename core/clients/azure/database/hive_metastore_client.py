import json
import logging

import fsspec
from pyspark.sql import SparkSession

from core.database.database_client import DatabaseClient
from core.utils.read_config import domain_db_config

spark = SparkSession.builder.getOrCreate()
from typing import Dict, Optional

MYLOGGERNAME = "QueryInsights"


class HiveMetaStore(DatabaseClient):
    def __init__(self):
        pass

    def create_database_connection(self, config, fs=None, update_table=False):
        """
        Creates a connection to a SQLite database.

        Args:
            config (dict): The configuration details for the database connection.
            fs (fsspec.AbstractFileSystem, optional): The filesystem object used for file operations. Defaults to None.

        Returns:
            None

        Raises:
            ValueError: If there is an error loading the HivemetastoreDB database.

        """

        print(f"Running from HiveMetastoreClient: {config}")

        if update_table:
            delta_parent_path = config.delta_parent_path

            tables = config.input_table_path

            # Hive Metastore DB Connection
            print("Establishing Hive Metastore DB connection ...")

            for table in list(tables.keys()):
                # Loading data from paths
                try:
                    df = spark.read.format("delta").load(tables[table]["path"])

                except Exception:
                    try:
                        df = spark.read.format("parquet").load(tables[table]["path"])
                    except Exception as e:
                        raise ValueError(f"{table} is not a valid table")

                # Slicing required columns
                df = df.select(tables[table]["columns"])
                table_name = config.database_name + "." + table

                # Getting column name and its dtype
                schema = df.schema.json()
                schema = json.loads(schema)

                col_names = []
                dtypes = []
                for colums in schema["fields"]:
                    col_names.append(colums["name"])
                    dtypes.append(colums["type"])

                delta_path = delta_parent_path + table

                # Create table query
                column_dtype = ", ".join(
                    [f"{col} {dt}" for col, dt in zip(col_names, dtypes)]
                )
                spark.sql(
                    f'CREATE TABLE IF NOT EXISTS {table_name}  ({column_dtype}) USING DELTA LOCATION "{delta_path}" TBLPROPERTIES (delta.deletedFileRetentionDuration = "interval 45 days")  COMMENT "EXT TABLE";'
                )

                df.write.format("delta").mode("overwrite").option(
                    "overwriteSchema", "true"
                ).saveAsTable(table_name)

        return

    def execute_query(
        self,
        connection,
        query,
        return_value: bool = False,
        content: Optional[Dict] = None,
    ):
        """
        Executes the given SQL query on the provided database name.

        Args:
            connection: The database connection object.
            query: The SQL query to be executed.
            return_value: A boolean indicating whether to return the query results or not.
            content: Optional dictionary containing additional content.

        Returns:
            If return_value is True, returns the query results as a pandas DataFrame.
            Otherwise, returns None.
        """

        spark.sql(f"USE {domain_db_config.database_name}")
        df = spark.sql(query)
        df_pd = df.toPandas()
        return df_pd

    def read_query(
        self, connection, query, fetch_columns: bool = False, table_name: str = ""
    ):
        pass
