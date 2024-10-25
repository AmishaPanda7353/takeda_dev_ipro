import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional

import fsspec

from core.utils.read_config import config, domain_db_config

MYLOGGERNAME = "QueryInsights"


class DatabaseClient(ABC):
    """
    Abstract base class for a database client.

    This class provides a common interface for creating a database connection,
    executing queries, and reading query results.

    Attributes:
        logger (Logger): The logger object for logging messages.
        database (str): The name of the database to connect to.

    Methods:
        create_database_connection: Creates a connection to the database.
        execute_query: Executes a database query.
        read_query: Reads the results of a database query.
    """

    def __init__(self):
        self.logger = logging.getLogger(MYLOGGERNAME)

        self.database = domain_db_config.domain_database

        list_of_databases = ["sqlite", "postgres", "hivedb","mysql", "athena"]

        if self.database not in list_of_databases:
            self.logger.error(
                f"Given wrong database {self.database}. Expected databases are sqlite, postgres,hivedb"
            )

            raise ValueError(
                f"Given wrong database {self.database}. Expected databases are sqlite, mysql"
            )

    @abstractmethod
    def create_database_connection(self, config, fs=None, update_table=False):
        """
        Creates a connection to the database.

        Args:
            config (dict): The configuration details for the database connection.
            fs (FileSystem, optional): The file system object to use for accessing the database files.

        Raises:
            ValueError: If the database path doesn't exist.
        """
        raise NotImplementedError(
            "create_database_connection method must be implemented by subclass"
        )

    @abstractmethod
    def execute_query(self, **kwargs):
        """
        Executes a database query.

        Args:
            **kwargs: Additional keyword arguments for the query.

        Returns:
            The result of the query execution.
        """
        raise NotImplementedError(
            "execute_query method must be implemented by subclass"
        )

    @abstractmethod
    def read_query(self, **kwargs):
        """
        Reads the results of a database query.

        Args:
            **kwargs: Additional keyword arguments for reading the query results.

        Returns:
            The query results.
        """
        raise NotImplementedError(
            "read_query method must be implemented by subclass"
        )
