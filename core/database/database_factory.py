from core.database.database_client import DatabaseClient


class DatabaseFactory:
    def __init__(self, database_service: DatabaseClient) -> None:
        self.database_service = database_service

    def create_database_connection(self, config, fs=None, update_table=False):
        """
        Creates a database connection using the provided configuration.

        Args:
            config (dict): The configuration parameters for the database connection.
            fs (FileSystem, optional): The file system object to use for the connection. Defaults to None.

        Returns:
            DatabaseConnection: The created database connection object.
        """
        return self.database_service.create_database_connection(
            config=config, fs=fs, update_table=False
        )

    def execute_query(self, connection, query, return_value=None, content=None):
        """
        Executes the given query on the specified database connection.

        Args:
            connection: The database connection to execute the query on.
            query: The SQL query to execute.
            return_value: Optional. The type of result to return. Defaults to None.
            content: Optional. Additional content to pass to the database service. Defaults to None.

        Returns:
            The result of the query execution, based on the specified return_value.

        """
        return self.database_service.execute_query(
            connection=connection,
            query=query,
            return_value=return_value,
            content=content,
        )

    def read_query(self, connection, query, fetch_columns=None, table_name=None):
        """
        Executes a read query on the database.

        Args:
            connection: The database connection object.
            query: The SQL query to execute.
            fetch_columns: Optional. The columns to fetch from the query result.
            table_name: Optional. The name of the table being queried.

        Returns:
            The result of the query execution.

        """
        return self.database_service.read_query(
            connection=connection,
            query=query,
            fetch_columns=fetch_columns,
            table_name=table_name,
        )
