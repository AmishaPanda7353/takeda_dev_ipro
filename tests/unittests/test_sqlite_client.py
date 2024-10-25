from unittest.mock import MagicMock, patch

from core.clients.azure.database.sqlite_client import SqliteClient
import unittest
from core.database.database_factory import DatabaseFactory
from core.utils.client_utils import get_database_client
import unittest
from dynaconf import Dynaconf


class TestSqliteClient(unittest.TestCase):
    def setUp(self):
        """
        Set up the test case by creating a mock connector and initializing the SqliteClient instance.
        """
        self.mock_connector = MagicMock()
        self.client = SqliteClient()
    
    @patch("core.clients.azure.database.sqlite_client.sqlite3.connect")
    def test_database_connection(self, mock_get_connection):
        """
        Test case for establishing a database connection using a dummy configuration.

        Args:
            mock_get_connection: Mock object for the sqlite3.connect function.

        Returns:
            None
        """
        
        # Create a mock connector object
        mock_conn = mock_get_connection.return_value


        # Mocking actual configs
        dummy_db_config = Dynaconf()
        domain_database = "sqlite"
        
        data = {"db_file_path": "databasepath.db"}
        dummy_db_config.update(data)

        # Calling clients and establishing connection using dummy config
        database_client = get_database_client(database_type=domain_database)
        db_factory = DatabaseFactory(database_client)

        # Test Case: Establishing connection using dummy config
        test_conn = db_factory.create_database_connection(config=dummy_db_config)

        # Ensure that psycopg2.connect is called once with desired parameters
        mock_get_connection.assert_called_once_with("databasepath.db", check_same_thread=False)

        self.assertEqual(test_conn, mock_conn)
    
    @patch("core.clients.azure.database.sqlite_client.pd.read_sql_query")
    def test_execute_query(self, mock_read_sql_query):
        """
        Test case for the execute_query method of the SQLite client.

        This test verifies that the execute_query method correctly calls the
        read_sql_query function with the provided query and connection, and
        returns the expected result.

        Args:
            self: The test case instance.
            mock_read_sql_query: The mock object for the read_sql_query function.

        Returns:
            None
        """
        connection = MagicMock()
        query = "SELECT * FROM test_table"
        return_value = self.client.execute_query(connection, query)

        mock_read_sql_query.assert_called_once_with(query, connection)
        self.assertEqual(return_value, mock_read_sql_query.return_value)

    def test_read_query(self):
        """
        Test case for the read_query method.

        This method tests the functionality of the read_query method in the SQLiteClient class.
        It verifies that the method correctly executes a read query on the SQLite database.

        """
        # TODO: Implement test case for read_query method
        pass


if __name__ == "__main__":
    unittest.main()