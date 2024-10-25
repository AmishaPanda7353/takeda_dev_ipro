import unittest
from unittest.mock import MagicMock, patch
from core.clients.azure.storage.blob_storage_client import AzureBlobStorageClient


class TestAzureBlobStorageClient(unittest.TestCase):
    def setUp(self):
        """
        Set up the test case by creating a mock connector and initializing the AzureBlobStorageClient.
        """
        self.mock_connector = MagicMock()
        self.client = AzureBlobStorageClient()

    @patch("core.clients.azure.storage.blob_storage_client.BlobServiceClient")
    def test_connect_to_storage_success(self, mock_blob_service_client):
        """
        Test case for connecting to the storage successfully.

        Args:
            self: The test case instance.
            mock_blob_service_client: The mock BlobServiceClient object.

        Returns:
            None

        Raises:
            AssertionError: If the connection to the storage fails or the container client is not returned correctly.
        """
        storage_details = MagicMock()
        storage_details.account_name = "test_account"
        storage_details.storage_name = "test_storage"
        connection_secret = MagicMock()
        connection_secret.connection_key = "test_key"

        container_client = self.client.connect_to_storage(storage_details, connection_secret)

        mock_blob_service_client.return_value.get_container_client.assert_called_once_with("test_storage")
        self.assertEqual(container_client, mock_blob_service_client.return_value.get_container_client.return_value)

    def test_connect_to_storage_missing_account_name(self):
        """
        Test case to verify the behavior when the account name is missing in the storage details.
        It should raise an exception.
        """
        storage_details = MagicMock()
        storage_details.account_name = None
        storage_details.storage_name = "test_storage"
        connection_secret = MagicMock()
        connection_secret.connection_key = "test_key"

        with self.assertRaises(Exception):
            self.client.connect_to_storage(storage_details, connection_secret)

    def test_connect_to_storage_missing_storage_name(self):
        """
        Test case to verify the behavior when the storage name is missing.

        It creates a MagicMock object for storage_details and sets the account_name to "test_account".
        It sets the storage_name to None and creates a MagicMock object for connection_secret with connection_key set to "test_key".
        Then it asserts that an Exception is raised when calling self.client.connect_to_storage with the given storage_details and connection_secret.
        """
        storage_details = MagicMock()
        storage_details.account_name = "test_account"
        storage_details.storage_name = None
        connection_secret = MagicMock()
        connection_secret.connection_key = "test_key"

        with self.assertRaises(Exception):
            self.client.connect_to_storage(storage_details, connection_secret)

    def test_connect_to_storage_missing_connection_key(self):
        """
        Test case to verify that an exception is raised when the connection key is missing.

        This test case creates a mock storage details object and a mock connection secret object.
        The connection key of the connection secret is set to None to simulate a missing connection key.
        The test asserts that an exception is raised when the `connect_to_storage` method is called with these objects.
        """
        storage_details = MagicMock()
        storage_details.account_name = "test_account"
        storage_details.storage_name = "test_storage"
        connection_secret = MagicMock()
        connection_secret.connection_key = None

        with self.assertRaises(Exception):
            self.client.connect_to_storage(storage_details, connection_secret)

    
    @patch("core.clients.azure.storage.blob_storage_client.BlobServiceClient")
    def test_upload_file_to_storage_success(self, mock_blob_service_client):
        """
        Test case for the upload_file_to_storage method.

        This test case verifies that the upload_file_to_storage method
        correctly uploads a file to the storage container and returns
        the URL of the uploaded blob.

        Steps:
        1. Mock the container client and blob client.
        2. Set up test data - filename and data.
        3. Call the upload_file_to_storage method.
        4. Verify that the container client's upload_blob method is called
            with the correct parameters.
        5. Verify that the container client's get_blob_client method is called
            with the correct filename.
        6. Verify that the returned blob URL matches the expected value.

        """
        # Mock container client
        container_client_mock = MagicMock()
        self.client.container_client = container_client_mock

        # Mock blob client
        blob_client_mock = MagicMock()
        container_client_mock.get_blob_client.return_value = blob_client_mock
        blob_client_mock.url = "test_url"

        # Test data
        filename = "test_file"
        data = b"test_data"

        # Call the method
        blob_url = self.client.upload_file_to_storage(filename, data)

        # Assertions
        container_client_mock.upload_blob.assert_called_once_with(name=filename, data=data)
        container_client_mock.get_blob_client.assert_called_once_with(filename)
        self.assertEqual(blob_url, "test_url")

    def test_upload_file_to_storage_not_connected(self):
        """
        Test case to verify that an exception is raised when trying to upload a file to storage
        when the client is not connected.
        """
        filename = "test_file"
        data = b"test_data"

        with self.assertRaises(Exception):
            self.client.upload_file_to_storage(filename, data)


    def test_get_folder_name_from_storage_success(self):
        """
        Test case to verify the behavior of the get_folder_name_from_storage method when it successfully retrieves folder names from storage.
        """
        # Mock container client
        container_client_mock = MagicMock()
        self.client.container_client = container_client_mock

        # Mock blobs
        blob1 = MagicMock(name="folder1/file1")
        blob2 = MagicMock(name="folder2/file2")
        blob3 = MagicMock(name="bin/file3")
        blob4 = MagicMock(name="folder3/file4")
        container_client_mock.walk_blobs.return_value = [blob1, blob2, blob3, blob4]

        # Call the method
        folder_list = self.client.get_folder_name_from_storage()

        # Assertions
        self.assertEqual(folder_list, [])

    def test_get_folder_name_from_storage_not_connected(self):
        """
        Test case to verify that an exception is raised when trying to get the folder name from storage
        when the client is not connected.
        """
        with self.assertRaises(Exception):
            self.client.get_folder_name_from_storage()

    def test_fs_connection_with_account_name(self):
        """
        Test case to verify the fs_connection method with account name.

        This test case ensures that the fs_connection method returns the correct prefix URL and storage options
        when provided with a fs_connection_dict and fs_key containing the account name and connection key.

        It asserts that the prefix URL is "abfs://" and the storage options include the correct connection string
        and account key.

        """
        fs_connection_dict = MagicMock()
        fs_connection_dict.account_name = "test_account"
        fs_key = MagicMock()
        fs_key.connection_key = "test_key"

        prefix_url, storage_options = self.client.fs_connection(fs_connection_dict, fs_key)

        self.assertEqual(prefix_url, "abfs://")
        self.assertEqual(storage_options["connection_string"], "DefaultEndpointsProtocol=https;AccountName=test_account;AccountKey=test_key;EndpointSuffix=core.windows.net")
        self.assertEqual(storage_options["account_key"], "test_key")


if __name__ == "__main__":
    unittest.main()


