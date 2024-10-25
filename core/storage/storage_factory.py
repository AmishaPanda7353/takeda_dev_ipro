from core.storage.storage_client import StorageClient


class StorageFactory:
    """
    A factory class for managing storage operations.

    Parameters:
        StorageClient: An instance of the storage client class.

    Attributes:
        storage_client (StorageClient): The storage client instance used for performing storage operations.

    """

    def __init__(self, StorageClient) -> None:
        """
        Initializes a new StorageFactory instance.

        Parameters:
            StorageClient: An instance of the storage client class.
        """
        self.storage_client = StorageClient

    def connect_to_storage(self, storage_details, connection_keys):
        """
        Connects to the storage service using the provided credentials.

        This method establishes a connection to the storage service using the specified account name,
        account key, and storage name.

        Parameters:
            storage_details (str): A string containing information specific to the storage service,
                                such as the account name, storage name, etc.
            connection_keys (str): The access keys or credentials required to authenticate with
                                the storage service.
        """
        return self.storage_client.connect_to_storage(storage_details, connection_keys)

    def upload_file_to_storage(self, filename, data):
        """
        Uploads a file to the storage container.

        This method uploads a file to the specified storage container using the provided credentials.

        Parameters:
            filename (str): The name of the file to be uploaded.
            data (bytes): The data to be uploaded.

        Returns:
            str: The URL of the uploaded file.

        """
        return self.storage_client.upload_file_to_storage(filename, data)

    def get_folder_name_from_storage(self):
        """
        Retrieves folder names from the storage container.

        This method retrieves folder names from the connected storage container.

        Returns:
            list: A list of folder names.

        """
        return self.storage_client.get_folder_name_from_storage()

    def fs_connection(self, fs_connection_dict, fs_key):
        return self.storage_client.fs_connection(fs_connection_dict, fs_key)
