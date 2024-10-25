from abc import ABC, abstractmethod


class StorageClient(ABC):
    """
    Base class for storage client.

    This class defines the interface for interacting with different storage services.
    Subclasses must implement the abstract methods connect_to_storage(), upload_file_to_storage() and get_folder_name_from_storage.
    """

    def __init__(self):
        """
        Initializes a new instance of the StorageClient class.

        Parameters:
            None

        Returns:
            None
        """
        pass

    @abstractmethod
    def connect_to_storage(self, storage_details, connection_keys):
        """
        Abstract method to establish a connection to the storage service.

        This method must be implemented by subclasses to establish a connection to the
        respective storage service.

        Parameters:
            storage_details (str): A string containing information specific to the storage service,
                                such as the account name, storage name, etc.
            connection_keys (str): The access keys or credentials required to authenticate with
                                the storage service.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.

        Returns:
            None
        """
        raise NotImplementedError(
            "connect_to_storage method must be implemented by subclass"
        )

    @abstractmethod
    def upload_file_to_storage(self, filename, data):
        """
        Abstract method to upload a file to the storage service.

        This method must be implemented by subclasses to upload a file to the respective
        storage service.

        Parameters:
            filename (str): The name of the file in the storage service.
            data (bytes): The data to be uploaded.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.

        Returns:
            str: The URL of the uploaded file.
        """
        raise NotImplementedError(
            "upload_file_to_storage method must be implemented by subclass"
        )

    @abstractmethod
    def get_folder_name_from_storage(self):
        """
        Abstract method to retrieve the folder names from the storage service.

        This method must be implemented by subclasses to retrieve the folder names
        from the respective storage service.

        Returns:
            list: A list of folder names.
        """
        raise NotImplementedError(
            "get_folder_name_from_storage method must be implemented by subclass"
        )

    @abstractmethod
    def fs_connection(self, fs_connection_dict, fs_key):
        raise NotImplementedError(
            "fs_connection method must be implemented by subclass"
        )
