import logging
from datetime import datetime, timedelta

from azure.storage.blob import (BlobSasPermissions, BlobServiceClient,
                                generate_blob_sas)

from core.storage.storage_client import StorageClient
from core.utils.read_config import cloud_secrets


class AzureBlobStorageClient(StorageClient):
    """
    A class for interacting with Azure Blob Storage.

    This class inherits from the `StorageClient` base class and provides methods
    for connecting to Azure Blob Storage and performing operations such as uploading
    and downloading files.
    """

    def __init__(self):
        """
        Initialize the AzureBlobStorageClient instance.
        """
        pass

    def connect_to_storage(self, storage_details, connection_secret):
        """
        Connects to Azure Blob Storage using the provided storage details and connection keys.

        Parameters:
            storage_details (obj): A object containing <account_name>, <storage_name>
                account_name (str): The name of the Azure Storage account.
                storage_name (str): The name of the storage container.
            connection_secret (obj): The access key or connection string for the Azure Storage account.

        Raises:
            Exception: If the connection to Azure Blob Storage fails.
        """
        # Parse the storage details from config
        # Check if account_name exists and has a defined value
        if (
            hasattr(storage_details, "account_name")
            and storage_details.account_name is not None
        ):
            self.account_name = storage_details.account_name
        else:
            raise Exception("account_name is not defined or has no value")
        # Check if storage_name exists and has a defined value
        if (
            hasattr(storage_details, "storage_name")
            and storage_details.storage_name is not None
        ):
            self.storage_name = storage_details.storage_name
        else:
            raise Exception("storage_name is not defined or has no value")
        # Check if connection_keys exists and has a defined value
        if (
            hasattr(connection_secret, "connection_key")
            and connection_secret.connection_key is not None
        ):
            self.account_key = cloud_secrets.domain_storage.connection_key
        else:
            raise Exception("connection_key is not defined or has no value")
        try:
            # Initialize Azure Blob connection
            self.blob_service_client = BlobServiceClient(
                account_url=f"https://{self.account_name}.blob.core.windows.net/",
                credential=self.account_key,
            )
            self.container_client = self.blob_service_client.get_container_client(
                self.storage_name
            )
            return self.container_client
        except Exception as e:
            raise Exception("Azure Blob Storage Connection Failed")

    def upload_file_to_storage(self, filename, data):
        """
        Uploads a file to the storage container.

        This method uploads a file to the specified storage container in the Azure Blob Storage service.
        If the Azure Blob Storage Client is not connected or if an error occurs during the upload process,
        an exception is raised.

        Parameters:
            filename (str): The name of the blob in the storage container.
            data (bytes): The data to be uploaded.

        Returns:
            str: The URL of the uploaded file.

        Raises:
            Exception: If the Azure Blob Storage Client is not connected or if an error occurs during
                    the upload process.
        """
        # Check if the container client attribute exists in the class instance
        if hasattr(self, "container_client"):
            try:
                # Upload the blob to the storage container
                self.container_client.upload_blob(name=filename, data=data)
                # Get the blob client for the uploaded blob
                blob_client = self.container_client.get_blob_client(filename)
                # Get the URL of the uploaded blob
                blob_url = blob_client.url
                # Log a success message with the blob URL
                logging.info("File uploaded successfully : {}".format(blob_url))
            except Exception as e:
                try:
                    # Attempt to get the blob client using the blob service client directly
                    blob_client = self.blob_service_client.get_blob_client(
                        container=self.storage_name, blob=filename
                    )
                    # Get the URL of the blob
                    blob_url = blob_client.url
                except Exception as e:
                    # If an error occurs, raise an exception with the error message
                    raise Exception(
                        "An error occurred in uploading file to storage: ", str(e)
                    )
            # Return the URL of the uploaded blob
            return blob_url
        else:
            # If the container client is not connected, raise an exception
            raise Exception(
                "Azure Blob Storage Client is not connected. Call connect_to_storage() first."
            )

    def get_folder_name_from_storage(self):
        """
        Retrieves folder names from the storage container.

        Returns:
            list: A list of folder names found in the storage container.
                Returns None if an error occurs during the retrieval process.
        """
        # Check if the container client attribute exists in the class instance
        if hasattr(self, "container_client"):
            try:
                # Initialize an empty list to store folder names
                folder_list = list()
                # Iterate through all blobs in the container using walk_blobs()
                for blob in self.container_client.walk_blobs():
                    # Get the name of the blob
                    blob_name = blob.name
                    # Extract the first part of the blob name before '/'
                    parts = blob_name.split("/")[0]
                    # Skip processing if the part is 'bin'
                    if parts == "bin":
                        continue
                    # Replace underscores with spaces and capitalize the first letter of each word
                    parts = parts.replace("_", " ").title()
                    # Append the processed part to the folder_list if it's not empty
                    if len(parts) > 1:
                        folder_list.append(parts)
                # Return the list of folder names
                return folder_list
            except Exception as e:
                # Log the traceback and the error message if an exception occurs
                logging.error(traceback.format_exc())
                logging.error(e)
                # Return None to indicate an error occurred
                return None
        else:
            # Raise an exception if the container client is not connected
            raise Exception(
                "Azure Blob Storage Client is not connected. Call connect_to_storage() first"
            )

    def fs_connection(self, fs_connection_dict, fs_key):
        """
        Establishes a connection to Azure Blob Storage using the provided connection details.

        Parameters:
            fs_connection_dict (dict): A dictionary containing connection details.
            fs_key (str or None): The access key for the Azure Storage account.

        Returns:
            tuple: A tuple containing the prefix URL and storage options.

        Raises:
            ValueError: If the fs_key is None or not specified in fs_connection_dict.
        """
        if fs_connection_dict.account_name is None:
            prefix_url = ""
            default_endpoints_protocol = ""
            endpoint_suffix = None
            fs_key.connection_key = None
            storage_options = None
        else:
            # Define the prefix URL for Azure Blob Storage
            prefix_url = "abfs://"
            default_endpoints_protocol = "https"
            endpoint_suffix = "core.windows.net"

            # Build the connection string for Azure Blob Storage
            fs_connection_string = f"DefaultEndpointsProtocol={default_endpoints_protocol};AccountName={fs_connection_dict.account_name};AccountKey={fs_key.connection_key};EndpointSuffix={endpoint_suffix}"

            # Define storage options including connection string and account key
            storage_options = {
                "connection_string": fs_connection_string,
                "account_key": fs_key.connection_key,
            }

        # Return the prefix URL and storage options as a tuple
        return prefix_url, storage_options
