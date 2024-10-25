import unittest
from unittest.mock import MagicMock, patch

from core.clients.azure.model.openai_client import GPTModelCall
from core.utils.client_utils import get_model_type, get_database_client, get_storage_client

from core.clients.azure.database.postgres_client import AzurePostgresClient
from core.clients.azure.database.sqlite_client import SqliteClient
from core.clients.azure.storage.blob_storage_client import AzureBlobStorageClient

from easydict import EasyDict


class TestGetModelType(unittest.TestCase):
    def setUp(self):
        """
        Set up the test environment before each test case.

        This method initializes the necessary variables and configurations for the test case.

        Args:
            None

        Returns:
            None
        """
        self.config = EasyDict({
            "llm_model_type": "openai"
        })
        self.prompt_dict = {
            "static_prompt": "This is a static prompt",
            "additional_context": "<additional_context>",
            "guidelines": "These are guidelines\n<sql>",
            "business_overview": "This is a business overview",
            "system_role": "This is the system role"
        }
        self.question = "What is the answer?"
        self.additional_context = ""
        self.connection_param_dict = {
            "api_type": "azure",
            "api_base": "https://api.example.com",
            "api_version": "v1"
        }
        self.user_config = EasyDict({
            "today": "2022-01-01"
        })
        
        self.language = "english"
        self.db_param_dict = {
            "db_name": "MySql"
        }
        self.dictionary = {
            "key": "value"
        }
        self.business_overview = "Business overview"
        self.suggestion = "Some suggestion"
        self.table = "Some table"
        self.history = "Some history"
        self.error_message = "Some error message"
        self.sample_input = ["Sample question", "Sample response"]
        self.data_dictionary = {}
        self.db_params = {}
        self.code_result = "Code result"
        self.table = "Table"

    def test_get_model_type_openai(self):
        """
        Test case for the get_model_type method with OpenAI model.

        This test verifies that the get_model_type method returns an instance of GPTModelCall
        when the OpenAI model is used.

        """
        model_client = get_model_type(
            self.config,
            self.prompt_dict,
            self.question,
            self.additional_context,
            self.connection_param_dict,
            self.user_config,
            self.language,
            self.data_dictionary,
            self.business_overview,
            self.db_param_dict,
            self.sample_input,
            self.code_result,
            self.table
        )
        self.assertIsInstance(model_client, GPTModelCall)

    def test_get_model_type_unsupported_model(self):
        """
        Test case to verify the behavior of get_model_type when an unsupported model type is provided.
        It should raise a ValueError.
        """
        self.config["llm_model_type"] = "invalid_model"
        with self.assertRaises(ValueError):
            get_model_type(
                self.config,
                self.prompt_dict,
                self.question,
                self.additional_context,
                self.connection_param_dict,
                self.user_config,
                self.language,
                self.data_dictionary,
                self.business_overview,
                self.db_params,
                self.sample_input,
                self.code_result,
                self.table
            )

    def test_get_model_type_invalid_model_client(self):
        """
        Test case to verify the behavior of get_model_type when an invalid model client is used.
        It should raise a ValueError.
        """
        with patch("core.utils.client_utils.issubclass") as mock_issubclass:
            mock_issubclass.return_value = False
            with self.assertRaises(ValueError):
                get_model_type(
                    self.config,
                    self.prompt_dict,
                    self.question,
                    self.additional_context,
                    self.connection_param_dict,
                    self.user_config,
                    self.language,
                    self.data_dictionary,
                    self.business_overview,
                    self.db_params,
                    self.sample_input,
                    self.code_result,
                    self.table
                )
    
    def test_get_database_client_postgres(self):
        """
        Test case for the get_database_client function with 'postgres' database type.
        """
        database_type = "postgres"
        database_client = get_database_client(database_type)
        
        mock_obj = MagicMock()
        instance_val = AzurePostgresClient()
        
        mock_obj.instance = instance_val
        
        self.assertEqual(database_client.__class__, mock_obj.instance.__class__)
    
    def test_get_database_client_SqlClient(self):
        """
        Test case for the get_database_client function with 'sqlite' database type.
        It checks if the returned database client is an instance of SqliteClient.
        """
        database_type = "sqlite"
        database_client = get_database_client(database_type)
        
        mock_obj = MagicMock()
        instance_val = SqliteClient()
        
        mock_obj.instance = instance_val
        
        self.assertEqual(database_client.__class__, mock_obj.instance.__class__)

    def test_get_storage_client(self):
        """
        Test case for the get_storage_client method.
        """
        cloud_config = EasyDict({"cloud_provider" : "azure"})
        
        database_client = get_storage_client(cloud_config)
        
        mock_obj = MagicMock()
        
        instance_val = AzureBlobStorageClient()
        
        mock_obj.instance = instance_val
         
        self.assertEqual(database_client.__class__, mock_obj.instance.__class__)


if __name__ == "__main__":
    unittest.main()

