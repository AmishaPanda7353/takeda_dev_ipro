import unittest
from unittest.mock import patch, MagicMock
import openai
from core.clients.azure.model.openai_client import GPTModelCall
from easydict import EasyDict
from dynaconf import Dynaconf,settings

class TestOpenAIClient(unittest.TestCase):

    def setUp(self):
        # self.client = GPTModelCall()
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
            "db_name": None
        }
        self.dictionary = {
            "key": "value"
        }
        self.business_overview = None
        self.suggestion = "Some suggestion"
        self.table = "Some table"
        self.history = "Some history"
        self.error_message = "Some error message"
        self.sample_input = ["Sample question", "Sample response"]
        
        self.gpt_model_call = GPTModelCall(
            self.prompt_dict,
            self.question,
            self.additional_context,
            self.connection_param_dict,
            self.user_config,
            self.language,
            self.db_param_dict,
            self.dictionary,
            self.business_overview,
            self.suggestion,
            self.table,
            self.history,
            self.error_message,
            self.sample_input
        )
        self.api_key = "API_KEY"
        
        # self.client = self.gpt_model_call.set_connection_params()
        
    def test_set_connection_params(self):
        """
        Test case for the set_connection_params method of the GPTModelCall class.

        This test verifies that the set_connection_params method correctly sets the connection parameters
        for the AzureOpenAI client.

        It uses the patch decorator from the unittest.mock module to mock the AzureOpenAI class and assert
        that it is called with the expected parameters.

        """
        with patch("core.clients.azure.model.openai_client.AzureOpenAI") as mock_azure_openai:
            self.client = self.gpt_model_call.set_connection_params()
            mock_azure_openai.assert_called_with(
                azure_endpoint=self.connection_param_dict["api_base"],
                api_key=self.gpt_model_call.api_key,
                api_version=self.connection_param_dict["api_version"]
            )
            
    def test_generate_prompt(self):
        """
        Test case for the generate_prompt method of the GPTModelCall class.

        This test verifies that the generate_prompt method sets the correct prompt value
        based on the mocked return value of the parse_today_date function.

        It checks if the prompt is set to the expected value after calling the generate_prompt method.

        """
        with patch("core.clients.azure.model.openai_client.parse_today_date") as mock_parse_today_date:
            mock_parse_today_date.return_value = "2022-01-01"
            self.gpt_model_call.generate_prompt()
            expected_prompt = "This is a static prompt\n\nThese are guidelines\nMySql"
            self.assertEqual(self.gpt_model_call.prompt, expected_prompt)

    @patch("core.clients.azure.model.openai_client.AzureOpenAI")
    def test_model_response_chatcompletion(self, mock_client):
        
        # with patch("core.clients.azure.model.openai_client.AzureOpenAI") as mock_azure_openai:
        # self.client = self.gpt_model_call.set_connection_params()
        #     mock_azure_openai.assert_called_with(
        #         azure_endpoint=self.connection_param_dict["api_base"],
        #         api_key=self.gpt_model_call.api_key,
        #         api_version=self.connection_param_dict["api_version"]
        #     )
        
        # with patch("core.clients.azure.model.openai_client.parse_today_date") as mock_parse_today_date:
        #     mock_parse_today_date.return_value = "2022-01-01"
        #     self.gpt_model_call.generate_prompt()
        #     expected_prompt = "This is a static prompt\n\nThese are guidelines\nMySql"
        #     self.assertEqual(self.gpt_model_call.prompt, expected_prompt)

        
        # with patch("core.clients.azure.model.openai_client.AzureOpenAI") as mock_client:
        mock_client = mock_client.return_value
        model_param_dict = {
            "function": "chatcompletion",
            "engine": "davinci",
            "temperature": 0.8,
            "max_tokens": 100,
            "n": 1,
            "stop": None,
            "history": None
        }
        debug_prompt = "debug prompt"
        history = "previous history"

        # Mock the OpenAI API response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Generated response"
        mock_response.choices[0].finish_reason = "completed"
        mock_response.usage = 50
        print(mock_client,"--------------------------------------")
        mock_client.chat.completions.create.return_value = mock_response
        

        output, finish_reason, tokens, error_message = self.gpt_model_call.model_response(model_param_dict, debug_prompt, history)

        self.assertEqual(output, "Generated response")
        self.assertEqual(finish_reason, "completed")
        self.assertEqual(tokens, 50)
        self.assertIsNone(error_message)

    @patch('openai.Client')
    def test_model_response_completion(self, mock_client):
        model_param_dict = {
            "function": "completion",
            "engine": "davinci",
            "temperature": 0.8,
            "max_tokens": 100,
            "n": 1,
            "stop": None
        }
        debug_prompt = "debug prompt"
        history = "previous history"

        # Mock the OpenAI API response
        mock_response = MagicMock()
        mock_response.choices[0].text = "Generated response"
        mock_response.choices[0].finish_reason = "completed"
        mock_response.usage = 50
        mock_client.completions.create.return_value = mock_response

        output, finish_reason, tokens, error_message = self.gpt_model_call.model_response(model_param_dict, debug_prompt, history)

        self.assertEqual(output, "Generated response")
        self.assertEqual(finish_reason, "completed")
        self.assertEqual(tokens, 50)
        self.assertIsNone(error_message)

    def test_model_response_invalid_function(self):
        model_param_dict = {
            "function": "invalid",
            "engine": "davinci",
            "temperature": 0.8,
            "max_tokens": 100,
            "n": 1,
            "stop": None
        }
        debug_prompt = "debug prompt"
        history = "previous history"

        with self.assertRaises(ValueError):
            self.gpt_model_call.model_response(model_param_dict, debug_prompt, history)
            
#     def test_model_response():
#         # Create a dummy config using dynaconf
#         settings.configure(
#             ENVVAR_PREFIX_FOR_DYNACONF="DUMMY",
#             DUMMY_ENGINE="gpt-3.5-turbo",
#             DUMMY_TEMPERATURE=0.8,
#             DUMMY_MAX_TOKENS=100,
#             DUMMY_N=1,
#             DUMMY_STOP=["stop"],
#             DUMMY_API_TYPE="azure",
#         )

#         # Create an instance of GPTModelCall
#         model_call = GPTModelCall()

#         # Define the model parameters
#         model_param_dict = {
#             "function": "chatcompletion",
#             "engine": settings.DUMMY_ENGINE,
#             "temperature": settings.DUMMY_TEMPERATURE,
#             "max_tokens": settings.DUMMY_MAX_TOKENS,
#             "n": settings.DUMMY_N,
#             "stop": settings.DUMMY_STOP,
#         }

#         # Define the debug prompt and history
#         debug_prompt = "Debug prompt"
#         history = "Previous response by GPT"

#         # Call the model_response method
#         output, finish_reason, tokens, error_message = model_call.model_response(
#             model_param_dict, debug_prompt, history
#         )

#         # Assert the output, finish_reason, tokens, and error_message
#         assert output == "Dummy output"
#         assert finish_reason == "completed"
#         assert tokens == 50
#         assert error_message is None

#         # Define the model parameters for completion function
#         model_param_dict = {
#             "function": "completion",
#             "engine": settings.DUMMY_ENGINE,
#             "temperature": settings.DUMMY_TEMPERATURE,
#             "max_tokens": settings.DUMMY_MAX_TOKENS,
#             "n": settings.DUMMY_N,
#             "stop": settings.DUMMY_STOP,
#         }

#         # Call the model_response method for completion function
#         output, finish_reason, tokens, error_message = model_call.model_response(
#             model_param_dict, debug_prompt, history
#         )

#         # Assert the output, finish_reason, tokens, and error_message for completion function
#         assert output == "Dummy output"
#         assert finish_reason == "completed"
#         assert tokens == 50
#         assert error_message is None

#         # Define the model parameters with invalid function
#         model_param_dict = {
#             "function": "invalid",
#             "engine": settings.DUMMY_ENGINE,
#             "temperature": settings.DUMMY_TEMPERATURE,
#             "max_tokens": settings.DUMMY_MAX_TOKENS,
#             "n": settings.DUMMY_N,
#             "stop": settings.DUMMY_STOP,
#         }

#         # Call the model_response method with invalid function
#         try:
#             model_call.model_response(model_param_dict, debug_prompt, history)
#         except ValueError as e:
#             assert str(e) == "Invalid function invalid is passed in the config. Acceptable values are 'Completion' and 'ChatCompletion'"

# if __name__ == '__main__':
#     unittest.main()