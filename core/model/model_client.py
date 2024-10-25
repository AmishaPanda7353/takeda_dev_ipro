import logging
import os
from abc import ABC, abstractmethod

import openai
from openai import AzureOpenAI
from core.utils.read_config import config

MYLOGGERNAME = "QueryInsights"


class Model(ABC):
    """
    Abstract base class for models.

    Args:
        prompt_dict (dict): A dictionary containing prompts for different scenarios.
        question (str): The main question or query.
        additional_context (str): Additional context or information.
        connection_param_dict (dict): A dictionary containing connection parameters.
        user_config (dict): User configuration settings.
        language (str): The language used for the model.
        db_param_dict (dict): A dictionary containing database parameters.
        dictionary (str): The dictionary used for the model.
        business_overview (str): Business overview information.
        suggestion (str): Suggestion or recommendation.
        table (str): Table information.
        history (str): History information.
        error_message (str): Error message.
        sample_input (tuple): A tuple containing a sample question and response.

    Attributes:
        logger (Logger): The logger object for logging.
        user_config (dict): User configuration settings.
        api_key (str): The API key for accessing the OpenAI API.
        dictionary (str): The dictionary used for the model.
        prompt_dict (dict): A dictionary containing prompts for different scenarios.
        question (str): The main question or query.
        additional_context (str): Additional context or information.
        business_overview (str): Business overview information.
        suggestion (str): Suggestion or recommendation.
        table (str): Table information.
        history (str): History information.
        error_message (str): Error message.
        connection_param_dict (dict): A dictionary containing connection parameters.
        sample_question (str): A sample question for testing.
        sample_response (str): The expected response for the sample question.
        db_param_dict (dict): A dictionary containing database parameters.
        language (str): The language used for the model.

    """

    def __init__(
        self,
        prompt_dict,
        question,
        additional_context,
        connection_param_dict,
        track,
        user_config,
        language,
        db_param_dict,
        dictionary,
        business_overview,
        suggestion,
        table,
        history,
        error_message,
        sample_input,
    ):
        self.logger = logging.getLogger(MYLOGGERNAME)
        self.user_config = user_config
        self.logger.info(f"Configuring LLM model for track : {track} with {self.user_config.connection_params.api_type[track]}")

        if self.user_config.connection_params.api_type[track] == 'openai':
            self.api_key = os.environ.get("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("API key not found in environment variables")
            AzureOpenAI.api_key = self.api_key

        # If language is None or empty string, default to "english" language
        if language is None or not bool(str(language).strip()):
            language = "english"
        language = language.lower().title()

        self.dictionary = dictionary
        self.prompt_dict = prompt_dict
        self.question = question
        self.additional_context = additional_context
        self.business_overview = business_overview
        self.suggestion = suggestion
        self.table = table
        self.history = history
        self.error_message = error_message
        self.connection_param_dict = connection_param_dict

        self.sample_question = None
        self.sample_response = None
        if sample_input is not None:
            self.sample_question = sample_input[0]
            self.sample_response = sample_input[1]
        self.db_param_dict = db_param_dict
        self.language = language
        self.track = track

    @abstractmethod
    def set_connection_params(self):
        """
        Abstract method to set the connection parameters for the model client.
        This method should be implemented by the child classes.
        """
        pass

    @abstractmethod
    def generate_prompt(self):
        """
        Generates a prompt for the model.

        This method should be implemented by subclasses to generate a prompt
        that will be used as input for the model.

        Returns:
            str: The generated prompt.
        """
        pass

    @abstractmethod
    def model_response(
        self, model_param_dict: dict, debug_prompt: str = None, history: str = None
    ):
        """
        Abstract method for handling model response.

        Args:
            model_param_dict (dict): A dictionary containing model parameters.
            debug_prompt (str, optional): A string representing a debug prompt. Defaults to None.
            history (str, optional): A string representing the history of interactions. Defaults to None.
        """
        pass
