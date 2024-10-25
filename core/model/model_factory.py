import logging
import os
from abc import ABC, abstractmethod

import openai

from core.model.model_client import Model

MYLOGGERNAME = "QueryInsights"


class ModelFactory:
    """
    A factory class for creating models.

    Args:
        model_type (Model): The type of model to create.

    Attributes:
        model_type (Model): The type of model.

    Methods:
        set_connection_params: Sets the connection parameters for the model.
        generate_prompt: Generates a prompt for the model.
        model_response: Generates a response from the model.

    """

    def __init__(self, model_type: Model) -> None:
        self.model_type = model_type

    def set_connection_params(self):
        """
        Sets the connection parameters for the model.
        """
        self.model_type.set_connection_params()

    def generate_prompt(self):
        """
        Generates a prompt for the model.
        """
        self.model_type.generate_prompt()

    def model_response(
        self, model_param_dict: dict, debug_prompt: str = None, history: str = None
    ):
        """
        Generates a response from the model.

        Args:
            model_param_dict (dict): A dictionary of model parameters.
            debug_prompt (str, optional): A debug prompt. Defaults to None.
            history (str, optional): A history of previous interactions. Defaults to None.

        Returns:
            The model's response.

        """
        return self.model_type.model_response(model_param_dict)
