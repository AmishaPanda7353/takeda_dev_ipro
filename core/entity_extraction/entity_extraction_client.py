import json
from abc import ABC, abstractmethod


class EntityExtractionClient(ABC):
    def __init__(self, config, all_data_dictionaries) -> None:
        """
        Abstract base class for EntityExtractionClient.

        Args:
            config (dict): It contains the parameters related to EntityExtractionClient.
            question (str): The main question or query.
            data_dictionary (dict): Dictionary having details for each table.

        Attributes:
            logger (Logger): The logger object for logging.

        """

        self.config = config
        self.all_data_dictionaries = all_data_dictionaries

    @abstractmethod
    def get_entities(self, text):
        """
        Abstract method to find entities.
        Parameters
        ----------
        text : str

        Returns
        -------
        List
            List of entities
        """

        pass
