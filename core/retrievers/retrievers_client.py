import logging
from abc import ABC, abstractmethod

import numpy as np

MYLOGGERNAME = "QueryInsights"


class Retrievers(ABC):

    """
    Abstract base class for Retrievers.

    Args:
        config (dict): It contains the parameters related to Retrievers.
        question (str): The main question or query.
        data_dictionary (dict): Dictionary having details for each table.

    Attributes:
        logger (Logger): The logger object for logging.

    """

    def __init__(
        self,
        config,
        data_frame,
    ) -> None:
        self.logger = logging.getLogger(MYLOGGERNAME)
        self.config = config
        self.data_frame = data_frame

    @abstractmethod
    def _convert_to_docs(self, column_selection_method):
        """
        Abstract method to create one list having documents.
        Parameters
        ----------
        column_selection_method : str
            column selection method

        Returns
        -------
        List
            List of documents having column name, column description and table name concatenated as page content
        """

        pass

    @abstractmethod
    def create_retriever_object(self, documents):
        """
        Abstract method to create one retriever object using Retriever from langchain.
        Parameters
        ----------
        documents:List
            List containing documents
        Returns
        -------
        Retriever
            Retriever

        """
        pass

    @abstractmethod
    def get_relevent_documents(self, retriever_object, question):
        """
        Abstarct method  to get relevent documents object using Retriever object.
        Parameters
        ----------
        retriever_object: object
            object of Retriever
        question: str

        Returns
        -------
        List : top k relevent documents
        """
        pass

    @abstractmethod
    def retrieve_column_ids(self, top_k_relevent_documents):
        """
        Abstract method to get one list having column indexes using top k relevent documents.
        Parameters
        ----------
        top_k_relevent_documents: List
            List containing documents

        question: str

        Returns
        -------
        List : top k relevent documents
        """
        pass