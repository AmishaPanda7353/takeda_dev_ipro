import logging
from abc import ABC, abstractmethod

import numpy as np

MYLOGGERNAME = "QueryInsights"


class VectorDB(ABC):

    """
    Abstract base class for VectorDB.

    Args:
        config (dict): It contains the parameters related to VectorDB.
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
        """
        Initializes a new instance of the VectorDB class.

        Parameters:
            config (dict): The configuration object containing vectordb information.
            data_frame (pd.Dataframe): Dataframe having unique index for unique colum name and other information related to that column,
                                e.g. column description, table name etc.


        Returns:
            None
        """

        self.logger = logging.getLogger(MYLOGGERNAME)
        self.config = config
        self.data_frame = data_frame

    @abstractmethod
    def connect(self):
        """
        Abstract method to connect with vectordb.
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def create_collections(self):
        """
        Abstract method to create a schema with unique id for each type of column_selection_method,
        for e.g. 'column_name', 'column_description' etc.
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def _create_index(self, embedding_field_name: str):
        """
        Abstract method to create indexes for each type of column selection method,
        for e.g. 'column_name', 'column_description' etc.
        Parameters
        ----------
        embedding_field_name : str
            column selection method
        Returns
        -------
        None

        """
        pass

    @abstractmethod
    def _generate_embeddings(self, text: str) -> np.array:
        """
        Abstract method to generate embeddings for given text
        Parameters
        ----------
        text : str

        Returnss
        -------
        np.ndarray
            numpy array containing embeddings for given text

        """
        pass

    @abstractmethod
    def insert_data(self):
        """
        Abstract method to insert data from Dataframe(Dataframe having information about each column
        from data dictionary) into collections in vector db.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        pass

    @abstractmethod
    def search(self, query_text: str):
        """
        Abstract method to get one list having relevant column indexes to the question.
        These indexes can be used to get information from relevant columns.

        Parameters
        ----------
        query_text : str

        Returns
        -------
        List
           List having relevant column indexes to the question

        """
        pass