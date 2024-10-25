
import logging

import numpy as np
import openai

from core.vectordb.vectordb_client import VectorDB

MYLOGGERNAME = "QueryInsights"


class VectorDBFactory:

    """
    A factory class for creating VectorDB.

    Args:
        vectordb_type (VectorDB): The type of vectordb to create.

    Attributes:
        vectordb_type (VectorDB): The type of vectordb.

    Methods:
        connect: Function to connect with vectordb.
        create_collections: Function to create a schema with unique id for each type of column_selection_method
        _create_index: Function to create indexes for each type of column selection method
        _generate_embeddings: Function to generate embeddings for given text
        insert_data: Function to insert data from Dataframe into collections in Milvus db
        search: Function to get one list having relevant column indexes to the question.
    """

    def __init__(self, vectordb_type: VectorDB) -> None:
        self.vectordb_type = vectordb_type

    def connect(self):
        """
        Function to connect with vectordb.

        Parameters
        ----------
        None

        Returns
        -------
        None


        """
        self.vectordb_type.connect()

    def create_collections(self):
        """
        Function to create indexes for each type of column selection method.
        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.vectordb_type.create_collections()

    def _create_index(self, embedding_field_name: str):
        """
        Function to create indexes for each type of column selection method
        Parameters
        ----------
        embedding_field_name : str
            column selection method

        Returns
        -------
        None


        """
        self.vectordb_type._create_index()

    def _generate_embeddings(self, text: str) -> np.array:
        """
        Function to generate embeddings for given text
        Parameters
        ----------
        text : str
        Returns
        -------
        np.ndarray
            numpy array containing embeddings for given text


        """
        return self.vectordb_type._generate_embeddings(text)

    def insert_data(self):
        """
        Function to insert data from Dataframe(Dataframe having information about each column from data dictionary)
        into collections in vector db.
        Parameters
        ----------
        None

        Returns
        -------
        None


        """
        self.vectordb_type.insert_data()

    def search(self, query_text: str):
        """
        Function to get one list having relevant column indexes to the question.
        These indexes can be used to get information from relevant columns.

        Parameters
        ----------
        query_text : str

        Returns
        -------
        List
           List having relevant column indexes to the question

        """
        return self.vectordb_type.search(query_text)
