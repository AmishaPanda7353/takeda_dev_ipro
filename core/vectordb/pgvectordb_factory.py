
import logging

import numpy as np
import openai

from core.vectordb.pgvectordb_client import PGVectorDB

MYLOGGERNAME = "QueryInsights"


class PGVectorDBFactory:

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

    def __init__(self, vectordb_type: PGVectorDB) -> None:
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

    def adapt_vector(self):
        """
        Function to create indexes for each type of column selection method.
        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.vectordb_type.adapt_vector()

    def create_table(self):
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
       self.vectordb_type.create_table()

    def convert_to_embeddings(self, text: str) -> np.array:
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
        return self.vectordb_type.convert_to_embeddings(text)

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
    
    def create_index(self):
        
        self.vectordb_type.create_index()        

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

    def close_connection(self):

        self.vectordb_type.close_connection()