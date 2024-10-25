import logging

import numpy as np
import openai

from core.retrievers.retrievers_client import Retrievers

MYLOGGERNAME = "QueryInsights"


class RetrieversFactory:

    """
    A factory class for creating Retrievers.

    Args:
        retrievers_type (Retrievers): The type of retrievers to create.

    Attributes:
        retrievers_type (Retrievers): The type of retrievers.

    Methods:
        _convert_to_docs: Function to create one list having documents.
        create_retriever_object: Function to create one retriever object using Retriever from langchain.
        get_relevent_documents: Function to get relevent documents object using Retriever object.
        retrieve_column_ids: Function to get one list having column indexes using top k relevent documents.

    """

    def __init__(self, retrievers_type: Retrievers) -> None:
        self.retrievers_type = retrievers_type

    def _convert_to_docs(self, column_selection_method):
        """
        Function to create one list having documents.
        Each document will have column name, column description and table name concatenated as page content and metadata will have unique_id, column_name, column_description, table_name and id_flag.

        Parameters
        ----------
        column_selection_method : str
            column selection parameter e.g, column_name, column_description, table_name or combined
        Returns
        -------
        List
            List of documents having column name, column description and table name concatenated as page content
        """

        return self.retrievers_type._convert_to_docs(column_selection_method)

    def create_retriever_object(self, documents):
        """
        Function to create one retriever object using Retriever from langchain.

        Parameters
        ----------
        docs : List
            List containing documents

        Returns
        -------
        Retriever
            Retriever
        """
        return self.retrievers_type.create_retriever_object(documents)

    def get_relevent_documents(self, retriever, question):
        """
        Function to get relevent documents object using Retriever object.
        Parameters
        ----------
        retriever_object: object
            object of Retriever
        question: str

        Returns
        -------
        List : top k relevent documents


        """
        return self.retrievers_type.get_relevent_documents(retriever, question)

    def retrieve_column_ids(self, top_k_relevent_documents):
        """
        Function to get one list having column indexes using top k relevent documents.
        These indexes can be used to get column related information from the dataframe having unique index
        for unique colum name and other information related to that column.

        Parameters
        ----------
        top_k_relevent_documents: List
            List containing documents

        Returns
        -------
        List : top k indexes


        """
        return self.retrievers_type.retrieve_column_ids(top_k_relevent_documents)