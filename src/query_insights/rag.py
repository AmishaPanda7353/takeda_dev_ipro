import logging
import sys

import pandas as pd
import spacy

from core.utils.client_utils import get_retriever_client, get_vectordb_client # type: ignore
from src.query_insights.utils.utils import DataConverter, get_word_chunks # type: ignore

MYLOGGERNAME = "QueryInsights"
en_core_web_model = "en_core_web_lg"


class RAG:
    """A class for getting union of results generated from using different methods of retrieval methods

    Parameters
    ----------
    data_dictionary: str
        data dictionary
    question : str
        Question that is asked by user

    Raises
    ------
    ValueError
        if any of the argument is missing or invalid.
    """

    def __init__(self, data_dictionary, question) -> None:
        self.question = question
        self.data_dictionary = data_dictionary
        self.df = DataConverter.json_to_dataframe(self.data_dictionary)
        self.logger = logging.getLogger(MYLOGGERNAME)
        # self.nlp = spacy.load(en_core_web_model)
        # self.stop_words = spacy.lang.en.STOP_WORDS
        # doc = self.nlp(self.question)
        # self.question_chunks = get_word_chunks(doc, self.question)

    def embedding_retrievals(self, vectordb_config) -> list:
        """
        Function to get one list having column indexes using embedding method for table selection.
        These indexes can be used to get column related information from the dataframe having unique index
        for unique colum name and other information related to that column.

        Parameters
        ----------
        None

        Returns
        -------
        List
            List having column indexes from embedding method for table selection
        """
        try:
            vector_db = get_vectordb_client(vectordb_config, self.df)
            vector_db.insert_data()
            all_top_ids_embeddings = []
            top_ids_embeddings = vector_db.search(self.question)
            all_top_ids_embeddings.extend(top_ids_embeddings)
            self.logger.info("retrieved column indexes using embeddings")
            return all_top_ids_embeddings
        except Exception as e:
            self.logger.error(f"embedding based retrieval not completed. Error :\n {e}")

    def keyword_matching_retrievals(self, retriever_config) -> list:
        """
        Function to get one list having column indexes using keyword retrieval method for table selection.
        These indexes can be used to get column related information from the dataframe having unique index
        for unique colum name and other information related to that column.

        Parameters
        ----------
        None

        Returns
        -------
        List
            List having column indexes from keyword retrieval method for table selection
        """
        try:
            retriever = get_retriever_client(retriever_config, self.df)

            for column_selection_method in retriever_config.column_selection_method:
                documents = retriever._convert_to_docs(column_selection_method)
                retriever_object = retriever.create_retriever_object(documents)
                all_top_ids_documents = []
                for ques_chunk in self.question_chunks:
                    relevent_documents = retriever.get_relevent_documents(
                        retriever_object, ques_chunk
                    )
                    top_ids_documents = retriever.retrieve_column_ids(
                        relevent_documents
                    )
                    all_top_ids_documents.extend(top_ids_documents)

            self.logger.info("retrieved relevant column indexes")
            return all_top_ids_documents

        except Exception as e:
            self.logger.error(
                f"Error while getting column indexes using keyword matching. Error :\n {e}"
            )

    def get_data_dict(self, all_columns_ids):
        """

        Function to get data dictionary after completing table selection

        Parameters
        ----------
        None

        Returns
        -------
        dict
            data dictionary having only filtered tables and columns after table selection

        """
        try:
            data_dict_union = DataConverter.dataframe_to_json(self.df, all_columns_ids)
            self.logger.info("final data dictionary after table selection created")
            return data_dict_union
        except Exception as e:
            self.logger.error(
                f"Error while creating final data dictionary after table selection. Error :\n {e}"
            )