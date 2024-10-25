import logging

from langchain.retrievers import BM25Retriever
from langchain.schema import Document

from core.retrievers.retrievers_client import Retrievers

MYLOGGERNAME = "QueryInsights"


class BM25(Retrievers):

    """
    A class for interacting with BM25 retriever.

    This class inherits from the `Retrievers` base class and provides methods for connecting to BM25 retriever and
    performing converting to documents, creating retriever object and get relevant documents and retrieve column ids.

    Parameters
    ----------
    config (dict): The configuration object containing retrievers information.
    data_frame (pd.Dataframe): Dataframe having unique index for unique colum name and other information related to that column,
                                e.g. column description, table name etc.

    """

    def __init__(self, config, data_frame):
        super().__init__(config, data_frame)
        self.top_k = self.config.top_k
        self.logger = logging.getLogger(MYLOGGERNAME)

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

        try:
            docs = []
            for index, row in self.data_frame.iterrows():
                doc_ = Document(
                    page_content=row[column_selection_method].lower(),
                    metadata={
                        "unique_id": row["unique_id"],
                        "column_name": row["column_name"],
                        "column_description": row["column_description"],
                        "table_name": row["table_name"],
                        "id_flag": row["id"],
                    },
                )

                docs.append(doc_)
            self.logger.info(
                "created and saved documents using column name, description and table name successfully."
            )
            return docs
        except Exception as e:
            self.logger.error(
                f"Error while creating and saving documents using column name, description and table name. Error :\n {e}"
            )

    def create_retriever_object(self, documents):
        """
        Function to create one retriever object using BM25Retriever from langchain.

        Parameters
        ----------
        docs : List
            List containing documents

        Returns
        -------
        Retriever
            BM25 Retriever
        """
        try:
            retriever = BM25Retriever.from_documents(documents)
            retriever.k = self.top_k
            self.logger.info("BM25 retriever created successfully.")
            return retriever
        except Exception as e:
            self.logger.error(f"Error while creating BM25 retriever. Error :\n {e}")

    def get_relevent_documents(self, retriever_object, question):
        """
        Function to get relevent documents object using BM25Retriever object.
        Parameters
        ----------
        retriever_object: object
            object of BM25Retriever
        question: str

        Returns
        -------
        List : top k relevent documents


        """
        try:
            result = retriever_object.get_relevant_documents(question.lower())
            self.logger.info("Get relevent documents successfully.")
            return result

        except Exception as e:
            self.logger.error(f"Error while getting relevent documents. Error :\n {e}")

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
        try:
            result_ids = [doc.metadata["unique_id"] for doc in top_k_relevent_documents]
            self.logger.info("retrieved relevant column indexes")
            return list(set(result_ids))

        except Exception as e:
            self.logger.error(
                f"Error while getting relevent column indexes. Error :\n {e}"
            )