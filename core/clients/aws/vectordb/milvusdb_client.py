import logging
import sys

import numpy as np
import openai
import pandas as pd
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)
from sentence_transformers import SentenceTransformer

from core.vectordb.vectordb_client import VectorDB

MYLOGGERNAME = "QueryInsights"


class MilvusDB(VectorDB):

    """
    A class for interacting with MilvusDB database.

    This class inherits from the `VectorDB` base class and provides methods for connecting to MilvusDB database and
    performing creating collections, inserting embeddings and searching relevant column indexes to the question.

    Parameters
    ----------
    config (dict): The configuration object containing vectordb information.
    data_frame (pd.Dataframe): Dataframe having unique index for unique colum name and other information related to that column,
                                e.g. column description, table name etc.

    """

    def __init__(self, config, data_frame):
        super().__init__(config, data_frame)

        self.host = self.config.host
        self.port = self.config.port
        self.collection_name_prefix = self.config.collection_name_prefix
        self.embedding_dim = self.config.embedding_dim
        self.model_name = self.config.embedding_model
        self.column_selection_method = self.config.column_selection_method

        self.top_k = self.config.top_k

        self.collections = {}

        if self.model_name == "openai":
            # Model platform - Can be openai or azure
            self.platform = config.connection_params.platform
            # API type - Can be openai or azure
            self.api_type = config.connection_params.api_type
            # End point of Azure OpenAI resource group
            self.api_base = config.connection_params.api_base
            # API version.
            self.api_version = config.connection_params.api_version

            openai.api_type = self.api_type
            openai.api_base = self.api_base
            openai.api_version = self.api_version
            openai.api_key = "39e6e5b05bc741f4b6fb74acf22c71ca"

            self.model = openai

        elif self.model_name == "sentence_transformer":
            self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.logger = logging.getLogger(MYLOGGERNAME)

        self.connect()
        # List all existing collections
        existing_collections = utility.list_collections()
        print(f"Existing collections: {existing_collections}")
        self.create_collections()

    def connect(self):
        """
        Function to connect with Milvus db.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        try:
            connections.connect("default", host=self.host, port=self.port)
            self.logger.info(f"Connected to Milvus at {self.host}:{self.port}")
            print(f"Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            self.logger.error(f"Error in connecting with Milvus db. Error :\n {e}")

    def create_collections(self):
        """
        Function to create a schema with unique id for each type of column_selection_method, for e.g. 'column_name', 'column_description' etc.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        try:
            base_fields = [
                FieldSchema(
                    name="unique_id",
                    dtype=DataType.INT64,
                    is_primary=True,
                    auto_id=False,
                ),
            ]

            embedding_field_names = self.column_selection_method

            for embedding_field_name in embedding_field_names:
                fields = base_fields + [
                    FieldSchema(
                        name=embedding_field_name,
                        dtype=DataType.FLOAT_VECTOR,
                        dim=self.embedding_dim,
                    )
                ]
                schema = CollectionSchema(fields)
                collection_name = (
                    f"{self.collection_name_prefix}_{embedding_field_name}"
                )

                if not utility.has_collection(collection_name):
                    print(f"Creating Collection: {collection_name}")
                    self.collections[embedding_field_name] = Collection(
                        name=collection_name, schema=schema
                    )
                    self._create_index(embedding_field_name)
                    self.logger.info(
                        "collections created for each type of column selection method"
                    )

                else:
                    print(f"Loading Collection: {collection_name}")
                    self.collections[embedding_field_name] = Collection(
                        name=collection_name
                    )
                    self.logger.info(
                        "collections loaded for each type of column selection method"
                    )

        except Exception as e:
            self.logger.error(
                f"Error in loading or creation of collection for each type of column selection method. Error :\n {e}"
            )

    def _create_index(self, embedding_field_name: str):
        """
        Function to create indexes for each type of column selection method, for e.g. 'column_name', 'column_description' etc.

        Parameters
        ----------
        embedding_field_name : str
            column selection method

        Returns
        -------
        None
        """
        try:
            collection = self.collections[embedding_field_name]
            index_params = {
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 16, "efConstruction": 500},
            }
            print(f"Creating Index for {embedding_field_name}")
            collection.create_index(
                field_name=embedding_field_name, index_params=index_params
            )
            utility.index_building_progress(embedding_field_name)
            self.logger.info(f"Indexes for collections created successfully")
        except Exception as e:
            self.logger.error(f"Indexes for collections not created. Error :\n {e}")

    def _generate_embeddings(self, text: str):
        """
        Function to generate embeddings for given text

        Parameters
        ----------
        text : str

        Returnss
        -------
        np.ndarray
            numpy array containing embeddings for given text
        """
        try:
            if self.model_name == "sentence_transformer":
                self.logger.info(
                    "Embeddings created succesfully for data dictionary using sentence transformer"
                )
                return self.model.encode([text])[0]
            elif self.model_name == "openai":
                response = self.model.Embedding.create(
                    input=text, model="text-embedding-ada-002"
                )
                # Convert JSON to DataFrame
                df = pd.json_normalize(response["data"])
                # Extract embeddings
                embeddings = np.array(df["embedding"].values[0])
                self.logger.info(
                    "Embeddings created succesfully for data dictionary using openai"
                )
                return embeddings

        except Exception as e:
            self.logger.error(
                f"Embeddings not created for data dictionary. Error :\n {e}"
            )

    def insert_data(self):
        """
        Function to insert data from Dataframe(Dataframe having information about each column from data dictionary) into collections in Milvus db

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        try:
            for embedding_field_name in self.collections:
                field_base_name = embedding_field_name
                if field_base_name not in self.data_frame.columns:
                    raise ValueError(
                        f"Column {field_base_name} does not exist in DataFrame."
                    )

                embeddings = (
                    self.data_frame[field_base_name]
                    .apply(self._generate_embeddings)
                    .tolist()
                )
                unique_ids = self.data_frame["unique_id"].tolist()

                insert_records = [
                    [int(uid) for uid in unique_ids],
                    [emb.tolist() for emb in embeddings],
                ]
                self.data_frame['embeddings'] = embeddings
                print(f"Inserting data into {embedding_field_name}")
                insert = self.collections[embedding_field_name].insert(insert_records)
                self.collections[embedding_field_name].flush()
                self.collections[embedding_field_name].load()

            self.logger.info(f"Data inserted successfully in Milvus db")

        except Exception as e:
            self.logger.error(
                f"Error in inserting the data into Milvus db. Error :\n {e}"
            )

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
        try:
            query_embedding = self._generate_embeddings(query_text).tolist()
            search_params = {"metric_type": "COSINE", "params": {"ef": self.top_k}}

            result_ids = set()

            for embedding_field_name in self.collections:
                collection = self.collections[embedding_field_name]
                collection.load()
                print(f"Searching in {embedding_field_name}")
                results = collection.search(
                    data=[query_embedding],
                    anns_field=embedding_field_name,
                    param=search_params,
                    limit=self.top_k,
                    output_fields=["unique_id"],
                )

                for hits in results:
                    result_ids.update(hit.entity.get("unique_id") for hit in hits)
            self.logger.info("searching relevant column ids for the question completed")
            print("searching relevant column ids for the question completed")
        except Exception as e:
            self.logger.error(
                f"searching for relevant column ids not completed. Error :\n {e}"
            )

        return list(set(result_ids))