import logging

import pandas as pd
from core.utils.client_utils import (  # type: ignore
    get_retriever_client,
    get_vectordb_client,
)
from core.utils.read_config import (
    app_database_config,
    cloud_config,
    config,
    initialize_config,
)
from src.query_insights.utils.time_logging import timing_decorator
from src.query_insights.utils.utils import read_data
from core.utils.read_config import initialize_config, app_database_config, cloud_config
from core.utils.client_utils import get_retriever_client, get_vectordb_client # type: ignore
from .entity_extraction import EntityExtraction
from .rag import RAG, DataConverter

# log_stage_start, log_stage_end
MYLOGGERNAME = "QueryInsights"
en_core_web_model = "en_core_web_lg"


class QuestionClassifier:
    """A class for getting union of results generated from using different methods of table selection

    Parameters
    ----------
    user_config : dict
        It contains the parameters related to table selection, e.g. column selection method, top_k , path to data dictionary etc.
    data_dictionary: str
        data dictionary
    question : str
        Question that is asked by user

    Raises
    ------
    ValueError
        if any of the argument is missing or invalid.
    """

    def __init__(self, user_config, data_config):
        self.rag_flag = user_config.question_classifier.rag_flag
        self.rag_threshold_score = user_config.question_classifier.rag_threshold_score
        self.fsi_flag = user_config.question_classifier.fsi_flag
        self.recreate_rag_table_flag = (
            user_config.question_classifier.recreate_rag_table_flag
        )
        self.qc_vectordb_table_name = (
            user_config.question_classifier.qc_vectordb_table_name
        )
        self.question_template_path = (
            data_config.path.question_classifier.question_classification_data_path
        )
        self.tracl0_data_path = data_config.path.question_classifier.track0__data_path
        self.bucket_name = cloud_config.domain_storage.account_name
        self.logger = logging.getLogger(MYLOGGERNAME)

    def read_questionaire(self, path, bucket_name):
        """
        Funtion to read the questionaire for rag and track 0.

        Parameters
        ----------
        path : path to the questionaire

        returns
        -------

        pd.DataFrame"""

        return read_data(path, self.bucket_name)
        

    # @timing_decorator(track_app_start=False)
    def find_category(self, question):
        """
        Function to get the category of the question to which it falls into

        Parameters
        ----------
        None

        Returns
        -------
        dictionary
            category of the question and the similarity score of the question.

        """
        self.df_questions = self.read_questionaire(
            self.question_template_path, self.bucket_name
        )
        vectordb_config = config.embedding_details
        try:
            vector_db = get_vectordb_client(
                vectordb_config,
                self.df_questions,
                module_name="question_classification",
            )
            vector_db.insert_data()
            category = vector_db.search(question)
            self.logger.info(f"The question falls under the category :{category}")
            return category
        except Exception as e:
            self.logger.error(
                f"Error predicting the category for the question. Developer intervention Required ! Error : {e}"
            )
        return
