import logging

from core.utils.read_config import config

from .entity_extraction import EntityExtraction
from .rag import RAG, DataConverter
from src.query_insights.utils.time_logging import timing_decorator
# log_stage_start, log_stage_end
MYLOGGERNAME = "QueryInsights"


class TableSelection:
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

    def __init__(self, user_config):
        self.rag_flag = user_config.table_selection_parameters.rag_flag
        self.rag_methods = user_config.table_selection_parameters.rag_methods
        self.embedding_retrieval_methods = (
            user_config.table_selection_parameters.embedding_retrieval_methods
        )
        self.keywords_matching_methods = (
            user_config.table_selection_parameters.keywords_matching_methods
        )
        self.logger = logging.getLogger(MYLOGGERNAME)
        self.entity_flag = user_config.table_selection_parameters.entity_flag
        self.entity_extraction_methods = (
            user_config.table_selection_parameters.entity_extraction_methods
        )
        
    @timing_decorator(track_app_start=False)
    def get_datadictionary_after_tableselection(self, data_dictionary, question):
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
        # log_stage_start("Table_selection_results", "Track 1")
        all_columns_ids = []
        final_data_dict = {}

        if self.rag_flag:
            self.logger.info("RAG Enabled. filtering the data dictionary using RAG.")
            rag = RAG(data_dictionary, question)

            for rag_method in self.rag_methods:
                if rag_method in self.embedding_retrieval_methods:
                    embedding_indexes = rag.embedding_retrievals(
                        config.embedding_details
                    )
                    print(f"Indexes through Embeddings : {embedding_indexes}")
                    all_columns_ids.extend(embedding_indexes)

                elif rag_method in self.keywords_matching_methods:
                    keyword_matchings_indexes = rag.keyword_matching_retrievals(
                        config.retriever_details
                    )
                    print(
                        f"Indexes through Keyword matchings : {keyword_matchings_indexes}"
                    )
                    all_columns_ids.extend(keyword_matchings_indexes)
            self.logger.info("final data dictionary after table selection created")
            data_dict = rag.get_data_dict(all_columns_ids)
            print(f"Result through RAG : {data_dict}")
            # return data_dict, None, None
        
        if self.entity_flag:
            self.logger.info("Entity flag  Enabled. filtering the data dictionary using Entity.")
            self.entity_extraction = EntityExtraction(
            )
            for entity_extraction_method in self.entity_extraction_methods:
                if entity_extraction_method == "keybert":
                    final_data_dictionary, match_detail_dictionary = self.entity_extraction.keybert_entity_extraction(
                        config.entity_extraction_details, question, data_dictionary
                    )
                    print(
                        f"Result through KeyBert Entity Extraction : {self.entity_extraction.final_data_dictionary_output}"
                    )
        
        # final_data_dictionary.update(data_dict)

        self.logger.info("final data dictionary after table selection created")
        # log_stage_end("Table_selection_results", "Track 1")
        return final_data_dictionary, match_detail_dictionary
