import copy
import re
import tracemalloc
import nltk
from core.utils.read_config import cloud_config
from src.query_insights.facilitator import (
    ConfigValidation,
    DataLoader,
    DataProcessor,
    FolderManager,
    Generic,
    KnowledgeBase,
    SkipFlag,
)
from src.query_insights.query_to_chart import QueryToChart
from src.query_insights.question_classifier import QuestionClassifier
from src.query_insights.table_selection import TableSelection
from src.query_insights.table_to_insights import TableToInsights
from src.query_insights.text_to_sqlquery import TextToSqlQuery
from src.query_insights.utils.time_logging import (
    initialize_app_start_time,
    save_timing_info_and_merge,
)
from src.query_insights.utils.utils import process_historical_guidelines

nltk.download("punkt")
nltk.download("stopwords")


class BuiltInInitializations:
    """BuiltInInitializations class that initializes all common parameters and common functionalities for all 3 tracks and questions
    Parameters
    ----------
    user_config : dict
        It contains the parameters used to interact with the OpenAI GPT-4 API to generate insights from data.
        It includes the user interface, API call, why question threshold, table rows limit when token limit exceeds.
    data_config : dict
        It contains the paths to the input data, database, output folder, and data dictionaries.
    model_config : dict
        It contains the model_params(like engine, temperature, max_tokens...), system_role, static prompt and guidelines to follow for all the tracks
    debug_config : dict
        It handles errors such as invalid columns, syntax errors, and ALTER or DROP TABLE queries.
    Raises
    ------
    ValueError
        if any of the initialization or functionality is incorrect .

    """

    def __init__(
        self,
        user_config: dict,
        data_config: dict,
        model_config: dict,
        debug_config: dict,
    ):
        # config parameters
        # log_process_start()
        self.user_config = user_config
        self.data_config = data_config
        self.model_config = model_config
        self.debug_config = debug_config

        print(">>> Generic Initialisation.")
        self.generic_initializations = Generic(
            user_config=self.user_config,
            data_config=self.data_config,
            model_config=self.model_config,
            debug_config=self.debug_config,
        )

        print(">>> Initialising dataloader.")
        self.dataloader = DataLoader(
            user_config=self.user_config,
            data_config=self.data_config,
            model_config=self.model_config,
            debug_config=self.debug_config,
        )
        print(">>> Loaading Datadictionary.")
        self.dataloader._db_and_dictionary_loading()

        print(">>> Validation Configuration")
        self.config_validation = ConfigValidation(
            user_config=self.user_config,
            data_config=self.data_config,
            model_config=self.model_config,
            debug_config=self.debug_config,
        )
        self.config_validation._config_validation_calling_and_response_json_initialization()

    def runtime_functions(self, question, additional_context):
        """Common functionalites for all 3 tracks and for different questions

        Raises
        ------
        ValueError
            if any of the functionality is incorrect .

        """

        self.foldercreation = FolderManager(
            user_config=self.user_config,
            data_config=self.data_config,
            model_config=self.model_config,
            debug_config=self.debug_config,
            question=question,
            additional_context=additional_context,
        )
        self.foldercreation._folder_creation_for_each_question()

        self.dataprocessor = DataProcessor(
            user_config=self.user_config,
            data_config=self.data_config,
            model_config=self.model_config,
            debug_config=self.debug_config,
            existing_question=self.foldercreation.existing_question,
            output_path=self.foldercreation.output_path,
            question=question,
        )
        self.dataprocessor._preprocess(
            question=question, additional_context=additional_context
        )

        self.skip_flag = SkipFlag(
            user_config=self.user_config,
            data_config=self.data_config,
            model_config=self.model_config,
            debug_config=self.debug_config,
            existing_question=self.foldercreation.existing_question,
            existing_status=self.foldercreation.existing_status,
            prev_folder_path=self.foldercreation.prev_folder_path,
            prev_runtime_result_exists=self.foldercreation.prev_runtime_result_exists,
            bot_history=self.foldercreation.bot_history,
        )
        self.skip_flag._followup()

        if self.data_config.path.question_classification:
            obj_qc = QuestionClassifier(
                user_config=self.user_config, data_config=self.data_config
            )
            self.question_category = obj_qc.find_category(question)
        else:
            self.question_category = self.data_config.path.default_classification

        if self.question_category == "Historical":
            raw_data_dictionary = copy.deepcopy(
                self.dataloader.historical_db_data_dictionary
            )

            # modified_guidelines = process_historical_guidelines(
            #     self.data_config.path.guidelines_athena,
            #     self.model_config["text_to_query"]["prompts"]["guidelines"],
            # )
            # self.model_config["text_to_query"]["prompts"][
            #     "guidelines"
            # ] = modified_guidelines
        else:
            raw_data_dictionary = copy.deepcopy(
                self.dataloader.reporting_db_data_dictionary
            )

        # raw_data_dictionary = copy.deepcopy(self.dataloader.data_dictionary)

        if self.user_config.table_selection_parameters.table_selection:
            tableselection = TableSelection(self.user_config)
            (
                self.filtered_data_dictionary,
                self.match_detail_dictionary,
                # self.unmatched_entities,
            ) = tableselection.get_datadictionary_after_tableselection(
                raw_data_dictionary, question
            )
            # print(self.filtered_data_dictionary)
            self.dataloader.data_dictionary = copy.deepcopy(
                self.filtered_data_dictionary
            )
        else:
            self.dataloader.data_dictionary = copy.deepcopy(raw_data_dictionary)


initialize_app_start_time()


class InsightsPro(BuiltInInitializations):
    """Main class that controls individual tracks outputs. Given a user query that may have additional context, this class provides charts and insights to best answer the given query.

    Parameters
    ----------
    user_config : dict
        It contains the parameters used to interact with the OpenAI GPT-4 API to generate insights from data.
        It includes the user interface, API call, why question threshold, table rows limit when token limit exceeds.
    data_config : dict
        It contains the paths to the input data, database, output folder, and data dictionaries.
    model_config : dict
        It contains the model_params(like engine, temperature, max_tokens...), system_role, static prompt and guidelines to follow for all the tracks
    debug_config : dict
        It handles errors such as invalid columns, syntax errors, and ALTER or DROP TABLE queries.
    question : str, optional
        Business user query, by default None
    additional_context : str, optional
        Additional context to answer the question, by default None
    language : str, optional
        Language to answer the question in, for example "english", "spanish", "german", by default "english"

    Raises
    ------
    ValueError
        if any of the argument is missing or invalid.

    """

    def __init__(
        self,
        # config parameters
        user_config: dict,
        data_config: dict,
        model_config: dict,
        debug_config: dict,
    ):
        print(">>> Initialising Insights Pro")
        BuiltInInitializations.__init__(
            self, user_config, data_config, model_config, debug_config
        )

    # @timing_decorator
    def text_to_query(self, question, additional_context, language):
        """Track 1: Given a user query and dataset, generate SQL and run that SQL on the dataset to get an output dataframe for further processing.

        If you want to run all the tracks, then pass the question parameter only once in the track 1. Track 2 and track 3 will automatically take from track 1.
        If you want to run track 2 or track 3 separately without running track 1, then pass the question parameter directly while calling respective functions. Don't call track 1 with the same class object if you want to do this.

        Parameters
        ----------

        Returns
        -------
        dict
            Output format changes depending on track 1 execution resulted in success, failure or skip.

            If it's success, format::

                return_value = {
                    "status": "success",
                    "output": Tuple[pd.DataFrame, dict],
                }

            If it's failure, format::

                return_value = {
                    "status": "failure",
                    "output": error_message,
                }

            If it's skip, format::

                return_value = {
                    "status": "skip",
                    "output": skip_reason,
                }
        dict
            # JSON is created to be used for front-end applications.

        """
        # print("@@@@@@@@@@@@@@@", self.data_config, "^^^^^^^^^^^^")
        BuiltInInitializations.runtime_functions(self, question, additional_context)
        # set_current_track("track1")
        self.track1_ins = TextToSqlQuery(
            user_config=self.user_config,
            data_config=self.data_config,
            model_config=self.model_config,
            debug_config=self.debug_config,
            question=question,
            additional_context=additional_context,
            language=language,
            question_category=self.question_category,
            logging_level="INFO",
        )
        print(f"Memory Consumption layer 2 : after creating a TextTpSqlQuery object : {tracemalloc.get_traced_memory()}")

        self.track1_output = self.track1_ins.text_to_query(
            self.generic_initializations,
            self.foldercreation,
            self.dataloader,
            self.config_validation,
            self.dataprocessor,
            self.skip_flag,
        )

        print(f"Memory Consumption layer 2 : after calling text_to_query func : {tracemalloc.get_traced_memory()}")

        self.alltracks_status = [self.track1_output["status"], None, None]
        self.foldercreation.existing_status = [self.track1_output["status"], None, None]
        self.tracks_flag = [True, False, False]
        if self.user_config.run_knowledge_base_similarity_check:
            knowledge_base = KnowledgeBase()
            knowledge_base.update_knowledgebase(
                self.tracks_flag,
                self.generic_initializations,
                self.foldercreation,
                self.dataloader,
                self.config_validation,
                self.dataprocessor,
                self.skip_flag,
                self.alltracks_status,
                self.track1_ins.all_tokens,
                self.generic_initializations.feedback,
                self.track1_ins,
            )
        # save_timing_info_to_csv(self.generic_initializations._fs, self.foldercreation.text_to_query_path, "TIMESTAMPS.CSV")
        # log_process_end("track1", "track1")
        # save_timing_history(self.generic_initializations._fs, self.foldercreation.text_to_query_path, "TIMESTAMPS.CSV", "track1", "track1")
        # save_timing_info_and_merge(
        #     cloud_config.domain_storage.account_name,
        #     self.data_config.path.time_logging_excel_path,
        #     self.generic_initializations._fs,
        #     self.foldercreation.output_path,
        #     "TIMESTAMPS.csv",
        #     cloud_config.cloud_provider,
        # )
        print(f"Memory Consumption layer 2 : end of track 1 process : {tracemalloc.get_traced_memory()}")
        return self.track1_output, self.track1_ins.response_json

    def query_to_chart(
        self,
        question,
        additional_context,
        question_path,
        language,
        track1_output_table = None,
        track1_output_table_dict = None,
    ):
        """Track 2: Given user query, generate python code that generates chart and display to the user.

        If you want to run track 2 or track 3 separately without running track 1, then pass the track1_output_table and track1_output_table_dict.

        Parameters
        ----------
        track1_output_table : pd.DataFrame, optional
            Output of text_to_query function by running the SQL generated to answer the ``question``, by default None
        track1_output_table_dict : dict, optional
            Data dictionary of ``track1_output_table`` parameter, by default None

        Returns
        -------
        dict
            Output format changes depending on track 1 execution resulted in success, skip or failure.

            If it's success, format::

                return_value = {
                    "status": "success",
                    "output": (chart object, track 1 table),
                }

            If chart object is None, format::

                return_value = {
                    "status": "skip",
                    "output": (None, track 1 table),
                }

            If track 2 have an error, format::

                return_value = {
                    "status": "failure",
                    "output": (error message, track 1 table),
                }

        dict
            # JSON is created to be used for front-end applications.

        """
        if self.user_config.execute_track2:
            # set_current_track("track2")

            self.track2_ins = QueryToChart(
                user_config=self.user_config,
                data_config=self.data_config,
                model_config=self.model_config,
                debug_config=self.debug_config,
                question=question,
                question_path=question_path,
                additional_context=additional_context,
                language=language,
                track1_output_table=track1_output_table,
                track1_output_table_dict=track1_output_table_dict,
                logging_level="INFO",
            )

            self.track2_output = self.track2_ins.query_to_chart(
                self.generic_initializations,
                self.foldercreation,
                self.config_validation,
                self.dataprocessor,
                self.skip_flag,
            )
            if self.tracks_flag[0]:
                self.alltracks_status = [
                    self.track1_output["status"],
                    self.track2_output["status"],
                    None,
                ]
            self.foldercreation.existing_status = [
                self.track1_output["status"],
                self.track2_output["status"],
                None,
            ]
            self.tracks_flag = [True, True, False]
            if self.user_config.run_knowledge_base_similarity_check:
                knowledge_base = KnowledgeBase()
                knowledge_base.update_knowledgebase(
                    self.tracks_flag,
                    self.generic_initializations,
                    self.foldercreation,
                    self.dataloader,
                    self.config_validation,
                    self.dataprocessor,
                    self.skip_flag,
                    self.alltracks_status,
                    self.track2_ins.all_tokens,
                    self.generic_initializations.feedback,
                    self.track2_ins,
                )
            # log_process_end("track2", "track2")
            # save_timing_history(self.generic_initializations._fs, self.foldercreation.query_to_chart_path, "TIMESTAMPS.CSV", "track2", "track2")\
            # save_timing_info_to_csv(self.generic_initializations._fs, self.foldercreation.query_to_chart_path, "TIMESTAMPS.CSV")
            return self.track2_output, self.track2_ins.response_track2
        else:
            return ['',{'Request JSON': {'question': ''},'Response JSON': {'status': ['succes', 'skip'], 'type': 'Chart'}}]



    def table_insights(
        self,
        question,
        additional_context,
        language,
        track1_output_table = None ,
        track1_output_table_dict = None ,
    ):
        """Track 3: Given user query, generate python code to derive insights on the underlying table and summarize to a business audience.

        If you want to run track 2 or track 3 separately without running track 1, then pass the track1_output_table and track1_output_table_dict.

        Parameters
        ----------
        track1_output_table : pd.DataFrame, optional
            Output of text_to_query function by running the SQL generated to answer the ``question``, by default None
        track1_output_table_dict : dict, optional
            Data dictionary of ``track1_output_table`` parameter, by default None

        Returns
        -------
        dict
            Output format changes depending on track 3 execution resulted in success or failure.

            If it's success, format::

                return_value = {
                    "status": "success",
                    "output": actual_output,
                }

            If it's failure, format::

                return_value = {
                    "status": "failure",
                    "output": error_message,
                }

        dict
            # JSON is created to be used for front-end applications.

        """
        if self.user_config.execute_track3:
            # set_current_track("Track 3")
            self.track3_ins = TableToInsights(
                user_config=self.user_config,
                data_config=self.data_config,
                model_config=self.model_config,
                debug_config=self.debug_config,
                question=question,
                additional_context=additional_context,
                language=language,
                track1_output_table=track1_output_table,
                track1_output_table_dict=track1_output_table_dict,
                logging_level="INFO",
            )

            self.track3_output = self.track3_ins.table_insights(
                self.generic_initializations,
                self.foldercreation,
                self.dataloader,
                self.config_validation,
                self.dataprocessor,
                self.skip_flag,
            )

            if self.tracks_flag[0]:  # and self.tracks_flag[1]:
                self.alltracks_status = [
                    self.track1_output["status"],
                    # self.track2_output["status"],
                    self.track3_output["status"],
                ]

            self.tracks_flag = [True, True, True]
            self.foldercreation.existing_status = [
                self.track1_output["status"],
                # self.track2_output["status"],
                self.track3_output["status"],
            ]
            if self.user_config.run_knowledge_base_similarity_check:
                knowledge_base = KnowledgeBase()
                knowledge_base.update_knowledgebase(
                    self.tracks_flag,
                    self.generic_initializations,
                    self.foldercreation,
                    self.dataloader,
                    self.config_validation,
                    self.dataprocessor,
                    self.skip_flag,
                    self.alltracks_status,
                    self.track3_ins.all_tokens,
                    self.generic_initializations.feedback,
                    self.track3_ins,
                )
            # log_process_end("track3", "track3")
            # save_timing_history(self.generic_initializations._fs, self.foldercreation.table_to_insights_path, "TIMESTAMPS.CSV", "track3", "track3")
            # save_combined_history(self.generic_initializations._fs, self.foldercreation.output_path, "COMBINED_TIMESTAMPS.CSV")
            save_timing_info_and_merge(
                cloud_config.domain_storage.account_name,
                self.data_config.path.time_logging_excel_path,
                self.generic_initializations._fs,
                self.foldercreation.output_path,
                "TIMESTAMPS.csv",
                cloud_config.cloud_provider,
            )
            return self.track3_output, self.track3_ins.response_track3
        else:
            return ['',{'Request JSON': {'question': ''},'Response JSON': {'status': ["success", "skip"], 'type': 'insights'}}]
