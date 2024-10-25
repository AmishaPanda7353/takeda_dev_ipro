import datetime
import json
import logging
import os
import posixpath as pp
import random
import gc
import timeit
import traceback
import tracemalloc
import openai
import pandas as pd
import spacy
from core.database.database_factory import DatabaseFactory
from core.utils.client_utils import get_database_client, get_storage_client
from core.utils.read_config import (
    cloud_config,
    config,
    database_config,
    domain_db_config,
)
from src.query_insights.utils.utils import write_data

from .facilitator import (
    ConfigValidation,
    DataLoader,
    DataProcessor,
    FolderManager,
    SimilarityAnalyzer,
    SkipFlag,
)
from .sql_generator.text_to_query_generator import BotResponse, TextToQuery
from .utils.time_logging import timing_decorator
from .utils.utils import (
    SensitiveContentError,
    TimeoutError,
    TokenLimitError,
    download_nltk_data,
    download_spacy_data,
    format_dataframe,
    upload_data,
)

# SPACY_MODEL global variable will be created when init fuction is called for the first time spacy.load("en_core_web_lg")
LOADED_SPACY_MODEL = None
NLTK_MODEL = None
class TextToSqlQuery:
    """Given a user query and dataset, generate SQL and run that SQL on the dataset to get an output dataframe for further processing.

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
    en_core_web_model : str, optional
        Specify which Spacy web model is to be used. For example "en_core_web_sm", "en_core_web_md", "en_core_web_lg".
        If unable to download en_core_web model, path of the model can be provided manually here.
        If providing the directory location of the model, make sure it is a valid string, by default "en_core_web_lg"
    question : str
            New question from the user
    additional_context : str, optional
            Additional context to answer the question, by default None
    language : str, optional
            Language to answer the question in, for example "english", "spanish", "german", by default "english"

    logging_level : str, optional
        Level or severity of the events they are used to track. Acceptable values are ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], by default "WARNING", by default "WARNING"

    Raises
    ------
    ValueError
        if any of the argument is missing or invalid.

    """

    def __init__(
        self,
        user_config: dict,
        data_config: dict,
        model_config: dict,
        debug_config: dict,
        question: str = None,
        language: str = "english",
        question_category: str = None,
        additional_context: str = None,
        en_core_web_model: str = "en_core_web_lg",
        logging_level: str = "INFO",
    ) -> None:
        print(f"Memory Consumption layer 3 : before init function : {tracemalloc.get_traced_memory()}")
        self.question = question
        self.additional_context = additional_context
        self.language = language

        self.user_config = user_config
        self.data_config = data_config
        self.model_config = model_config
        self.debug_config = debug_config

        self.logging_level = logging_level
        self.question_category = question_category
        self.all_tokens = []
        self.business_overview = None
        self.MYLOGGERNAME = "QueryInsights"
        self.logger = logging.getLogger(self.MYLOGGERNAME)

        # Download NLTK Data for preprocessing for questions
        self.nltk_error_flag = download_nltk_data()
        self.logger.info("Downloading NLTK data data")
        print(f"Memory Consumption layer 3 : after download NLTK : {tracemalloc.get_traced_memory()}")
        if self.nltk_error_flag:
            raise ValueError(
                "Failed to download NLTK punkt/stopwords. Please check with IT team."
            )

        # If en_core_web_model is not a path, download the model if it doesn't exist

        # If en_core_web_model is a path
        if "\\" in en_core_web_model or "/" in en_core_web_model:
            # If path doesn't exist
            if not os.path.isdir(en_core_web_model):
                raise ValueError("Path to en_core_web_model doesn't exist.")
        # If en_core_web_model is not a directory
        elif (
            "/" not in en_core_web_model or "\\" not in en_core_web_model
        ) and not os.path.isdir(en_core_web_model):
            # Download NLTK Data for preprocessing for questions
            self.spacy_error_flag = download_spacy_data(
                en_core_web_model=en_core_web_model
            )
            print(f"Memory Consumption layer 3 : after downloading spacy data : {tracemalloc.get_traced_memory()}")
            self.logger.info("Downloading spacy data")
            if self.spacy_error_flag:
                raise ValueError(
                    "Failed to download Spacy data. Please check with IT team."
                )
        # If provided path doesn't exist, then raise path error
        elif not os.path.isdir(en_core_web_model):
            raise ValueError("Path to en_core_web_model doesn't exist.")

        # Loading the spaCy language model and getting list of stop words.
        if self.user_config.run_knowledge_base_similarity_check:
            self._load_spacy(en_core_web_model)
            self.nlp = SPACY_MODEL
            print(f"Memory Consumption layer 3 : after loading the spacy model : {tracemalloc.get_traced_memory()}")
            self.stop_words = spacy.lang.en.STOP_WORDS
        else:
            print(f"Not Loading the Spacy Modules as application is not configured for the NLP tasks.")

        print(f"Memory Consumption layer 3 : end of init : {tracemalloc.get_traced_memory()}")


    def _load_spacy(self, model_name="en_core_web_lg"):
        self.logger.info(f"Attempting to load space model : {model_name}")
        global SPACY_MODEL
        global LOADED_SPACY_MODEL
        if LOADED_SPACY_MODEL == model_name:
            # there is a spacy model
            if SPACY_MODEL is not None:
                # it is as same as current requested model
                self.logger.info(f"Spacy Model {model_name} already exist. saving time by not re-loading.")
                return 
            else:
                # it is a new model different from loaded model
                SPACY_MODEL = spacy.load(model_name) #spacy.load("en_core_web_lg")
                self.logger.info(f"Spacy Model {model_name} loaded succussfully. Hypothetical case ! Developer attemtion required")  
                return
        else:
            # there is no spacy model loaded
            self.logger.info(f"Either loaded spacy model is different or model is not loaded. Attempting to load model {model_name}")
            SPACY_MODEL = spacy.load(model_name)
            LOADED_SPACY_MODEL = model_name
            self.logger.info(f"Spacy Model {model_name} successfully loaded.")
            return 
    
    @timing_decorator(track_app_start=True)
    def text_to_query(
        self,
        generic_initializations,
        foldercreation,
        dataloader,
        config_validation,
        dataprocessor,
        skip_flag,
    ):
        """This function is responsible for running track1(from text to query)

        Parameters
        ----------
        generic_initializations : object
            object for common attributes

        foldercreation : object
            object for FolderManager class

        dataloader : object
            object for DataLoader class

        config_validation : object
            object for ConfigValidation class

        dataprocessor : object
            object for DataProcessor class

        skip_flag : object
            object for SkipFlag class

        Returns
        -------


        """
        track1_start_time = timeit.default_timer()
        
        if cloud_config.cloud_provider != "s3":
            foldercreation._individual_track_folder_creation("01_text_to_query")
        db_details = database_config["reporting_db"] #defaulting to reporting db connection. 

        if self.question_category == "Historical":
            # db_details = database_config["historical_db"]
            self.business_overview = generic_initializations.business_overview_athena
        else:
            # db_details = database_config["reporting_db"]
            self.business_overview = generic_initializations.business_overview

        # database connection string
        self.db_factory = DatabaseFactory(
            get_database_client(db_details.domain_database)
        )
        db_connection = self.db_factory.create_database_connection(db_details)

        if skip_flag.is_followup:
            if hasattr(skip_flag, "model_tokens"):
                if skip_flag.model_tokens is not None:
                    self.all_tokens.append(
                        {"Followup question tag": skip_flag.model_tokens.to_dict()}
                    )

        track1_similarity_chk, similar_response, similar_question = False, None, None
        # Check if there is a similar question present and it's track 1 result is 'success'.
        # TODO: Update this logic using feedback also once it is implemented.

        if self.user_config.run_knowledge_base_similarity_check:
            # skips the similarity check based on the knowlede base.
            # --> this is skipped by default as this feature is obsolete.

            if (
                foldercreation.similarity[0]
                and foldercreation.similarity[1]["status"][0] == "success"
            ):
                # Adding 'similar' key to data dictionary with 'Yes' for all the columns
                # which are identified as important for the similar query.
                (
                    columns_list,
                    similar_question,
                    similar_response,
                ) = foldercreation._identify_columns_for_similar_query(
                    self.nlp, self.stop_words, dataloader.data_dictionary
                )
                track1_similarity_chk = True

                for table_name in dataloader.data_dictionary.keys():
                    columns = dataloader.data_dictionary[table_name]["columns"]
                    for column in columns:
                        # For all the columns in the list with format tablename.columnname.
                        if column["name"] in [
                            col for col in columns_list if "." not in col
                        ]:
                            column["similar"] = "Yes"
                        # For all the columns in the list with just columnname.
                        if column["name"] in [
                            col.split(".")[1]
                            for col in columns_list
                            if col.startswith(table_name)
                        ]:
                            column["similar"] = "Yes"

        if hasattr(self, 'nlp'):
            del self.nlp
            gc.collect()
            print(f"Memory Consumption layer 3 : after deleting self.nlp : {tracemalloc.get_traced_memory()}")
            self.logger.info("deleted nlp variable for some free space")

        # to display the question being used for the process in the logger
        if hasattr(foldercreation, "bot_history") and (
            foldercreation.bot_history is not None
        ):
            history = foldercreation.bot_history
            logger_question = " ; ".join([q for [q, a] in history])
        else:
            logger_question = self.question
        self.logger.info(f"Question to the API: {logger_question}")
        self.logger.info(f"Question to be answered in {self.language} language")
        self.logger.info(f"Additional Context to the API: {self.additional_context}")

        if dataprocessor.why_qn_flag:
            skip_reason = f"As given question {self.question} is a why question, user query to SQL generation will be skipped."
            self.logger.info(skip_reason)
            self.track1_output_table = None  # Default
            self.track1_output_table_dict = None  # Default
            return_value = {"status": "skip", "output": (skip_reason, None)}
            return return_value

        self.logger.info("SQL generation started.")
        skip_flag._skip_model_kb()
        self.logger.info(
            f"For SQL generation: existing_question check is {foldercreation.existing_question} and its status in {foldercreation.existing_status[0]}."
        )
        if skip_flag.skip_track1_followup:
            self.logger.info(
                "SQL generation is skipped because this question is identified as a follow up question not related to SQL query"
            )
        skip_model_flag = skip_flag.skip_model_kb or skip_flag.skip_track1_followup
        # print(foldercreation.output_path)
        self.track1_ins = TextToQuery(
            user_config=self.user_config,
            model_config=self.model_config,
            data_config=self.data_config,
            debug_config=self.debug_config,
            question=self.question,
            additional_context=self.additional_context,
            language=self.language,
            data_dictionary=dataloader.data_dictionary,
            business_overview=self.business_overview,
            bot_history=foldercreation.bot_history,
            db_factory=self.db_factory,
            foldercreation=foldercreation,
            similarity=[
                track1_similarity_chk,
                (similar_question, similar_response),
            ],
            skip_model=skip_model_flag,
            fs=generic_initializations._fs,
        )

        # self.track1_ins.conn = dataloader.conn
        # self.track1_ins.db_factory = dataloader.db_factory
        self.track1_ins.get_query_suggestion(db_connection)
        self.logger.info("SQL generation completed.")
        error_message = self.track1_ins.query_error_message

        self.track1_output_query = self.track1_ins.output_query
        self.track1_output_table_dict = self.track1_ins.output_table_dict
        self.track1_output_table = self.track1_ins.output_table

        if hasattr(self.track1_ins, "query_model_tokens"):
            if self.track1_ins.query_model_tokens is not None:
                token_str = self.track1_ins.query_model_tokens
                token_dict = {
                    "completion_tokens": token_str.completion_tokens,
                    "prompt_tokens": token_str.prompt_tokens,
                    "total_tokens": token_str.total_tokens,
                }
                self.all_tokens.append({"Track 1": token_dict})

        self.track1_error_message = error_message

        if error_message is None:
            self.logger.debug(
                f"First 5 rows of the table after running the generated SQL is given below:\n{self.track1_ins.output_table.head()}"
            )
            self.logger.debug(
                f"Generated table's data dict is given below:\n{self.track1_ins.output_table_dict}"
            )

            return_value = {
                "status": "success",
                "output": (
                    self.track1_output_query,
                    self.track1_output_table,
                    self.track1_output_table_dict,
                ),
            }
        else:
            return_value = {
                "status": "failure",
                "output": (error_message, self.track1_output_query),
            }

        bot_response = ""
        self.completion_response = ""

        if bool(self.user_config.bot_response):
            if return_value["status"] == "failure":
                if self.user_config.bot_response == "rule_based":
                    # to generate the bot response using hardcoded custom responses
                    bot_response = BotResponse(mode="rule_based").get_bot_error_message(
                        error_message
                    )
                elif self.user_config.bot_response == "model_based":
                    # To use davinci 003 for bot response (currently not in use)
                    bot_response_ins = BotResponse(
                        user_config=self.user_config,
                        model_config=self.model_config,
                        conversation_history=foldercreation.bot_history,
                        error_message=error_message,
                        skip_model=False,
                        mode="model_based",
                    )
                    bot_response_ins.process_sql_error()
                    bot_response = bot_response_ins.bot_response_output

                if bot_response is None:
                    bot_response = random.choice(
                        generic_initializations.completion_error_phrases
                    )
                self.completion_response = random.choice(
                    generic_initializations.completion_error_phrases
                )

        # Creation of Response JSON for track 1

        question = self.question
        type_ = "insights"

        data_dict = {}
        data_dict["insight_type"] = "sql_query"
        if return_value["status"] == "success":
            data_dict["content"] = return_value["output"][0]
            data_dict["table"] = [format_dataframe(return_value["output"][1])]
            data_dict["error"] = ""
            data_dict["showError"] = False
        elif return_value["status"] == "failure":
            data_dict["content"] = return_value["output"][1]
            data_dict["error"] = return_value["output"][0]
            data_dict["showError"] = True
        data_dict["bot_response"] = bot_response

        for key, value in self.model_config.items():
            if "engine" in value["model_params"]:
                generic_initializations.model_engine_dict[key] = value["model_params"][
                    "engine"
                ]

        request_json = {}
        response_json = {}

        request_json["question"] = question
        response_json["error"] = ""
        response_json["status"] = [return_value["status"]]
        response_json["type"] = type_
        response_json["data"] = [data_dict]
        # response_json["created_time"] = foldercreation.current_ts
        response_json["completion_response"] = self.completion_response
        response_json["response_for_history"] = bot_response
        response_json["question_index"] = foldercreation.question_index
        response_json["output_path"] = foldercreation.output_path
        response_json["engines"] = generic_initializations.model_engine_dict

        config_validation.response_json["Request JSON"] = request_json
        config_validation.response_json["Response JSON"] = response_json

        # JSON is created to be used for front-end applications.
        try:
            # Save the JSON data to a file
            if cloud_config.cloud_provider != "s3":
                output_file_path = "response.json"  # Set the desired file path
                with generic_initializations._fs.open(
                    pp.join(foldercreation.output_path, output_file_path), "w"
                ) as json_file:
                    json.dump(config_validation.response_json, json_file, indent=4)
            else:
                write_data(
                    file_path=f"{foldercreation.output_path}/response.json",
                    content=config_validation.response_json,
                )

        except Exception as e:
            self.logger.error(f"Response JSON not saved due to an error : {e}")

        track1_end_time = timeit.default_timer()
        self.response_json = config_validation.response_json
        self.logger.info(
            f"Time taken to run track 1: {round(track1_end_time - track1_start_time, 2)} seconds."
        )

        if config.user_config_domains.mcd.save_track_output:

            track_status = {
                # "question_id": [self.question_id],
                "question": [self.question],
                "run_ts": [datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")],
                "track": [1],
                "experiment": [self.data_config.path.exp_name],
                "output_path": [f"{foldercreation.output_path}/01_text_to_query"],
                "engine": [self.user_config.connection_params.api_type.text_to_query],
                "track_latency": [round(track1_end_time - track1_start_time, 2)],
                # "LLM_latency":[self.track1_ins.latency],
                "LLM_Response": [self.track1_output_query],
                # "SQL_executed":[self.track1_ins.sql_run]
            }
            if cloud_config.cloud_provider != "s3":
                pd.DataFrame(track_status).to_csv(
                    pp.join(
                        foldercreation.output_path,
                        "01_text_to_query",
                        "track_run_status.csv",
                    ),
                    index=False,
                )
            else:
                write_data(
                    file_path=f"{foldercreation.output_path}/01_text_to_query/track_run_status.csv",
                    content=pd.DataFrame(track_status),
                )

        # if config.cloud_details.mcd.cloud_provider == "s3":
        #     upload_data(self.data_config.path.exp_name)
        print(f"Memory Consumption layer 3 : end of texg to query : {tracemalloc.get_traced_memory()}")
        return return_value
