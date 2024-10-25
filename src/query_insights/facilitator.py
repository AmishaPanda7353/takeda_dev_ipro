import datetime
import json
import logging
import posixpath as pp
import random
import re
import timeit
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

import boto3
import numpy as np
import pandas as pd
import s3fs
import spacy
import yaml

# log_stage_start, log_stage_end, set_current_track
from core.database.database_factory import DatabaseFactory
from core.model.model_factory import ModelFactory
from core.storage.storage_factory import StorageFactory
from core.utils.client_utils import get_database_client, get_storage_client
from core.utils.read_config import (
    cloud_config,
    cloud_secrets,
    config,
    domain_db_config,
    secrets_config,
)
from sentence_transformers import SentenceTransformer, util
from sql_metadata import Parser
from src.query_insights.utils.time_logging import timing_decorator

from .utils.cloud_config_validation import validate_cloud_config
from .utils.config_validation import (  # DataConfigValidator,
    ModelConfigValidator,
    UserConfigValidator,
)
from .utils.data_path_config_validation import validate_data_path
from .utils.database_config_validation import validate_database_config
from .utils.followup_question_tag import followup_q_tag
from .utils.model_prompts_validation import validate_model_prompts
from .utils.pre_processing import HybridQuestionClassifier
from .utils.user_config_validation import validate_user_config
from .utils.utils import (
    convert_data_dictionary_to_pandas_df,
    copy_folder_s3,
    copy_folders,
    create_logger,
    fs_connection,
    get_dummy_data_dictionary,
    get_fs_and_abs_path,
    get_table_names,
    get_word_chunks,
    load_data_dictionary,
    load_data_dictionary_from_s3,
    load_key_to_env,
    log_uncaught_errors,
    read_data,
    read_text_file,
)


class Generic:
    """Generic class that initializes all common parameters.

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
    api_key : str, optional
        API key string. If left as blank, it will look for the path in the data_config and read the key from there, by default None
    fs_key : str, optional
        Account key for connecting to the File storage. If left as blank and platform specified in the data_config (data_config.cloud_storage.platform) is not blank, it will look for the path in the data_config and read the key from there, by default None
    en_core_web_model : str, optional
        Specify which Spacy web model is to be used. For example "en_core_web_sm", "en_core_web_md", "en_core_web_lg".
        If unable to download en_core_web model, path of the model can be provided manually here.
        If providing the directory location of the model, make sure it is a valid string, by default "en_core_web_lg"
    logging_level : str, optional
        Level or severity of the events they are used to track. Acceptable values are ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], by default "WARNING", by default "WARNING"
    log_file_path : str, optional
        File path to save the logs, by default None
    verbose : bool, optional
        If `True` logs will be printed to console, by default True

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
        api_key: str = None,
        fs_key: str = None,
        logging_level: str = "INFO",
        log_file_path: str = None,
        verbose: bool = True,
    ) -> None:
        """Class constructor"""

        # config parameters
        self.user_config = user_config
        self.data_config = data_config
        self.model_config = model_config
        self.debug_config = debug_config
        # initializations
        self.logging_level = logging_level
        self.verbose = verbose
        self.fs_key = fs_key

        self.why_qn_flag = False
        self.bucket = cloud_config.domain_storage.account_name
        # Instantiate cloud service for storage client

        print(">>> Initialising storage service")
        storage_obj = StorageFactory(get_storage_client(cloud_config))

        # key connections
        prefix_url, storage_options = storage_obj.fs_connection(
            fs_connection_dict=cloud_config.domain_storage,
            fs_key=cloud_secrets.domain_storage,
        )
        # BLOB_ACCOUNT_KEY variable is set here
        self._fs, _ = get_fs_and_abs_path(
            path=prefix_url, storage_options=storage_options
        )
        logging.info(prefix_url)

        if 'openai' in list(set(self.user_config['connection_params']['api_type'].values())):
            # Load openai api key
            self.load_openai_api_key(api_key)
            self.api_key = api_key

        # else:
        #     self._fs = s3fs.S3FileSystem()

        self.MYLOGGERNAME = "QueryInsights"
        self.units_to_skip = [
            "integer",
            "count",
            "text",
            None,
            "yyyy-ww",
            "na",
            "unit of measure for the quantity",
            "float",
        ]
        self.completion_error_phrases = [
            "My apologies! Can you ask something else?",
            "Oh no! That didn't work. How about asking a different question?",
            "Sorry, I encountered an error. Maybe try a different query?",
            "Oops, I didn't catch that, can you please modify the question or change it",
            "Looks like I'm having trouble with that one. Can you ask me something else?",
            "Sorry about that, can you try a different question for me?",
            "Hm, I'm not quite sure about that. Can you ask me something else?",
        ]

        print(">>> Initialising Conversation Models")
        self.model = SentenceTransformer(self.user_config.similarity_check.model)
        # Load question classification model if threshold is specified in config

        self.model_engine_dict = {}

        # paths
        self.text_to_query_path = None
        self.query_to_chart_path = None
        self.table_to_insights_path = None
        self.log_file_path = log_file_path

        self.logger = logging.getLogger(self.MYLOGGERNAME)

        self.business_overview = None
        self.business_overview_athena = None
        self.load_question_classifier()
        self.read_business_overview()
        # self.set_language(language)
        self.feedback = None

        print(">>> validating configs")
        validate_cloud_config("../configs/cloud/cloud.yaml", "../configs/config.yaml")
        validate_data_path(
            "../configs/data_files/data_path.yaml", "../configs/config.yaml"
        )
        validate_user_config("../configs/user/user.yaml", "../configs/config.yaml")
        validate_database_config(
            "../configs/database/database.yaml", "../configs/config.yaml"
        )
        validate_model_prompts(
            "../configs/model/model_params.yaml", "../configs/model.yaml"
        )
        return

    def read_business_overview(self):
        # We check if it is already read. If not, we read it. This is to facilitate each track running individually
        print(">>> Reading Business overview")
        self.business_overview = None
        if cloud_config.cloud_provider != "s3":
            if self.business_overview is None and bool(
                self.data_config.path.business_overview_path
            ):
                mysql_file_path = self.data_config.path.business_overview_path
                # if self._fs.exists(mysql_file_path):
                with self._fs.open(mysql_file_path, "r") as file_:
                    self.business_overview_reporting = file_.read()

                athena_file_path = self.data_config.path.business_overview_path_athena
                if self._fs.exists(athena_file_path):
                    with self._fs.open(athena_file_path, "r") as file_:
                        self.business_overview_historical = file_.read()
        else:
            if self.business_overview is None and bool(
                self.data_config.path.data_glossary_path_s3
            ):
                file_path = self.data_config.path.data_glossary_path_s3
                if self._fs.exists(file_path):
                    with self._fs.open(file_path, "r") as file_:
                        self.business_overview = file_.read()
            if self.business_overview_athena is None and bool(
                self.data_config.path.data_glossary_path_s3_athena
            ):
                file_path = self.data_config.path.data_glossary_path_s3_athena
                if self._fs.exists(file_path):
                    with self._fs.open(file_path, "r") as file_:
                        self.business_overview_athena = file_.read()

    def load_openai_api_key(self, api_key=None):
        if api_key is not None:
            load_key_to_env(
                secret_key=api_key,
                env_var="OPENAI_API_KEY",
                fs=None,
            )
        else:
            load_key_to_env(
                secret_key=secrets_config.cloud_details.api_key.openai_api_key,
                env_var="OPENAI_API_KEY",
                fs=None,
            )

    def load_question_classifier(self):
        self.question_threshold = self.user_config.why_question_threshold
        if bool(self.question_threshold):
            self.classifier = HybridQuestionClassifier(embedding_model="bert")
        else:
            self.classifier = None

    def set_language(self, language):
        if language is None or not bool(str(language).strip()):
            language = "english"
        return language.lower().title()


class SimilarityAnalyzer(Generic):
    """SimilarityAnalyzer class that is used to get similar index for existing similar questions and identifying common columns for similar query

    Parameters:
    -----------
    user_config : dict
        A dictionary containing parameters for interacting with the OpenAI GPT-4 API to generate insights from data.
        This includes user interface settings, API call configuration, thresholds for "why" questions, and limits
        for table rows when the token limit exceeds.

    data_config : dict
        A dictionary containing paths to input data, databases, output folders, and data dictionaries.

    model_config : dict
        A dictionary containing model parameters such as engine, temperature, max_tokens, etc.,
        along with system roles, static prompts, and guidelines for all tracks.

    debug_config : dict
        A dictionary handling errors such as invalid columns, syntax errors, and queries like ALTER or DROP TABLE.

    Raises:
    -------
    ValueError:
        If any initialization or functionality is incorrect.

    """

    def __init__(
        self,
        user_config: dict,
        data_config: dict,
        model_config: dict,
        debug_config: dict,
    ):
        Generic.__init__(
            self,
            user_config,
            data_config,
            model_config,
            debug_config,
        )

    def check_similarity_and_get_question_index(
        self, question, questions_dict, path_dict, status_dict
    ) -> str:
        """
        Calculates similarity with other questions using sentence embeddings and cosine similarity.
        And it gets the index for the new question based on the existing question base.

        Two scenarios can happen in this function after identifying similar questions if available -
        1. If the similarity condition is False or KB has no questions, it goes with the question index declared in the "_folder_creation_for_each_question" function.
            The created index will be in the format <Prefix>_<Timestamp>_1.
        2. If the condition is True, then it takes the maximum value from the dir list and increments the secondary index.
            conditon = True
            dir_list = ['Q_20230622142919300_1', 'Q_20230622142919300_2']
            Then the new index which is returned is 'Q_20230622142919300_3'.

        Parameters
        ----------
        question : str
            New question from the user
        questions_dict : dict
            It has the existing questions from the KB as a dictionary.
            Question indexes are Keys and the questions are values.
        path_dict : dict
            Question indexes are Keys and the results path are values.
        status_dict : dict
            Question indexes are Keys and the results status are values.
        Returns
        -------
        str
            Prefix + "_" Primary index + "_" + Secondary Index
        """

        # Check if the KB has no existing questions.

        if questions_dict is None:
            new_folder_name = self.question_index
        else:
            # Looking for similar question if any in the KB using sentence embeddings and cosine similarity.
            # Get the list of questions and indexes from the dictionary.
            folders_list = [i for i in questions_dict.keys()]
            questions_list = [v for v in questions_dict.values()]
            path_list = [p for p in path_dict.values()]
            status_list = [s for s in status_dict.values()]

            # Encode the list of sentences and the standalone sentence
            encoded_list = self.model.encode(questions_list, convert_to_tensor=True)
            encoded_new_qs = self.model.encode(question, convert_to_tensor=True)

            # Calculate cosine similarity and the maximum similarity
            cos_similarities = util.pytorch_cos_sim(
                encoded_new_qs, encoded_list
            ).tolist()[0]
            max_similarity = max(cos_similarities)
            if (
                max_similarity > self.user_config.similarity_check.threshold
            ):  # Threshold for similarity
                # TODO Do this step only if the user wants the similar query to be used. After UI changes, this need to be updated.
                max_index = cos_similarities.index(max_similarity)
                similar_index = folders_list[max_index]
                similar_question = questions_list[max_index]
                similar_question_path = path_list[max_index]
                similar_status = status_list[max_index].strip("][").split(", ")
                similar_dict = {
                    "index": similar_index,
                    "score": max_similarity,
                    "question": similar_question,
                    "path": similar_question_path,
                    "status": similar_status,
                }
                self.similarity = [True, similar_dict]

                similar_indexes = [
                    i
                    for i in folders_list
                    if i.startswith(similar_index.rsplit("_", 1)[0])
                ]
                folder_name = max(similar_indexes)
                n = int(folder_name.split("_")[2]) + 1
                new_folder_name = (
                    folder_name.split("_")[0]
                    + "_"
                    + folder_name.split("_")[1]
                    + "_{}".format(n)
                )
            else:
                new_folder_name = self.question_index

        # adding random integer after timestamp to avoid issues with multiple users
        rand = random.randint(0, 10000)
        new_folder_name = f"{new_folder_name}_{rand}"
        return new_folder_name

    def _identify_columns_for_similar_query(
        self, nlp, stop_words, data_dictionary
    ) -> list:
        """
        Identifies columns for a similar query by comparing word chunks and calculating cosine similarities.
        Returns a list of identified columns, the similar question, and the corresponding SQL response.

        Steps followed in the function -
        1. Process the similar question and extract meaningful word chunks.
        2. Process the user question and additional context (if available) and extract meaningful word chunks.
            - Word chunks are identified based on Noun chunks and POS.
        3. Keep only the new word chunks which are part of the user question. Remove the common word chunks b/w similar question and the user question.
        4. Iterate over the new word chunks from the user question and the column descriptions and find the most similar column names using a Sentence Transformer model and cosine similarity.
            - For each new word chunk, columns are identified.
                (The number of such columns identified can be changed from user config file. 10 columns would be ideal.)
        5. Combine these column names with the columns extracted from the SQL query of similar question and return the final list of unique column names.

        Returns
        -------
        list
            [<similar columns list>, similar question, similar response]
        """
        # Getting the columns list from the SQL Query.
        self.logger.info(
            f"Similar question {self.similarity[1]['question']} already exists. Getting the columns from knowledge base."
        )
        # Constructing the path to the SQL query file of the similar question
        similar_qs_path = self.similarity[1]["path"]
        similar_question_path = pp.join(
            similar_qs_path, "01_text_to_query", "sql_query.sql"
        )
        # Reading the SQL Query from the similar question's folder.
        similar_response = read_text_file(similar_question_path, fs=self._fs)
        # Parsing the SQL query to extract columns
        query_columns = Parser(similar_response).columns

        # Fetching the similar question and processing it with the language model and extract meaningful word chunks.
        similar_question = self.similarity[1]["question"]
        doc = nlp(similar_question)
        all_chunks_old = get_word_chunks(doc, stop_words)

        # Processing the user question with the language model and extract meaningful word chunks.
        if self.additional_context is None:
            doc = nlp(self.question)
        else:
            doc = nlp(self.question + " " + self.additional_context)
        all_chunks_new = get_word_chunks(doc, stop_words)

        # Filtering only new word chunks present in the user question compared to the old question.
        new_chunks_only = [ch for ch in all_chunks_new if ch not in all_chunks_old]

        # Converting the raw data dictionary to a Pandas DataFrame
        data_dictionary_df = convert_data_dictionary_to_pandas_df(data_dictionary)
        data_dictionary_df = data_dictionary_df.reset_index(drop=True)
        # Creating a new column by concatenating table name and column name
        data_dictionary_df["table_column"] = (
            data_dictionary_df["table_name"] + "." + data_dictionary_df["name"]
        )
        # Getting the list of column descriptions and column names.
        description_list = data_dictionary_df["description"].tolist()
        columns_list = data_dictionary_df["name"].tolist()

        # Loading the sentence transformer model and initializing a list to store identified column names.
        possible_columns_list = []
        # Encoding the column descriptions
        encoded_list = self.model.encode(description_list, convert_to_tensor=True)
        # Looping over the new word chunks from the user question
        for i in range(len(new_chunks_only)):
            # Calculating cosine similarities between the new word chunk and column descriptions
            encoded_new_qs = self.model.encode(
                new_chunks_only[i], convert_to_tensor=True
            )
            cos_similarities = util.pytorch_cos_sim(
                encoded_new_qs, encoded_list
            ).tolist()[0]
            # Getting the indices of the highest similarity values
            max_values = sorted(
                range(len(cos_similarities)),
                key=lambda x: cos_similarities[x],
                reverse=True,
            )[: self.user_config.similarity_check.num_columns_per_chunk]
            # Getting the corresponding column names and adding it to the list
            max_value_list = [columns_list[j] for j in max_values]
            possible_columns_list = list(set(possible_columns_list + max_value_list))

        final_cols = list(set(possible_columns_list + query_columns))

        return final_cols, similar_question, similar_response


class FolderManager(SimilarityAnalyzer, Generic):
    """FolderManager class is used to creation of folders for each question and individual tracks.

    Parameters:
    -----------
    user_config : dict
        A dictionary containing parameters for interacting with the OpenAI GPT-4 API to generate insights from data.
        This includes user interface settings, API call configuration, thresholds for "why" questions, and limits
        for table rows when the token limit exceeds.

    data_config : dict
        A dictionary containing paths to input data, databases, output folders, and data dictionaries.

    model_config : dict
        A dictionary containing model parameters such as engine, temperature, max_tokens, etc.,
        along with system roles, static prompts, and guidelines for all tracks.

    debug_config : dict
        A dictionary handling errors such as invalid columns, syntax errors, and queries like ALTER or DROP TABLE.

    question : str, optional
        Business user query, by default None

    additional_context : str, optional
        Additional context to answer the question, by default None

    existing_question : bool
        flag represents if the user question is an exact match of existing questions in KB.

    output_path : str
        output folder path
    Raises:
    -------
    ValueError:
        If any initialization or functionality is incorrect.

    """

    def __init__(
        self,
        user_config: dict,
        data_config: dict,
        model_config: dict,
        debug_config: dict,
        question: str = None,
        additional_context: str = None,
    ):
        Generic.__init__(
            self, user_config, data_config, model_config, debug_config, question
        )
        self.existing_question = False
        self.output_path = None
        self.question = question
        self.additional_context = additional_context
        self.uuid = uuid.uuid4().hex

    @timing_decorator(track_app_start=False)
    def _folder_creation_for_each_question(self):
        """Create folder for each question to store output files.

        This function sets up the folder structure to save logs and outputs for each question.

        Two scenarios can happen -
            1. if it is a new question, then we are creating a new index with the prefix and the timestamp. <Prefix>_<Timestamp>_<integer%d>_<rand_integer%d%d%d%d>
            2. if it is an exisiting question, then we are fetching the question index from the KB

        """
        # set_current_track("track1")
        # log_stage_start("Folder_creation_for_each_question", "Track 1")
        try:
            if isinstance(self.question, list):
                # if the entire conversation is passed as a list [[user1, bot1], [user2, bot2], [user3, bot3], ...]
                self.bot_history = self.question.copy()
                self.question = self.question[0][0]
            elif (isinstance(self.question, str)) and (";" in self.question):
                # if all the followup questions are passes as a ";" separated list
                question_list = self.question.split(";")
                self.bot_history = [[user.strip(), None] for user in question_list]
                self.question = question_list[0]
            else:
                # else question is just a string and there is no conversation
                self.bot_history = None
            date_today = datetime.datetime.now().strftime("%d%m%Y")
            # create the output folder structure for saving logs.
            if cloud_config.cloud_provider != "s3":
                self.exp_folder = pp.join(
                    self.data_config.path.output_path, self.data_config.path.exp_name
                )
            else:
                self.exp_folder = pp.join(
                    self.data_config.path.output_path_s3,
                    date_today,
                    self.data_config.path.exp_name,
                    self.uuid,
                )
            print(f"Experiment folder formed : {self.exp_folder}")
            if cloud_config.cloud_provider != "s3" and not self._fs.exists(
                self.exp_folder
            ):
                self._fs.makedirs(self.exp_folder)
            self.output_path = self.exp_folder
            self.existing_question = False
            self.existing_status = [None, None, None]
            self.prev_folder_path = None
            self.prev_runtime_result_exists = True
            self.question_index = self.uuid
            log_file = pp.join(self.output_path, "runtime.log")
            create_logger(
                logger_name=self.MYLOGGERNAME,
                level=self.logging_level,
                log_file_path=log_file,
                verbose=self.verbose,
                fs=self._fs,
            )
            self.logger = logging.getLogger(self.MYLOGGERNAME)
        except Exception as e:
            print(e)
            """self.knowledge_base = pp.join(self.exp_folder, "Knowledge_base.xlsx")
            self.bot_training_data = pp.join(self.exp_folder, "bot_training_data.csv")

            # Initialize some Flags in case the Knowledge base file is not created yet.
            # existing_question flag represents if the user question is an exact match of existing questions in KB.
            # existing_status represents the status of Track 1, 2 and 3 from the previous run.
            self.existing_question = False
            self.existing_status = [None, None, None]
            self.prev_folder_path = None
            # index_questions_dict is the dictionary with question index from KB as keys and Questions as values.
            # index_path_dict is the dictionary with question index from KB as keys and results path as values.
            # index_status_dict is the dictionary with question index from KB as keys and results status as values.
            index_questions_dict = None
            index_path_dict = None
            index_status_dict = None
            # Simlairity variable is a list with two elements.
            # First element - Flag which represents if the user question is similar to any of the existing question.
            # Second element - dictionary - (Index of the similar existing question, Similarity Score, Similar question, path, and results status) if the first element turns out to be True.
            # If the first element is False, then second element will be None.
            # This will be updated in the check_similarity_and_get_question_index function where similarity is calculated.
            self.similarity = [False, None]

            user_question = self.question
            user_prev_question = None
            if hasattr(self, "bot_history") and (self.bot_history is not None):
                history = self.bot_history
                user_question = " ; ".join([q for [q, a] in history])
                if len(history) > 1:
                    user_prev_question = " ; ".join([q for [q, a] in history[:-1]])
            if self.additional_context is not None:
                user_question = user_question + " :: " + self.additional_context

            check_for_similar_questions = True
            # Check for question in existing knowledge base if the file is present.
            if self._fs.exists(self.knowledge_base):
                # Get details from existing knowledge base.
                kb_df = read_data(self.knowledge_base, fs=self._fs)
                kb_df.fillna("", inplace=True)
                kb_df["question_w_context"] = np.where(
                    kb_df["additional_context"] == "",
                    kb_df["question"],
                    kb_df["question"] + " :: " + kb_df["additional_context"],
                )

                index_questions_dict = (
                    kb_df.copy().set_index("index")["question_w_context"].to_dict()
                )

                index_path_dict = (
                    kb_df.copy().set_index("index")["results_path"].to_dict()
                )

                index_status_dict = (
                    kb_df.copy().set_index("index")["results_status"].to_dict()
                )

                questions = kb_df["question_w_context"].to_list()
                self.indexes_list = kb_df["index"].to_list()
                status = kb_df["results_status"].to_list()
                questions = [
                    re.sub(r"[^\w\s]", "", q).lower().strip() for q in questions
                ]

                # If it is a followup question, extract the question index and path of the previous question of same conversation
                if user_prev_question is not None:
                    user_prev_question_stripped = (
                        re.sub(r"[^\w\s]", "", user_prev_question).lower().strip()
                    )
                    if user_prev_question_stripped in questions:
                        prev_index = questions.index(user_prev_question_stripped)
                        prev_question_index = self.indexes_list[prev_index]
                        # self.prev_folder_path -- previous folder's path. default is None
                        if cloud_config.cloud_provider != "s3":
                            self.prev_folder_path = pp.join(
                                self.data_config.path.output_path,
                                self.data_config.path.exp_name,
                                prev_question_index,
                            )
                        else:
                            self.prev_folder_path = pp.join(
                                self.data_config.path.output_path_s3,
                                self.data_config.path.exp_name,
                                prev_question_index,
                            )

                # Check if question already exists in knowledge base.
                user_question_stripped = (
                    re.sub(r"[^\w\s]", "", user_question).lower().strip()
                )
                if user_question_stripped in questions:
                    self.existing_question = True
                    index = questions.index(user_question_stripped)
                    self.question_index = self.indexes_list[index]
                    self.existing_status = status[index][1:-1].split(", ")

                    # Back up of existing outputs
                    if cloud_config.cloud_provider != "s3":
                        existing_folder_path = pp.join(
                            self.data_config.path.output_path,
                            self.data_config.path.exp_name,
                            self.question_index,
                        )
                    else:
                        existing_folder_path = pp.join(
                            self.data_config.path.output_path_s3,
                            self.data_config.path.exp_name,
                            self.question_index,
                        )
                    if self._fs.exists(existing_folder_path):
                        if not self._fs.exists(
                            pp.join(existing_folder_path, "back_up")
                        ):
                            self._fs.makedirs(pp.join(existing_folder_path, "back_up"))

                        # Get a list of all files in the folder
                        if cloud_config.cloud_provider != "s3":
                            file_list = self._fs.ls(existing_folder_path)
                        else:
                            file_list = self._fs.listdir(existing_folder_path)

                        # Filter files starting with "runtime"
                        runtime_files = [
                            pp.basename(filename)
                            for filename in file_list
                            if pp.basename(filename).startswith("runtime")
                        ]

                        # Sort the files based on their names
                        sorted_runtime_files = sorted(runtime_files)

                        # Extract timestamp from the first file
                        if len(sorted_runtime_files) > 0:
                            timestamp_pattern = r"runtime_(\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}_\d{6})Z.log"
                            match = re.search(
                                timestamp_pattern, sorted_runtime_files[0]
                            )
                            files_to_copy = [
                                "question.txt",
                                "response.json",
                                sorted_runtime_files[0],
                            ]
                        else:
                            match = False
                            files_to_copy = ["question.txt", "response.json"]

                        if match:
                            previous_question_timestamp = match.group(1)
                        else:
                            previous_question_timestamp = (
                                str(datetime.datetime.now(datetime.timezone.utc))
                                .replace("+00:00", "Z")
                                .replace(":", "_")
                                .replace(" ", "_")
                                .replace(".", "_")
                            )
                        if cloud_config.cloud_provider != "s3":
                            copy_folders(
                                source_path=existing_folder_path,
                                source_folders=[
                                    "01_text_to_query",
                                    "02_query_to_chart",
                                    "03_table_to_insights",
                                    "configs",
                                ],
                                source_files=files_to_copy,
                                destination_path=pp.join(
                                    existing_folder_path,
                                    "back_up",
                                    f"{previous_question_timestamp}",
                                ),
                                fs=self._fs,
                            )
                        else:
                            copy_folder_s3(
                                source_path=existing_folder_path,
                                source_folders=[
                                    "01_text_to_query",
                                    "02_query_to_chart",
                                    "03_table_to_insights",
                                    "configs",
                                ],
                                source_files=files_to_copy,
                                destination_path=pp.join(
                                    existing_folder_path,
                                    "back_up",
                                    f"{previous_question_timestamp}",
                                ),
                                fs=self._fs,
                            )
                    else:
                        self._fs.makedirs(existing_folder_path)
                        print(f"Experiment folder created. {existing_folder_path}")
                        self.existing_question = False
                        self.existing_status = [None, None, None]
                        check_for_similar_questions = False

                else:
                    prefix = "Q"
                    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
                    self.question_index = prefix + f"_{now}_1"
            else:
                prefix = "Q"
                now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
                self.question_index = prefix + f"_{now}_1"
            # get question index based on existing folders.
            if (not self.existing_question) and check_for_similar_questions:
                self.question_index = self.check_similarity_and_get_question_index(
                    user_question,
                    index_questions_dict,
                    index_path_dict,
                    index_status_dict,
                )
            if cloud_config.cloud_provider != "s3":
                self.output_path = pp.join(
                    self.data_config.path.output_path,
                    self.data_config.path.exp_name,
                    self.question_index,
                )
            else:
                self.output_path = pp.join(
                    self.data_config.path.output_path_s3,
                    self.data_config.path.exp_name,
                    self.question_index,
                )
            if not self._fs.exists(self.output_path):
                self._fs.makedirs(
                    self.output_path
                )  # create folder if not already present.
            # Init logger
            self.current_ts = (
                str(datetime.datetime.now(datetime.timezone.utc))
                .replace("+00:00", "Z")
                .replace(":", "_")
                .replace(" ", "_")
                .replace(".", "_")
            )

            if self.log_file_path is None:
                log_file = pp.join(self.output_path, f"runtime_{self.current_ts}.log")
                create_logger(
                    logger_name=self.MYLOGGERNAME,
                    level=self.logging_level,
                    log_file_path=log_file,
                    verbose=self.verbose,
                    fs=self._fs,
                )
            else:
                create_logger(
                    logger_name=self.MYLOGGERNAME,
                    level=self.logging_level,
                    log_file_path=self.log_file_path,
                    verbose=self.verbose,
                    fs=self._fs,
                )

            self.logger = logging.getLogger(self.MYLOGGERNAME)
            self.logger.info(
                f"The results will be saved in this output folder: {self.question_index} and output path: {self.output_path}"
            )
            if not check_for_similar_questions:
                self.logger.warning(
                    "Above path is present in the knowledge base excel but the folder doesnot exists. Created a new folder"
                )

        except Exception as e:
            self.logger.info(
                f"folder creation and log declaration ended with error - {e}."
            )
            if cloud_config.cloud_provider != "s3":
                self.output_path = pp.join(
                    self.data_config.path.output_path,
                    self.data_config.path.exp_name,
                    self.question_index,
                )
            else:
                self.output_path = pp.join(
                    self.data_config.path.output_path_s3,
                    self.data_config.path.exp_name,
                    self.question_index,
                )

            # Init logger
            if self.log_file_path is None:
                log_file = pp.join(self.output_path, "runtime.log")
                create_logger(
                    logger_name=self.MYLOGGERNAME,
                    level=self.logging_level,
                    log_file_path=log_file,
                    verbose=self.verbose,
                    fs=self._fs,
                )
            else:
                create_logger(
                    logger_name=self.MYLOGGERNAME,
                    level=self.logging_level,
                    log_file_path=self.log_file_path,
                    verbose=self.verbose,
                    fs=self._fs,
                )

            self.logger = logging.getLogger(self.MYLOGGERNAME)
            self.logger.info(
                f"The results will be saved in this output folder: {self.question_index} and output path: {self.output_path}"
            )
        log_uncaught_errors(self.logger)
        self.logger.info("Saving Data config")
        if not self._fs.exists(pp.join(self.output_path, "configs")):
            self._fs.makedirs(pp.join(self.output_path, "configs"))

        result_folders = [
            "01_text_to_query",
            "02_query_to_chart",
            "03_table_to_insights",
        ]
        if cloud_config.cloud_provider != "s3":
            self.prev_runtime_result_exists = any(
                [folder in self._fs.ls(self.output_path) for folder in result_folders]
            )
        else:
            self.prev_runtime_result_exists = any(
                [
                    folder in self._fs.listdir(self.output_path)
                    for folder in result_folders
                ]
            )

        # Save data Config file.
        with self._fs.open(
            pp.join(self.output_path, "configs", "data_config.yaml"), "w"
        ) as fout:
            yaml.dump(
                # json.loads(json.dumps(self.data_config)),
                self.data_config,
                fout,
                default_flow_style=False,
            )

        self.logger.info("Saving model config")

        # Save model Config file.
        with self._fs.open(
            pp.join(self.output_path, "configs", "model_config.yaml"), "w"
        ) as fout:
            yaml.dump(
                # json.loads(json.dumps(self.model_config)),
                self.model_config,
                fout,
                default_flow_style=False,
            )

        self.logger.info("Saving user config")
        # Save Config file.
        with self._fs.open(
            pp.join(self.output_path, "configs", "user_config.yaml"), "w"
        ) as fout:
            yaml.dump(
                # json.loads(json.dumps(self.user_config)),
                self.user_config,
                fout,
                default_flow_style=False,
            )

        self.logger.info("Saving debug config")
        # Save Config file.
        with self._fs.open(
            pp.join(self.output_path, "configs", "debug_config.yaml"), "w"
        ) as fout:
            yaml.dump(
                self.debug_config,
                fout,
                default_flow_style=False,
            )

        if self.data_config.path.question_classification:
            self.logger.info("Classification module enabled!.")
        else:
            self.logger.info(
                f"Classification disabled!.. default value --> {self.data_config.path.default_classification}"
            )
        # log_stage_end("Folder_creation_for_each_question", "Track 1")
        # self.current_ts = (
        #         str(datetime.datetime.now(datetime.timezone.utc))
        #         .replace("+00:00", "Z")
        #         .replace(":", "_")
        #         .replace(" ", "_")
        #         .replace(".", "_")
        #     )
        """

    def _individual_track_folder_creation(self, track: str = None):
        """Create individual subfolders for each track to store output files.

        Parameters
        ----------
        track : str, optional
            The name of the track for which the subfolder needs to be created.
            Valid values are '01_text_to_query', '02_query_to_chart', and '03_table_to_insights'.

        Returns
        -------
        None
            This function creates subfolders for each track and stores the paths in instance variables
            (text_to_query_path, query_to_chart_path, and table_to_insights_path).

        Raises
        ------
        ValueError
            If an invalid track value is provided.
        """
        try:
            # Create subfolder
            self.text_to_query_path = pp.join(self.output_path, "01_text_to_query")
            self.query_to_chart_path = pp.join(self.output_path, "02_query_to_chart")
            self.table_to_insights_path = pp.join(
                self.output_path, "03_table_to_insights"
            )

            if self.why_qn_flag:
                folders_to_create = self.table_to_insights_path
            else:
                if track == "01_text_to_query":
                    folders_to_create = self.text_to_query_path
                elif track == "02_query_to_chart":
                    folders_to_create = self.query_to_chart_path
                elif track == "03_table_to_insights":
                    folders_to_create = self.table_to_insights_path
                else:
                    raise ValueError(
                        "Invalid track provided. Valid values are '01_text_to_query', '02_query_to_chart', and '03_table_to_insights'."
                    )

            self.logger.debug(f"Folder - {folders_to_create} is created.")
            if not self._fs.exists(folders_to_create):
                self._fs.makedirs(folders_to_create)
        except ValueError as ve:
            self.logger.error(f"Caught ValueError: {ve}")


class DataLoader(Generic):
    """DataLoader is used to load the database and initialize data dictionary.

    Parameters:
    -----------
    user_config : dict
        A dictionary containing parameters for interacting with the OpenAI GPT-4 API to generate insights from data.
        This includes user interface settings, API call configuration, thresholds for "why" questions, and limits
        for table rows when the token limit exceeds.

    data_config : dict
        A dictionary containing paths to input data, databases, output folders, and data dictionaries.

    model_config : dict
        A dictionary containing model parameters such as engine, temperature, max_tokens, etc.,
        along with system roles, static prompts, and guidelines for all tracks.

    debug_config : dict
        A dictionary handling errors such as invalid columns, syntax errors, and queries like ALTER or DROP TABLE.

    Raises:
    -------
    ValueError:
        If any initialization or functionality is incorrect.

    """

    def __init__(
        self,
        user_config: dict,
        data_config: dict,
        model_config: dict,
        debug_config: dict,
    ) -> None:
        Generic.__init__(
            self,
            user_config,
            data_config,
            model_config,
            debug_config,
        )

    def _loading_s3_data_dictionary(self, response):
        data_dictionary_s3 = {}
        if "Contents" in response:
            data_dicts = [
                x["Key"] for x in response["Contents"][1:]
            ]  # Exclude the first entry if necessary
            for data_dict in data_dicts:
                table_name = data_dict.split("/")[-1].split(".")[0]
                self.logger.info(f"Loading data dictionary for table: {table_name}")

                try:
                    # Load data dictionary from S3
                    start = timeit.default_timer()
                    data_dictionary_s3[table_name] = load_data_dictionary_from_s3(
                        bucket=cloud_config.domain_storage.account_name,
                        path=data_dict,
                    )
                    end_time = timeit.default_timer()
                    print(f"seconds {round(end_time-start,2)}")
                except Exception as e:
                    self.logger.error(
                        f"Error loading data dictionary for table {table_name}: {str(e)}"
                    )
            return data_dictionary_s3
        else:
            self.logger.info("No objects found in the response.")

    @timing_decorator(track_app_start=False)
    def _db_and_dictionary_loading(self):
        """
        Load the database and initialize data dictionary.

        This function loads the database and initializes a data dictionary to store information about the columns and its description.

        Returns:
            None

        """
        # Load database
        # set_current_track("track1")

        # log_stage_start("DB_loading", "Pre_track")
        start_time_to_load_db = timeit.default_timer()
        # Instantiating the core clients for database and storage
        # reporting_db_config = config.database_details[config.domain_name][
        #     "reporting_db"
        # ]
        # historical_db_config = config.database_details[config.domain_name][
        #     "historical_db"
        # ]

        # #craete two type of connection variable for reporting and historical

        # logging.info("Establishing DB Connection")
        # self.db_factory_historical = DatabaseFactory(
        #     get_database_client(historical_db_config["domain_database"])
        # )
        # self.historical_db_conn = self.db_factory_historical.create_database_connection(
        #     historical_db_config, self._fs, False
        # )
        # logging.info("Connection Established - got connector")

        athena_tables_to_exclude = self.data_config.path.athena_exclude_table_names
        athena_tables = self.data_config.path.input_data_tables.historical
        athena_table_names = list(set(athena_tables) - set(athena_tables_to_exclude))

        # connection for mysql

        # logging.info("Establishing DB Connection")
        # self.db_factory_reporting = DatabaseFactory(
        #     get_database_client(reporting_db_config["domain_database"])
        # )
        # self.reporting_db_conn = self.db_factory_reporting.create_database_connection(
        #     reporting_db_config, self._fs, False
        # )
        # logging.info("Connection Established - got connector")

        mysql_tables_to_exclude = self.data_config.path.mysql_exclude_table_names
        mysql_tables = self.data_config.path.input_data_tables.reporting
        mysql_table_names = list(set(mysql_tables) - set(mysql_tables_to_exclude))

        end_time_to_load_db = timeit.default_timer()

        # log_stage_end("DB_loading", "Pre_track")

        # self.logger.info(
        #     f"Time taken to load mysql database: {round(end_time_to_load_db - start_time_to_load_db, 2)} seconds."
        # )

        # self.db_factory = self.db_factory_historical
        # self.conn = self.historical_db_conn
        # Initialize data dictionary
        self.data_dictionary = {}
        # self.data_dictionary_reporting = {}
        # self.data_dictionary_historical = {}
        self.reporting_db_data_dictionary = {}
        self.historical_db_data_dictionary = {}
        # IF LOCAL
        if cloud_config.cloud_provider != "s3":
            # IF TABLE SECLECTION
            if (
                self.user_config.table_selection_parameters.table_selection
                and self.user_config.table_selection_parameters.entity_flag
            ):
                for table_name in mysql_table_names:
                    self.reporting_db_data_dictionary[table_name] = (
                        load_data_dictionary(
                            pp.join(
                                self.data_config.path.entity_extraction_data_dictionary_path,
                                f"{table_name}.json",
                            ),
                            fs=self._fs,
                        )
                    )
                for table_name in athena_table_names:
                    self.historical_db_data_dictionary[table_name] = (
                        load_data_dictionary(
                            pp.join(
                                self.data_config.path.entity_extraction_data_dictionary_path_athena,
                                f"{table_name}.json",
                            ),
                            fs=self._fs,
                        )
                    )
            else:  # <<<<<<<<<<<<<<<<<only this has been implemented
                # Reporting Tables Data Dictionary
                for table_name in mysql_table_names:
                    self.reporting_db_data_dictionary[table_name] = (
                        load_data_dictionary(
                            pp.join(
                                self.data_config.path.data_dictionary_path,
                                f"{table_name}.json",
                            ),
                            fs=self._fs,
                        )
                    )

                # Historical Tables Data Dictionary
                for table_name in athena_table_names:
                    self.historical_db_data_dictionary[table_name] = (
                        load_data_dictionary(
                            pp.join(
                                self.data_config.path.data_dictionary_path_athena,
                                f"{table_name}.json",
                            ),
                            fs=self._fs,
                        )
                    )
        else:
            s3_client = boto3.client(cloud_config.cloud_provider)
            # data dictionary for mysql from s3
            if (
                self.user_config.table_selection_parameters.table_selection
                and self.user_config.table_selection_parameters.entity_flag
            ):
                self.logger.info("Loading data dictionary from S3")
                response = s3_client.list_objects(
                    Bucket=cloud_config.domain_storage.account_name,
                    Prefix=self.data_config.path.entity_extraction_data_dictionary_path_s3,
                )
            # IF TABLE SECLECTION
            else:
                response = s3_client.list_objects(
                    Bucket=cloud_config.domain_storage.account_name,
                    Prefix=self.data_config.path.data_dictionary_path_s3,
                )

            json_objects = [
                obj["Key"]
                for obj in response.get("Contents", [])
                if obj["Key"].endswith(".json")
            ]

            def read_json_file(file_key):
                response = s3_client.get_object(
                    Bucket=cloud_config.domain_storage.account_name, Key=file_key
                )
                file_content = response["Body"].read().decode("utf-8")
                json_content = json.loads(file_content)
                file_name = file_key.split("/")[-1].split(".")[0]
                return file_name, json_content

            with ThreadPoolExecutor(max_workers=5) as executor:
                results = executor.map(read_json_file, json_objects)

            for file_name, json_content in results:
                self.reporting_db_data_dictionary[file_name] = json_content
            # self.reporting_db_data_dictionary = self._loading_s3_data_dictionary(
            #     response
            # )

            # data disctionary for athena from s3
            if (
                self.user_config.table_selection_parameters.table_selection
                and self.user_config.table_selection_parameters.entity_flag
            ):
                response = s3_client.list_objects(
                    Bucket=cloud_config.domain_storage.account_name,
                    Prefix=self.data_config.path.entity_extraction_data_dictionary_path_s3_athena,
                )
            else:
                response = s3_client.list_objects(
                    Bucket=cloud_config.domain_storage.account_name,
                    Prefix=self.data_config.path.data_dictionary_path_s3_athena,
                )
            json_objects = [
                obj["Key"]
                for obj in response.get("Contents", [])
                if obj["Key"].endswith(".json")
            ]

            def read_json_file(file_key):
                response = s3_client.get_object(
                    Bucket=cloud_config.domain_storage.account_name, Key=file_key
                )
                file_content = response["Body"].read().decode("utf-8")
                json_content = json.loads(file_content)
                file_name = file_key.split("/")[-1].split(".")[0]
                return file_name, json_content

            with ThreadPoolExecutor(max_workers=5) as executor:
                results = executor.map(read_json_file, json_objects)

            for file_name, json_content in results:
                self.historical_db_data_dictionary[file_name] = json_content

            # self.historical_db_data_dictionary = self._loading_s3_data_dictionary(
            #     response
            # )

        # self.data_dictionary_reporting = self._filter_dict(
        #     self.reporting_db_data_dictionary
        # )

        # self.data_dictionary_historical = self._filter_dict(
        #     self.historical_db_data_dictionary
        # )
        # self.data_dictionary = self.data_dictionary_historical
        return None


class DataProcessor(FolderManager):
    """DataProcessor class is used to check whether the question is a reason based(why question) or not. we also save the question to the output folder.

    Parameters:
    -----------
    user_config : dict
        A dictionary containing parameters for interacting with the OpenAI GPT-4 API to generate insights from data.
        This includes user interface settings, API call configuration, thresholds for "why" questions, and limits
        for table rows when the token limit exceeds.

    data_config : dict
        A dictionary containing paths to input data, databases, output folders, and data dictionaries.

    model_config : dict
        A dictionary containing model parameters such as engine, temperature, max_tokens, etc.,
        along with system roles, static prompts, and guidelines for all tracks.

    debug_config : dict
        A dictionary handling errors such as invalid columns, syntax errors, and queries like ALTER or DROP TABLE.

    question : str, optional
        Business user query, by default None

    additional_context : str, optional
        Additional context to answer the question, by default None

    existing_question : bool
        flag represents if the user question is an exact match of existing questions in KB.

    output_path : str
        output folder path
    Raises:
    -------
    ValueError:
        If any initialization or functionality is incorrect.

    """

    def __init__(
        self,
        user_config: dict,
        data_config: dict,
        model_config: dict,
        debug_config: dict,
        existing_question: bool = False,
        output_path: str = None,
        question: str = None,
    ):
        FolderManager.__init__(
            self, user_config, data_config, model_config, debug_config, question
        )
        self.existing_question_ = existing_question
        self.output_path = output_path
        self.question = question

    @timing_decorator(track_app_start=False)
    def _preprocess(self, question: str = None, additional_context: str = None) -> None:
        """In this funciton, we check whether the question is a reason based(why question) or not.
           we also save the question to the output folder.

        Parameters
        ----------
        question : str, optional
            Question (may or may not contain bot history) to pass to the GPT. Can be the entire conversation either:
        additional_context : str, optional
            Additional context provided to answer the user question, by default None

        Raises
        ------
        ValueError
            When code is set to UI mode and user query is not provided as a parameter
        """
        # set_current_track("track1")
        # log_stage_start("Pre_processing_user_question", "Track 1")
        if question is None and not self.user_config.ui:
            first_msg = "User query is not given as input and main.py is called in UI mode, Hence question and additional context from config file will be used."
            self.question = self.user_config.user_inputs.question
            self.additional_context = self.user_config.user_inputs.additional_context
        if question is None and self.user_config.ui:
            raise ValueError("Code is set to UI mode, so user query is mandatory.")
        else:
            first_msg = "User query and/or additional context is given as input."
            self.question = question
            self.additional_context = additional_context

        self.all_tokens = []  # optional

        # Creating the multiple charts flag based on the user question
        multiple_charts_indicator = "multiple charts:"
        self.multiple_charts = False
        if multiple_charts_indicator in self.question.lower():
            self.multiple_charts = True
            # Removing the indicator tag from the question for GPT calls.
            self.question = re.sub(
                multiple_charts_indicator, "", self.question, flags=re.IGNORECASE
            )

        # Check if its why qn or not.
        if not bool(self.question_threshold):
            # why_qn_flag will be False when the self.question_threshold is: None, 0, "", False.
            # `if (self.question_threshold is None) or (self.question_threshold.strip() == ""):` could be used but it would've failed when self.question_threshold is float
            self.why_qn_flag = False
        else:
            self.question_threshold = float(self.question_threshold)
            # TODO: Handle edge cases where user question itself contains ';'
            reason_based_questions = self.classifier.find_reason_based_questions(
                [question.split(";")[-1].strip()], self.question_threshold
            )
            for qn, _ in reason_based_questions:
                if qn == question:
                    self.why_qn_flag = True
                else:
                    self.why_qn_flag = False

        # create the output folder structure for saving logs.
        # TODO: Potential place to fix issue#26
        # self.exp_folder = pp.join(
        #     self.data_config.path.output_path, self.data_config.path.exp_name
        # )
        # self._fs.makedirs(self.exp_folder, exist_ok=True)  # create folder if not already present.
        # self.logger.info(
        #     f"Question index generated for this question is {self.question_index}."
        # )
        # self.logger.debug(
        #     f"Existing question: {self.existing_question}, Question index: {self.question_index}, Existing status: {self.existing_status}"
        # )

        if (
            self.existing_question
            and self.user_config.skip_api_call
            and self.user_config.skip_list
        ):
            skip_list = self.user_config.skip_list
            skip_list = [re.sub(r"[^\w\s]", "", q).lower().strip() for q in skip_list]
            if re.sub(r"[^\w\s]", "", self.question).lower().strip() in skip_list:
                self.logger.info("Question part of Skip List.")
                self.existing_question = True
            else:
                self.logger.info("Question not a part of Skip List.")
                self.existing_question = False
        elif not self.user_config.skip_api_call:
            self.existing_question = False

        # self.logger.info(first_msg)

        if self.additional_context is None:
            if hasattr(self, "bot_history") and (self.bot_history is not None):
                history = self.bot_history
                file_content = " ; ".join([q for [q, a] in history])
            else:
                file_content = f"Question: {self.question}"
            if cloud_config.cloud_provider != "s3":
                with self._fs.open(
                    pp.join(self.output_path, "question.txt"), "w"
                ) as file:
                    file.write(file_content)
        else:
            if cloud_config.cloud_provider != "s3":
                file_content = f"Question: {self.question}\nAdditional context: {self.additional_context}"
                with self._fs.open(
                    pp.join(self.output_path, "question.txt"), "w"
                ) as file:
                    file.write(file_content)

        # log_stage_end("Pre_processing_user_question", "Track 1")


class ConfigValidation(Generic):
    """ConfigValidation is used to Perform configuration validation and initialize response_json.

    Parameters:
    -----------
    user_config : dict
        A dictionary containing parameters for interacting with the OpenAI GPT-4 API to generate insights from data.
        This includes user interface settings, API call configuration, thresholds for "why" questions, and limits
        for table rows when the token limit exceeds.

    data_config : dict
        A dictionary containing paths to input data, databases, output folders, and data dictionaries.

    model_config : dict
        A dictionary containing model parameters such as engine, temperature, max_tokens, etc.,
        along with system roles, static prompts, and guidelines for all tracks.

    debug_config : dict
        A dictionary handling errors such as invalid columns, syntax errors, and queries like ALTER or DROP TABLE.

    Raises:
    -------
    ValueError:
        If any initialization or functionality is incorrect.

    """

    def __init__(
        self,
        user_config: dict,
        data_config: dict,
        model_config: dict,
        debug_config: dict,
    ):
        Generic.__init__(
            self,
            user_config,
            data_config,
            model_config,
            debug_config,
        )

    @timing_decorator(track_app_start=False)
    def _config_validation_calling_and_response_json_initialization(self):
        """
        Perform configuration validation and initialize response_json.

        This function initializes the response_json dictionary and then proceeds to perform
        configuration validation using a list of validator classes and their respective arguments.
        If the validation fails, a ValueError is raised indicating the failure.

        Returns:
            None

        Raises:
            ValueError: If configuration validation fails for any validator.

        """
        # set_current_track("track1")
        # log_stage_start("config_validation_calling_and_response_json_initialization", "Pre_track")
        # Response variable initialized for creation of response.json
        self.response_json = {}
        self.response_track2 = {}
        self.response_track3 = {}

        validators = [
            # (DataConfigValidator, [self.data_config, self.fs_key]),
            (UserConfigValidator, [self.user_config]),
            (ModelConfigValidator, [self.model_config]),
        ]

        DataConfigValidator = "Abcd"
        for validator_cls, args in validators:
            config = args[0]
            if validator_cls == DataConfigValidator:
                fs_key = args[1]
                validator = validator_cls(config=config, fs_key=fs_key)
            else:
                validator = validator_cls(config=config)
            result = validator.validate_config()
            if not result:
                raise ValueError(
                    f"Config validation failed for {validator} from {config}. Result: {result}"
                )
        # log_stage_end("config_validation_calling_and_response_json_initialization", "Pre_track")
        return None


class SkipFlag(FolderManager):
    """SkipFlag class is used to check for skipping the mode

    Parameters:
    -----------
    user_config : dict
        A dictionary containing parameters for interacting with the OpenAI GPT-4 API to generate insights from data.
        This includes user interface settings, API call configuration, thresholds for "why" questions, and limits
        for table rows when the token limit exceeds.

    data_config : dict
        A dictionary containing paths to input data, databases, output folders, and data dictionaries.

    model_config : dict
        A dictionary containing model parameters such as engine, temperature, max_tokens, etc.,
        along with system roles, static prompts, and guidelines for all tracks.

    debug_config : dict
        A dictionary handling errors such as invalid columns, syntax errors, and queries like ALTER or DROP TABLE.

    question : str, optional
        Business user query, by default None

    additional_context : str, optional
        Additional context to answer the question, by default None

    existing_question : bool
        flag represents if the user question is an exact match of existing questions in KB.

    prev_folder_path : str
        previous folder path

    existing_status : list
        existing_status represents the status of Track 1, 2 and 3 from the previous run.

    prev_runtime_result_exists : bool
        flag represents any of the tracks folders are present from previous run

    bot_history : str
        previous conversation

    Raises:
    -------
    ValueError:
        If any initialization or functionality is incorrect.

    """

    def __init__(
        self,
        user_config: dict,
        data_config: dict,
        model_config: dict,
        debug_config: dict,
        existing_question: bool = False,
        existing_status: list = None,
        prev_folder_path: str = None,
        prev_runtime_result_exists: bool = False,
        bot_history: str = None,
    ):
        FolderManager.__init__(
            self,
            user_config,
            data_config,
            model_config,
            debug_config,
        )

        self.skip_track1_followup = False
        self.skip_track2_followup = False
        self.skip_track3_followup = False

        self.is_followup = False

        self.existing_question = existing_question
        self.existing_status = existing_status
        self.prev_folder_path = prev_folder_path
        self.prev_runtime_result_exists = prev_runtime_result_exists
        self.bot_history = bot_history
        self.model_tokens = None

    def _skip_model_kb(self):
        """function for checking whether the question is present in knowledge base"""

        self.skip_model_kb = False
        if self.existing_question and self.existing_status[0] == "success":
            self.skip_model_kb = True

    @timing_decorator(track_app_start=False)
    def _followup(self):
        # it should not be considered a followup question when:
        #   1. the result of previous question (of same conversation) does not exist
        #   2. when it is a KB run
        # set_current_track("track1")
        # log_stage_start("checking_for_followup_question", "Track 1")
        if (
            (self.prev_folder_path is not None)
            and (not self._fs.exists(self.prev_folder_path))
        ) or (self.existing_question and self.existing_status[0] == "success"):
            self.is_followup = False
            self.logger.info("Question is not considered as followup")
        else:
            # it is a followup question when:
            #   1. (len bot_history > 1) & (prev runtime result does NOT exist)
            #   2. (len bot_history > 1) & (prev runtime result exist) & (not a KB run)
            if (
                hasattr(self, "bot_history")
                and (self.bot_history is not None)
                and (len(self.bot_history) > 1)
            ):
                if (not self.prev_runtime_result_exists) or (
                    self.prev_runtime_result_exists and self.existing_question
                ):
                    self.is_followup = True

        if self.is_followup:
            # if this is a followup question, then copy the results of previous question's folder (of the same conversation) to the new folder
            self.logger.info(
                f"Question is considered as a followup question. copying the results from '{self.prev_folder_path}' to '{self.output_path}'"
            )
            if cloud_config.cloud_provider != "s3":
                copy_folders(
                    source_path=self.prev_folder_path,
                    source_folders=[
                        "01_text_to_query",
                        "02_query_to_chart",
                        "03_table_to_insights",
                    ],
                    source_files=[],
                    destination_path=self.output_path,
                    fs=self._fs,
                )
            else:
                copy_folder_s3(
                    source_path=self.prev_folder_path,
                    source_folders=[
                        "01_text_to_query",
                        "02_query_to_chart",
                        "03_table_to_insights",
                    ],
                    source_files=[],
                    destination_path=self.output_path,
                    fs=self._fs,
                )

            # if it is a Followup question:
            #   classify the latest user question as SQL related / Graph related / insights related / general followup / Greeting etc ..
            #   Example if followup is insight related, then SQL output and Graph output can be read from the previous question's folder
            #   (copied to the current question's folder in the above step)
            #   So, if follow_up question is related to track 3, we use track 1 and track 2 output as it is.

            # Error handling:
            #   If follow-up is track 1 -> we'll start from track 1 and copy won't impact anything
            #   Example, If follow-up is track 3, and track 1 is a success -> it will use copied SQL
            #   If follow-up is track 3, and track 1 is error -> It will try and error out copied SQL again. Use track 2 code also from copied folder. Call API for track 3

            follow_up_question = self.bot_history[-1][0]
            follow_up_ins = followup_q_tag(
                self.user_config, self.model_config, follow_up_question
            )
            (
                self.skip_track1_followup,
                self.skip_track2_followup,
                self.skip_track3_followup,
            ) = follow_up_ins.process_followup_question()
            self.model_tokens = follow_up_ins.model_tokens
        # log_stage_end("checking_for_followup_question", "Track 1")


class FindUnits:
    """ConfigValidation is used to Perform configuration validation and initialize response_json.

    Parameters:
    -----------
    user_config : dict
        A dictionary containing parameters for interacting with the OpenAI GPT-4 API to generate insights from data.
        This includes user interface settings, API call configuration, thresholds for "why" questions, and limits
        for table rows when the token limit exceeds.

    data_config : dict
        A dictionary containing paths to input data, databases, output folders, and data dictionaries.

    model_config : dict
        A dictionary containing model parameters such as engine, temperature, max_tokens, etc.,
        along with system roles, static prompts, and guidelines for all tracks.

    debug_config : dict
        A dictionary handling errors such as invalid columns, syntax errors, and queries like ALTER or DROP TABLE.

    Raises:
    -------
    ValueError:
        If any initialization or functionality is incorrect.

    """

    def __init__(self, track1_output_table_dict, units_to_skip):
        self.track1_output_table_dict = track1_output_table_dict
        self.units_to_skip = units_to_skip
        self.logger = logging.getLogger("QueryInsights")

    def _find_units_of_measurement(self) -> str:
        """Helper method to extract units of measurement from Track 1 output data dict.

        Returns
        -------
        str
            units of measurement
        """
        if (not hasattr(self, "track1_output_table_dict")) | (
            self.track1_output_table_dict is None
        ):
            self.logger.error(
                "Track 1 is not called or errored out. Thus there is no units."
            )
            unit = ""
        else:
            data_dict = self.track1_output_table_dict
            if len(data_dict) != 1:
                self.logger.info(
                    "Track 1 output is not scalar, thus we cannot use units of measurement."
                )
                unit = ""
            else:
                if "unit_of_measurement" not in list(data_dict[0].keys()):
                    unit = ""
                    self.logger.info(
                        "The data dictionary doesn't have units of measurement"
                    )
                else:
                    unit = data_dict[0]["unit_of_measurement"]
                    if unit is not None:
                        if unit.lower() in self.units_to_skip:
                            unit = ""
                    else:
                        unit = ""
                    self.logger.info(f"The units of measurement is {unit}.")

        return unit


class KnowledgeBase(FindUnits):
    """This class is used to creating or updating knowledge base

    Raises:
    -------
    ValueError:
        If any initialization or functionality is incorrect.

    """

    def __init__(self):
        pass

    def update_knowledgebase(
        self,
        tracks_flag,
        generic_initializations,
        foldercreation,
        dataloader,
        config_validation,
        dataprocessor,
        skip_flag,
        alltracks_status,
        all_tokens,
        feedback,
        track1_ins,
    ) -> None:
        """
        Creates the Knowledge base excel if it's not already available with the results.
        If it's available, it will add another row to the existing excel with the results.
        The new/updated row will include the below details -
        1. Question index
        2. Question/Context
        3. Results Status
        4. Results path

        Parameters
        ----------
        alltracks_status : list
            contains list all 3 tracks status. Example ['success', 'skip', 'success']
        feedback : list
            feedback from UI

        Returns
        -------
        None
        """
        try:
            # update the knowledge base
            self.logger = generic_initializations.logger
            self.units_to_skip = generic_initializations.units_to_skip
            generic_initializations.logger.info("Updating the Knowledge base...")
            alltracks_status_str = (
                "[" + ", ".join(st for st in alltracks_status if st is not None) + "]"
            )
            self.logger.info("All tokens: %s", all_tokens)
            if len(all_tokens) > 0:
                all_tokens_str = ", ".join([str(d) for d in all_tokens])
            else:
                all_tokens_str = ""

            empty_chart_flag = 0
            if hasattr(self, "all_zeros"):
                if all(self.all_zeros):
                    empty_chart_flag = 1

            total_tokens_str = ""
            if len(all_tokens) > 0:
                total_tokens = {
                    "Total": {
                        "completion_tokens": 0,
                        "prompt_tokens": 0,
                        "total_tokens": 0,
                    }
                }

                for item in all_tokens:
                    for key, values in item.items():
                        for k, v in values.items():
                            total_tokens["Total"][k] += v

                total_tokens_str = str(total_tokens["Total"])

            if foldercreation.similarity[0]:
                similar_question = foldercreation.similarity[1]["question"]
                similarity_score = foldercreation.similarity[1]["score"]
            else:
                similar_question, similarity_score = None, None

            col_list = [
                "q_index",
                "question",
                "sql_generator",
                "chart_generator",
                "insights_generator",
                "additional_context",
                "results_status",
                "results_path",
                "feedback",
                "similarity_score",
                "similar_question",
                "lastmodifiedtime",
                "unit_of_measurement",
                "empty_chart_flag",
                "token_information",
                "total_tokens",
            ]

            question = foldercreation.question
            if hasattr(self, "bot_history") and (self.bot_history is not None):
                history = self.bot_history
                question = " ; ".join([q for [q, a] in history])
            self.track1_output_table_dict = track1_ins.track1_output_table_dict
            unit = self._find_units_of_measurement()
            val_list = [
                foldercreation.question_index,
                foldercreation.question,
                tracks_flag[0],
                tracks_flag[1],
                tracks_flag[2],
                foldercreation.additional_context,
                alltracks_status_str,
                foldercreation.output_path,
                feedback,
                similarity_score,
                similar_question,
                foldercreation.current_ts,
                unit,
                empty_chart_flag,
                all_tokens_str,
                total_tokens_str,
            ]
            if generic_initializations._fs.exists(foldercreation.knowledge_base):
                kb_df = read_data(
                    foldercreation.knowledge_base, fs=generic_initializations._fs
                )
                kb_df.rename(columns={"index": "q_index"}, inplace=True)
                kb_q = kb_df[kb_df["q_index"] == foldercreation.question_index]
                if (
                    kb_q.shape[0] == 1
                    and kb_q["sql_generator"].values
                    and not (kb_q["chart_generator"].values)
                ) or (
                    kb_q.shape[0] == 1
                    and kb_q["sql_generator"].values
                    and kb_q["chart_generator"].values
                    and not (kb_q["insights_generator"].values)
                ):
                    self.logger.info("---------------Knowledge Base---------------")
                    self.logger.info(
                        "all_tokens_str %s %d", all_tokens_str, len(all_tokens_str)
                    )
                    self.logger.info("Type: %s", type(all_tokens_str))
                    kb_json = eval(kb_q["token_information"].values[0])
                    self.logger.info("------------KB JSON--------------- %s", kb_json)
                    if len(all_tokens_str) > 0:
                        tokens_json = eval(all_tokens_str)
                        self.logger.info(
                            "------------Tokens JSON--------------- %s Type: %s",
                            tokens_json,
                            type(tokens_json),
                        )
                        if isinstance(tokens_json, tuple):
                            for j in tokens_json:
                                self.logger.info("Sub json: %s", j)
                                kb_json.update(j)
                        else:
                            kb_json.update(tokens_json)

                    tokens_str = str(kb_json)
                    kb_dict = json.loads(
                        kb_df[kb_df["q_index"] == foldercreation.question_index][
                            "total_tokens"
                        ]
                        .values[0]
                        .replace("'", '"')
                    )
                    merged_tokens = {}
                    if len(all_tokens) > 0:
                        merged_tokens = {
                            key: kb_dict[key] + total_tokens["Total"][key]
                            for key in kb_dict
                        }

                    merged_tokens_str = str(merged_tokens)
                    val_list = [
                        foldercreation.question_index,
                        foldercreation.question,
                        tracks_flag[0],
                        tracks_flag[1],
                        tracks_flag[2],
                        foldercreation.additional_context,
                        alltracks_status_str,
                        foldercreation.output_path,
                        feedback,
                        similarity_score,
                        similar_question,
                        foldercreation.current_ts,
                        unit,
                        empty_chart_flag,
                        tokens_str,
                        merged_tokens_str,
                    ]
                    kb_df = kb_df[~(kb_df["q_index"] == foldercreation.question_index)]
                    update_data = pd.DataFrame(
                        dict(zip(col_list, val_list), index=[kb_df.index.max()])
                    )
                else:
                    self.logger.info(
                        "Deleting existing row(s) with same index to avoid duplicates."
                    )

                    kb_df = kb_df[~(kb_df["q_index"] == foldercreation.question_index)]

                    update_data = pd.DataFrame(
                        dict(zip(col_list, val_list)), index=[kb_df.index.max() + 1]
                    )
                kb_df = pd.concat([kb_df, update_data])
                kb_df.rename(columns={"q_index": "index"}, inplace=True)
                with generic_initializations._fs.open(
                    foldercreation.knowledge_base, mode="wb", newline=""
                ) as fp:
                    kb_df.to_excel(fp, index=False)
            else:
                update_data = dict(zip(col_list, val_list))
                kb_df = pd.DataFrame([update_data])
                kb_df.rename(columns={"q_index": "index"}, inplace=True)
                with generic_initializations._fs.open(
                    foldercreation.knowledge_base, mode="wb", newline=""
                ) as fp:
                    kb_df.to_excel(fp, index=False)

            generic_initializations.logger.info("Knowledge base is updated.")

        except Exception as e:
            generic_initializations.logger.error(
                f"Error in updating the knowledge base. Error :\n {e}"
            )
            generic_initializations.logger.error(traceback.format_exc())
