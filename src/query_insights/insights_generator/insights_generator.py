# import datetime
import logging
import os
import posixpath as pp
import timeit
import traceback
from ast import literal_eval
from typing import Union

import fsspec
import mlflow
import pandas as pd
from core.model.model_factory import ModelFactory
from core.utils.client_utils import get_model_type
from core.utils.read_config import cloud_config, model
from src.query_insights.utils.utils import read_data, write_data

from ..utils.post_processing import (
    append_user_query_track3,
    extract_code,
    run_insights_code,
)
from ..utils.time_logging import timing_decorator
from ..utils.utils import (
    capture_stdout_to_var,
    convert_df_to_csv_string,
    get_gpt_token_count,
    rate_limit_error_handler,
    save_results_as_json,
)

# log_stage_end, log_stage_start, set_current_track


# from ..benchmarking import MLflowManager


MYLOGGERNAME = "QueryInsights"


class GenerateInsights:
    """Generate Insights from Tabular data extracted from Track 1(or Text to Query). It consists of three steps.

    1. Business user query to generating additional related questions that gives us insights (which we are calling it as insight questions).
    2. Generating code to answer Insight questions and original user query.
    3. Using the code result to generate summary.

    Parameters
    ----------
    user_config : dict
        input user_config dictionary for storing and accessing user-specific configurations.
    data_config : dict
        input data_config dictionary contains the paths to the data.
    model_config : dict
        input model_config dictionary for storing and accessing model-related configurations.
    question : str
        Business user query
    dictionary : Union[list, dict]
        Data dictionary of the Track 1 data output.
    table : pd.DataFrame
        Track 1 data as a dataframe.
    skip_model : bool
        condition whether to skip the api call.
    language : str, optional
        Language to answer the question in, for example "english", "spanish", "german", by default "english"
    additional_context : str, optional
        sentence which provides additional context to the original question, by default None
    fs : fsspec.filesystem, optional
        Filesystem of the url, None will default to local file system, by default ``None``
    """

    def __init__(
        self,
        user_config: dict,
        data_config: dict,
        model_config: dict,
        question: str,
        dictionary: Union[list, dict],
        business_overview: str,
        bot_history: list,
        table: pd.DataFrame,
        output_path: str,
        sql_results_path: str,
        language: str = "english",
        additional_context: str = None,
        skip_model: bool = False,
        fs=None,
    ) -> None:
        """Class constructor"""
        self.user_config = user_config
        self.data_config = data_config
        self.model_config = model_config
        self.question = question
        self.dictionary = dictionary
        self.business_overview = business_overview
        self.bot_history = bot_history
        self.table = table
        self.additional_context = additional_context
        self.output_path = output_path
        self.track_3c_new_summary = False
        self.track_3a_runtime = 0
        self.track_3b_runtime = 0
        self.track_3c_runtime = 0
        self.query_to_qns = None
        self.qns_to_code = None
        self.question_suggestion = None

        # If language is None or empty string, default to "english" language
        if language is None or not bool(language.strip()):
            language = "english"
        language = language.lower().title()

        self.language = language

        self.connection_param_dict = user_config.connection_params
        self.logger = logging.getLogger(MYLOGGERNAME)
        self.skip_model = skip_model
        self.error_message = None

        self._fs = fs or fsspec.filesystem("file")

        # Required for decorator
        time_delay = self.user_config.time_delay
        max_retries_3a = (
            self.model_config.table_to_insight_questions.model_params.max_tries
        )
        max_retries_3b = (
            self.model_config.insight_questions_to_code.model_params.max_tries
        )
        max_retries_3c = self.model_config.summarize_insights.model_params.max_tries
        max_retries_3c2 = self.model_config.summarize_tables.model_params.max_tries

        # Normal way of using decorator as we are getting trouble passing arguments
        # in intended way of "@rate_limit_error_handler(...)"
        self._table_to_insight_questions = rate_limit_error_handler(
            logger=self.logger, time_delay=time_delay, max_retries=max_retries_3a
        )(self._table_to_insight_questions)
        self._insight_questions_to_code = rate_limit_error_handler(
            logger=self.logger, time_delay=time_delay, max_retries=max_retries_3b
        )(self._insight_questions_to_code)
        self._get_summary = rate_limit_error_handler(
            logger=self.logger, time_delay=time_delay, max_retries=max_retries_3c
        )(self._get_summary)
        self._get_new_summary = rate_limit_error_handler(
            logger=self.logger, time_delay=time_delay, max_retries=max_retries_3c2
        )(self._get_new_summary)

        # Creating a folder for saving additional charts
        self.additional_charts_file_path = pp.join(
            self.output_path, "additional_charts"
        )
        if not self._fs.exists(self.additional_charts_file_path):
            self._fs.makedirs(self.additional_charts_file_path)

        # Creating a folder for saving code blocks
        self.code_blocks_file_path = pp.join(self.output_path, "code_blocks")
        if not self._fs.exists(self.code_blocks_file_path):
            self._fs.makedirs(self.code_blocks_file_path)

        self.sql_results_path = pp.join(sql_results_path, "output_table.csv")

        # mlflow parameters

        # self.experiment_name_mlflow = self.data_config.experiment_params.experiment_name
        # self.mlflow_manager = MLflowManager(experiment_name=self.experiment_name_mlflow)

        return

    def __repr__(self):
        try:
            full_conversation = ""
            # Track 3a
            full_conversation += "Track 3a:-\n\n"
            full_conversation += "prompt:-\n"
            full_conversation += str(self.model_factory.model_type.prompt)
            full_conversation += "\n"
            full_conversation += "response:-\n"
            full_conversation += (
                str(self.question_suggestion)
                + "\n"
                + str(self.question_finish)
                + "\n"
                + str(self.question_tokens)
                + "\n"
            )
            full_conversation += "-" * 100
            full_conversation += "\n\n"

            # Track 3b
            full_conversation += "Track 3b:-\n\n"
            full_conversation += "prompt:-\n"
            full_conversation += str(self.model_factory.model_type.prompt)
            full_conversation += "\n"
            full_conversation += "response:-\n"
            full_conversation += (
                str(self.code_suggestion)
                + "\n"
                + str(self.code_finish)
                + "\n"
                + str(self.code_tokens)
                + "\n"
            )
            full_conversation += "code result:-\n"
            full_conversation += (
                str(self.code_result)
                + "\nCode Error Output"
                + str(self.code_err_output)
                + "\n"
            )
            full_conversation += "-" * 100
            full_conversation += "\n\n"

            # Track 3c
            full_conversation += "Track 3c:-\n\n"
            if self.model_factory.model_type is not None:
                full_conversation += "prompt:-\n"
                full_conversation += str(self.model_factory.model_type.prompt)
                full_conversation += "\n"
                full_conversation += "response:-\n"

            full_conversation += (
                str(self.summary_suggestion)
                + "\n"
                + str(self.summary_finish)
                + "\n"
                + str(self.summary_tokens)
                + "\n"
            )
            return full_conversation
        # If any of the attributes are not found, return the error message
        except Exception as e:
            self.logger.info("Error - ", e)
            self.logger.info(
                "It is likely that get_insights method is not called. Call it first and rerun the print statement."
            )

    @timing_decorator(track_app_start=False)
    def _table_to_insight_questions(self, save_folder: str = None) -> None:
        """Business user query to generating additional related questions that gives us insights (which we are calling it as insight questions). It is otherwise known as Track 3a.

        Parameters
        ----------
        save_folder : str
            Path where all intermediate input and outputs will be saved.
        """
        # Load Configuration for Track 3a.
        # set_current_track("track3")
        # log_stage_start("Track 3a", "Track 3")
        prompt_dict = self.model_config.table_to_insight_questions.prompts
        model_param_dict = self.model_config.table_to_insight_questions.model_params
        model_param_dict["history"] = self.bot_history
        self.track = "table_to_insight_questions"

        # Initialize for the API call to GPT

        config = dict()
        config["llm_model_type"] = model
        # Get the model client
        model_client = get_model_type(
            config,
            prompt_dict,
            self.question,
            self.additional_context,
            self.connection_param_dict,
            self.track,
            self.user_config,
            self.language,
            self.dictionary,
            self.business_overview,
        )
        # Create a model factory
        self.model_factory = ModelFactory(model_client)
        self.logger.debug("Saving prompt for Track 3a")
        if save_folder is not None:
            if cloud_config.cloud_provider != "s3":
                with self._fs.open(pp.join(save_folder, "prompt.txt"), "w") as f:
                    f.writelines("Track 3a\n")
                    f.writelines(self.query_to_qns.prompt)
            else:
                if self.model_factory.model_type is not None:
                    prompt_3a = "Track 3a\n" + str(self.model_factory.model_type.prompt)
                    write_data(file_path=f"{save_folder}/prompt.txt", content=prompt_3a)
                else:
                    write_data(file_path=f"{save_folder}/prompt.txt", content="")

        # Make the API call to GPT
        (
            self.question_suggestion,
            self.question_finish,
            self.question_tokens,
            self.error_message,
        ) = self.model_factory.model_response(model_param_dict)

        # Saving tokens and token information
        self.logger.info(
            f"Track 3a:-\n finish token - {self.question_finish},\n token information - {self.question_tokens}"
        )

        # Saving each individual responses of Track 3 subsections as txt and json
        self.logger.debug("Saving response for Track 3a")
        if save_folder is not None:
            if cloud_config.cloud_provider != "s3":
                with self._fs.open(
                    pp.join(save_folder, "track3_responses.txt"), "w"
                ) as f:
                    f.writelines("Track 3a\n")
                    f.writelines(self.question_suggestion)
                    f.write("\n" + "-" * 100 + "\n")
            else:
                question_suggestion_3a = (
                    "Track 3a\n" + self.question_suggestion + "\n" + "-" * 100 + "\n"
                )
                write_data(
                    file_path=f"{save_folder}/track3_responses.txt",
                    content=question_suggestion_3a,
                )
        # log_stage_end("Track 3a", "Track 3")
        return

    @timing_decorator(track_app_start=False)
    def _insight_questions_to_code(self, save_folder: str = None) -> None:
        """Generating code to answer Insight questions and original user query. It is otherwise known as Track 3a.

        Parameters
        ----------
        save_folder : str
            Path where all intermediate input and outputs will be saved.
        """
        # Append business user query to GPT generated questions.
        # set_current_track("track3")
        # log_stage_start("Track 3b", "Track 3")
        all_questions = append_user_query_track3(
            other_questions=self.question_suggestion, user_query=self.question
        )
        # Load Configuration for Track 3b
        prompt_dict = self.model_config.insight_questions_to_code.prompts
        model_param_dict = self.model_config.insight_questions_to_code.model_params
        model_param_dict["history"] = self.bot_history
        self.track = "insight_questions_to_code"

        config = dict()
        # Initialize for the API call to GPT
        config["llm_model_type"] = model
        # Get the model client
        model_client = get_model_type(
            config,
            prompt_dict,
            all_questions,
            self.additional_context,
            self.connection_param_dict,
            self.track,
            self.user_config,
            self.language,
            self.dictionary,
            self.business_overview,
        )
        # Create a model factory
        self.model_factory = ModelFactory(model_client)
        try:
            if cloud_config.cloud_provider != "s3":
                if self._fs.exists(pp.join(save_folder, "prompt.txt")):
                    with self._fs.open(pp.join(save_folder, "prompt.txt"), "r") as f:
                        _prompt = f.read()
            else:
                _prompt = read_data(
                    f"{save_folder}/prompt.txt",
                    cloud_config.domain_storage.account_name,
                )

        except Exception as e:
            print(e)
            _prompt = ""
        if cloud_config.cloud_provider != "s3":
            with self._fs.open(pp.join(save_folder, "prompt.txt"), "w") as f:
                f.write(_prompt)
                f.writelines("---------------------------")
                f.writelines("\n\nTrack 3b\n")
                f.writelines(self.qns_to_code.prompt)
        else:
            prompt_3b = (
                _prompt
                + "\n-------------------------\n"
                + "\n\nTrack 3b\n"
                + str(self.model_factory.model_type.prompt)
            )
            write_data(
                file_path=f"{save_folder}/prompt.txt",
                content=prompt_3b,
            )

        # Make the API call to GPT
        (
            self.code_suggestion,
            self.code_finish,
            self.code_tokens,
            self.error_message,
        ) = self.model_factory.model_response(model_param_dict)

        # Saving tokens and token information
        self.logger.info(
            f"Track 3b:-\n finish token - {self.code_finish},\n token information - {self.code_tokens}"
        )
        # Saving each individual responses of Track 3 subsections as txt and json
        self._process_code_suggestion(save_folder)
        # log_stage_end("Track 3b", "Track 3")
        return

    def _process_code_suggestion(self, save_folder: str = None) -> None:
        """
        Executes the code blocks iteratively and capture the results in stdout as str var and generated the code_result that will be passed to track 3

        Parameters
        ----------
        save_folder : str, optional
            folder path the save the track3 response, by default None
        """
        # Post-processing
        # Trim the code to be executed.
        self.logger.info("Post processing of Track 3b started.")
        self.track3_trimmed_code = extract_code(
            string_input=self.code_suggestion,
            start=["```", "```python", "<start>", "```\n```"],
            end=["```", "<end>", "```\n```"],
            extract="first",
        )

        # Execute the code blocks iteratively and capture the results in stdout as str var - code_result
        # stderr as str var - code_err_output
        # fs_key is used only when fs_connection_dict.platform is not None
        self.code_result, self.code_err_output = capture_stdout_to_var(
            func=run_insights_code,
            kwargs={
                "full_code_str": self.track3_trimmed_code,
                "input_file_path": self.sql_results_path,
                "track3_path": self.output_path,
                "fs": self._fs,
                "fs_connection_dict": cloud_config,
                "fs_key": os.getenv("BLOB_ACCOUNT_KEY"),
            },
        )
        self.logger.info("Post processing of Track 3b completed.")
        # Saving each individual responses of Track 3 subsections as txt and json
        if save_folder is not None:
            self.logger.debug("Saving code result for Track 3b")
            # saving the prompt
            try:
                if cloud_config.cloud_provider != "s3":
                    if self._fs.exists(pp.join(save_folder, "track3_responses.txt")):
                        with self._fs.open(
                            pp.join(save_folder, "track3_responses.txt"), "r"
                        ) as f:
                            _track3b_responses_txt = f.read()
                else:
                    _track3b_responses_txt = read_data(
                        f"{save_folder}/track3_responses.txt",
                        cloud_config.domain_storage.account_name,
                    )
            except Exception as e:
                print(e)
                _track3b_responses_txt = ""  # saving the prompt
            if cloud_config.cloud_provider != "s3":
                with self._fs.open(
                    pp.join(save_folder, "track3_responses.txt"), "w"
                ) as f:
                    f.write(_track3b_responses_txt)
                    f.writelines("\nTrack 3b code:\n")
                    f.writelines(self.code_suggestion)
                    f.writelines("\nTrack 3b code result:\n")
                    f.writelines(self.code_result)
                    f.write("\n" + "-" * 100 + "\n")
            else:
                response_3b = (
                    _track3b_responses_txt
                    + "\nTrack 3b code:\n"
                    + self.code_suggestion
                    + "\nTrack 3b code result:\n"
                    + self.code_result
                    + "\n"
                    + "-" * 100
                    + "\n"
                )
                write_data(
                    file_path=f"{save_folder}/track3_responses.txt", content=response_3b
                )
            # Saving the error responses
            if cloud_config.cloud_provider != "s3":
                with self._fs.open(
                    pp.join(save_folder, "track3_error_responses.txt"), "w"
                ) as f:
                    f.writelines("Track 3b code error:\n")
                    f.writelines(self.code_err_output)
            else:
                erroe_3b = "\nTrack 3b code error:\n" + self.code_err_output
                write_data(f"{save_folder}/track3_error_responses.txt", erroe_3b)
        return

    @timing_decorator(track_app_start=False)
    def _get_summary(self, save_folder: str = None) -> None:
        """Using the code result to generate summary. If code yielded no result due to syntax error or a blank result, a descriptive summary of the table data will be used as summary.

        Parameters
        ----------
        save_folder : str
            Path where all intermediate input and outputs will be saved.
        """
        # Load Configuration for Track 3c
        # set_current_track("track3")
        # log_stage_start("Track 3c", "Track 3")
        self.track = "summarize_tables"
        if self.skip_model:
            if self._fs.exists(pp.join(save_folder, "track3_responses.txt")):
                self.logger.info("Retreiving track 3b code result from Knowledge Base.")
                with self._fs.open(
                    pp.join(save_folder, "track3_responses.txt"), "r"
                ) as f:
                    kb_code_result = f.read()
                # get the python code from the saved KB response
                self.code_suggestion = extract_code(
                    string_input=kb_code_result,
                    start=["Track 3b code:"],
                    end=["Track 3b code result:"],
                    extract="first",
                )
                if self.code_suggestion is not None:
                    # Execute the code blocks iteratively and capture the results in stdout as str var - code_result
                    self._process_code_suggestion(save_folder)
                else:
                    # if code_suggestion cannot be extracted re-run 3a, 3b, 3c
                    self.code_result = ""
            else:
                self.logger.error(
                    f"Retrieval from KB failed because {pp.join(save_folder, 'track3_responses.txt')} not found"
                )
                raise ValueError
        if self.code_result.strip() != "":
            self.logger.info(
                "Track 3b result was not blank, hence it's output will be used to summarize."
            )
            # Run the summary insights on code result
            prompt_dict = self.model_config.summarize_insights.prompts
            model_param_dict = self.model_config.summarize_insights.model_params
            model_param_dict["history"] = self.bot_history

            config = dict()
            config["llm_model_type"] = model
            # Get the model client
            model_client = get_model_type(
                config,
                prompt_dict,
                self.question,
                self.additional_context,
                self.connection_param_dict,
                self.track,
                self.user_config,
                self.language,
                self.dictionary,
                self.business_overview,
                None,
                None,
                self.code_result,
            )
            # Create a model factory
            self.model_factory = ModelFactory(model_client)

            if save_folder is not None:
                self.logger.debug("Saving prompt for Track 3c")
                # saving the prompt
                try:
                    if cloud_config.cloud_provider != "s3":
                        if self._fs.exists(pp.join(save_folder, "prompt.txt")):
                            with self._fs.open(
                                pp.join(save_folder, "prompt.txt"), "r"
                            ) as f:
                                _prompt_track3c = f.read()
                    else:
                        _prompt_track3c = read_data(
                            f"{save_folder}/prompt.txt",
                            cloud_config.domain_storage.account_name,
                        )
                except Exception as e:
                    print(e)
                    _prompt_track3c = ""
                if cloud_config.cloud_provider != "s3":
                    with self._fs.open(pp.join(save_folder, "prompt.txt"), "w") as f:
                        f.write(_prompt_track3c)
                        f.writelines("Track 3c\n")
                        f.writelines(self.summary.prompt)
                else:
                    prompt_3c = (
                        _prompt_track3c
                        + "\nTrack 3c\n"
                        + str(self.model_factory.model_type.prompt)
                    )
                    write_data(file_path=f"{save_folder}/prompt.txt", content=prompt_3c)

            (
                self.summary_suggestion,
                self.summary_finish,
                self.summary_tokens,
                self.error_message,
            ) = self.model_factory.model_response(model_param_dict)
            # saving tokens and token information
            self.logger.info(
                f"Track 3c:-\n finish token - {self.summary_finish},\n token information - {self.summary_tokens}"
            )
            # Saving each individual responses of Track 3 subsections as txt and json
            if save_folder is not None:
                self.logger.debug("Saving response for Track 3c")
                try:
                    if cloud_config.cloud_provider != "s3":
                        if self._fs.exists(
                            pp.join(save_folder, "track3_responses.txt")
                        ):
                            with self._fs.open(
                                pp.join(save_folder, "track3_responses.txt"), "r"
                            ) as f:
                                _track3c_responses_txt = f.read()
                    else:
                        _track3c_responses_txt = read_data(
                            f"{save_folder}/track3_responses.txt",
                            cloud_config.domain_storage.account_name,
                        )
                except Exception as e:
                    print(e)
                    _track3c_responses_txt = ""
                if cloud_config.cloud_provider != "s3":
                    with self._fs.open(
                        pp.join(save_folder, "track3_responses.txt"), "w"
                    ) as f:
                        f.write(_track3c_responses_txt)
                        f.writelines("Track 3c\n")
                        f.writelines(self.summary_suggestion)
                else:
                    res_3c = (
                        _track3c_responses_txt
                        + "\nTrack 3c\n"
                        + self.summary_suggestion
                    )
                    write_data(
                        file_path=f"{save_folder}/track3_responses.txt", content=res_3c
                    )
        else:
            self._get_new_summary(token_limit=7000, save_folder=save_folder)

        # log_stage_end("Track 3c", "Track 3")

        return

    def _get_new_summary(
        self, token_limit: int = 7000, save_folder: str = None
    ) -> None:
        """Uses the entire table (it can be either track 1 table or full data) to generate insights. If entire table is too large, it will get trimmed to fit token limit of GPT model.

        Parameters
        ----------
        token_limit : int, optional
            Token limit as defined by GPT model we are using., by default 7000
        save_folder : str, optional
            Path where all intermediate input and outputs will be saved., by default None

        Raises
        ------
        ValueError
            if any of the arguments is missing or invalid or if process errors out.
        """
        # First check whether new token limit doesnt exceed 30k (2k reserved for prompt)
        prompt_dict = self.model_config.summarize_tables.prompts
        model_param_dict = self.model_config.summarize_tables.model_params
        model_param_dict["history"] = self.bot_history

        # Convert df to string
        table_string = convert_df_to_csv_string(self.table)
        # Get the number of tokens in the table
        num_tokens = int(get_gpt_token_count(input_data=table_string, model="gpt-4"))
        self.logger.info(f"Number of tokens for full table is {num_tokens}")

        curr_limit = int(self.user_config.table_top_rows)
        trimFlag = False
        self.logger.info(f"Num token = {num_tokens}, token_limit = {token_limit}")
        # If num_tokens exceeds token_limit, trim the table to fit the token limit
        while num_tokens > token_limit:
            curr_limit = int(curr_limit / 2)
            table_data = os.linesep.join(
                table_string.splitlines(keepends=True)[:curr_limit]
            )
            num_tokens = int(get_gpt_token_count(input_data=table_data, model="gpt-4"))
            trimFlag = True
        # If table is trimmed, log a warning message
        if trimFlag:
            warning_msg = f"Table has been trimmed to take top {curr_limit} rows to fit the token limit of GPT. Thus, any insights that is arises out of this data may not be correct as it's not the full representation of the data."
            self.logger.warning(warning_msg)
        else:
            warning_msg = ""

        self.logger.info(
            f"Number of tokens after handling for token limitation is {num_tokens}"
        )

        if trimFlag:  # We should update this var only after exiting the loop
            table_string = table_data

        self.track = "summarize_tables"
        try:
            # Load Configuration for Track 3c
            if num_tokens <= token_limit:
                config = dict()
                config["llm_model_type"] = model
                model_client = get_model_type(
                    config,
                    prompt_dict,
                    self.question,
                    self.additional_context,
                    self.connection_param_dict,
                    self.track,
                    self.user_config,
                    self.language,
                    self.dictionary,
                    self.business_overview,
                    None,
                    None,
                    None,
                    table_string,
                )
                self.model_factory = ModelFactory(model_client)

                if save_folder is not None:
                    self.logger.debug("Saving prompt for Track 3c - table summary")
                try:
                    if cloud_config.cloud_provider != "s3":

                        if self._fs.exists(pp.join(save_folder, "prompt.txt")):
                            with self._fs.open(
                                pp.join(save_folder, "prompt.txt"), "r"
                            ) as f:
                                _table_prompt = f.read()
                    else:
                        _table_prompt = read_data(
                            f"{save_folder}/prompt.txt",
                            cloud_config.domain_storage.account_name,
                        )

                except Exception as e:
                    print(e)
                    _table_prompt = ""
                if cloud_config.cloud_provider != "s3":
                    with self._fs.open(pp.join(save_folder, "prompt.txt"), "w") as f:
                        f.write(_table_prompt)
                        f.writelines("Track 3c - Table summary:\n")
                        f.writelines(self.summary.prompt)
                else:
                    prmt_3c = (
                        _table_prompt
                        + "\nTrack 3c - Table summary:\n"
                        + str(self.model_factory.model_type.prompt)
                    )
                    write_data(file_path=f"{save_folder}/prompt.txt", content=prmt_3c)
                # Make the API call to GPT
                (
                    self.summary_suggestion,
                    self.summary_finish,
                    self.summary_tokens,
                    self.error_message,
                ) = self.model_factory.model_response(model_param_dict)

                # Prepend warning msg
                self.summary_suggestion = warning_msg + "\n\n" + self.summary_suggestion

                # Saving each individual responses of Track 3 subsections as txt and json
                if save_folder is not None:
                    self.logger.debug("Saving response for Track 3c - table summary")
                    try:
                        if cloud_config.cloud_provider != "s3":
                            if self._fs.exists(
                                pp.join(save_folder, "track3_responses.txt")
                            ):
                                with self._fs.open(
                                    pp.join(save_folder, "track3_responses.txt"), "r"
                                ) as f:
                                    _track3c_responses_txt = f.read()
                        else:
                            _track3c_responses_txt = read_data(
                                f"{save_folder}/track3_responses.txt",
                                cloud_config.domain_storage.account_name,
                            )
                    except Exception as e:
                        print(e)
                        _track3c_responses_txt = ""

                    if cloud_config.cloud_provider != "s3":
                        with self._fs.open(
                            pp.join(save_folder, "track3_responses.txt"), "w"
                        ) as f:
                            f.write(_track3c_responses_txt)
                            f.writelines("Track 3c - Table summary:\n")
                            f.writelines(self.summary_suggestion)
                    else:
                        res_3c = (
                            _track3c_responses_txt
                            + "\nTrack 3c - Table summary:\n"
                            + self.summary_suggestion
                        )
                        write_data(
                            file_path=f"{save_folder}/track3_responses.txt",
                            content=res_3c,
                        )
            else:
                # Get top 100 rows
                raise ValueError("Token limit exceeded")
        # If there is an error in the process, then run the entire track 3
        except Exception as e:
            error_msg = f"Error occurred while forming summary from table. Error description:\n{e}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    @timing_decorator(track_app_start=False)
    def get_insights(self, units_to_skip=[]) -> str:
        """This is the main method that will call each subsection of track 3 in order

        Parameters
        ----------
        units_to_skip : list, optional
            Units to skip in the output when track 1 returns a scalar table. In some cases, we're seeing that the GPT returns units as `integer`, `count`, `text` etc which we want to skip. Defaulted this to empty list because we don't want this parameter to be a bottleneck for running track 2 code, by default []

        Returns
        -------
        str
            Final track 3 summary
        """
        # If we get scalar as input, no need to generate, just reply with Question and Answer as the scalar
        if self.table.shape == (1, 1):  # scalar df
            self.logger.info(
                "Track 1 result is scalar, thus we are returning the result as is."
            )
            try:
                if not isinstance(self.dictionary, str):
                    self.dictionary = str(self.dictionary)
                data_dict = literal_eval(self.dictionary)
                unit = data_dict[0]["unit_of_measurement"]
                if unit.lower() in units_to_skip:
                    unit = ""
            except Exception as e:
                self.logger.error(
                    f"unit of measurement not found in data dictionary. Error description: {e}"
                )
                unit = ""

            self.summary_suggestion = (
                "Q:"
                + self.question
                + "\nAns:"
                + str(self.table.iloc[0, 0])
                + f" {unit}"
            )
        elif self.table.shape == (1,):  # scalar series
            self.logger.info(
                "Track 1 result is scalar, thus we are returning the result as is."
            )
            self.summary_suggestion = (
                "Q:" + self.question + "\nAns:" + str(self.table.iloc[0])
            )
        # If we have single row but many columns, pass to summary as string.
        elif self.table.shape[0] == 1 and self.table.shape[1] > 1:
            self.logger.info("Track 3 - table summary started for single row.")
            # TODO: Explore pretty prints to avoid this data sending to GPT
            self._get_new_summary(save_folder=self.output_path)
            self.logger.info("Track 3 - table summary completed for single row.")

        else:
            # CODE APPROACH
            if self.skip_model:
                # It needs to be updated.
                try:
                    self.logger.info(
                        "Question already exists. Retreiving the suggestion from Knowledge Base."
                    )
                    # run Track 3c
                    self.logger.info("Track 3c started.")
                    self._get_summary(self.output_path)
                    self.logger.info("Track 3c completed.")
                # If there is an error in retreiving the existing results, then run the entire track 3.
                except Exception as e:
                    self.logger.error(
                        f"Error in retreiving the existing results. API call will be triggered. Error description: {e}"
                    )
                    self.logger.error(traceback.format_exc())

                    self.logger.info("Track 3a started.")
                    start_time_track_3a = timeit.default_timer()
                    self._table_to_insight_questions(self.output_path)
                    end_time_track_3a = timeit.default_timer()
                    self.track_3a_runtime = round(
                        end_time_track_3a - start_time_track_3a, 2
                    )
                    self.logger.info(
                        f"Time taken for Track 3a: {round(end_time_track_3a - start_time_track_3a, 2)} seconds."
                    )
                    self.logger.info("Track 3a completed.")
                    # time.sleep(5)  # Sleep for 5s to prevent server overloaded requests error.
                    # run Track 3b
                    self.logger.info("Track 3b started.")
                    start_time_track_3b = timeit.default_timer()
                    self._insight_questions_to_code(self.output_path)
                    end_time_track_3b = timeit.default_timer()
                    self.track_3b_runtime = round(
                        end_time_track_3b - start_time_track_3b, 2
                    )
                    self.logger.info(
                        f"Time taken for Track 3b: {round(end_time_track_3b - start_time_track_3b, 2)} seconds."
                    )
                    self.logger.info("Track 3b completed.")
                    # time.sleep(5)  # Sleep for 5s to prevent server overloaded requests error.
                    # run Track 3c
                    self.logger.info("Track 3c started.")
                    start_time_track_3c = timeit.default_timer()
                    self._get_summary(self.output_path)
                    end_time_track_3c = timeit.default_timer()
                    self.track_3c_runtime = round(
                        end_time_track_3c - start_time_track_3c, 2
                    )
                    self.logger.info(
                        f"Time taken for Track 3c: {round(end_time_track_3c - start_time_track_3c, 2)} seconds."
                    )
                    self.logger.info("Track 3c completed.")

            else:
                self.logger.info("Track 3a started.")
                start_time_track_3a = timeit.default_timer()
                self._table_to_insight_questions(self.output_path)
                end_time_track_3a = timeit.default_timer()
                self.track_3a_runtime = round(
                    end_time_track_3a - start_time_track_3a, 2
                )
                self.logger.info(
                    f"Time taken for Track 3a: {round(end_time_track_3a - start_time_track_3a, 2)} seconds."
                )
                self.logger.info("Track 3a completed.")
                # time.sleep(5)  # Sleep for 5s to prevent server overloaded requests error.
                # run Track 3b
                self.logger.info("Track 3b started.")
                start_time_track_3b = timeit.default_timer()
                self._insight_questions_to_code(self.output_path)
                end_time_track_3b = timeit.default_timer()
                self.track_3b_runtime = round(
                    end_time_track_3b - start_time_track_3b, 2
                )
                self.logger.info(
                    f"Time taken for Track 3b: {round(end_time_track_3b - start_time_track_3b, 2)} seconds."
                )
                self.logger.info("Track 3b completed.")
                # time.sleep(5)  # Sleep for 5s to prevent server overloaded requests error.
                # run Track 3c
                self.logger.info("Track 3c started.")
                start_time_track_3c = timeit.default_timer()
                self._get_summary(self.output_path)
                end_time_track_3c = timeit.default_timer()
                self.track_3c_runtime = round(
                    end_time_track_3c - start_time_track_3c, 2
                )
                self.logger.info(
                    f"Time taken for Track 3c: {round(end_time_track_3c - start_time_track_3c, 2)} seconds."
                )
                self.logger.info("Track 3c completed.")
        # Saving the final JSON.
        self.logger.debug("Saving JSON for evaluation of Track 3.")
        file_save_path = pp.join(self.output_path, "track3_final_result.json")
        save_results_as_json(
            question=self.question,
            # additional_context=self.additional_context,
            # actual_answer="",
            predicted_answer=self.summary_suggestion,
            file_save_path=file_save_path,
            fs=self._fs,
        )

        return self.summary_suggestion

    def create_child_run_track3(self, parent_track3_run_id):
        self.logger.info("mlflow logging for track3 started")
        # child_run_id = self.mlflow_manager.start_child_run(parent_track3_run_id)
        # Set the experiment in MLflowManager
        # self.mlflow_manager.set_experiment(experiment_name)
        # mlflow.set_experiment(self.experiment_name_mlflow)
        # Process for 'table_to_insight_questions'
        if hasattr(self, "query_to_qns") and self.query_to_qns:
            # child_run_id = self.mlflow_manager.start_child_run(
            #     parent_track3_run_id, "table_to_insight_questions"
            # )
            # child_run_id = self.mlflow_manager.start_child_run(
            #     parent_track3_run_id, "table_to_insight_questions"
            # )
            mlflow.set_experiment(self.experiment_name_mlflow)
            child_run = self.mlflow_manager.start_child_run(
                parent_track3_run_id, "table_to_insight_questions"
            )
            # Prepare and log parameters, metrics, and artifacts
            self.mlflow_manager.log_param(
                child_run,
                parent_track3_run_id,
                "LLM Call Type",
                "table_to_insight_questions",
            )
            self.mlflow_manager.log_param(
                child_run,
                parent_track3_run_id,
                "LLM Model",
                self.model_config.table_to_insight_questions.model_params["engine"],
            )
            self.mlflow_manager.log_artifact(
                child_run,
                parent_track3_run_id,
                "Prompt_for_LLM_Call.txt",
                self.query_to_qns.prompt,
            )
            self.mlflow_manager.log_artifact(
                child_run, parent_track3_run_id, "Result.txt", self.question_suggestion
            )
            self.mlflow_manager.log_artifact(
                child_run,
                parent_track3_run_id,
                "Result 1 post processing(code generated).txt",
                "",
            )
            self.mlflow_manager.log_artifact(
                child_run,
                parent_track3_run_id,
                "Result 2 post processing.txt",
                self.question_suggestion,
            )
            # artifacts = {
            #     "LLM_forResult 2 post processing.txt_static_prompt.txt": self.query_to_qns.prompt,
            #     "Result.txt": "",
            #     "Result 2 post processing.txt": self.question_suggestion,
            # }
            # self.mlflow_manager.log_artifacts(
            #     child_run, parent_track3_run_id, artifacts
            # )
            self.mlflow_manager.log_metric(
                child_run,
                parent_track3_run_id,
                "Prompt tokens",
                self.question_tokens["prompt_tokens"],
            )
            self.mlflow_manager.log_metric(
                child_run,
                parent_track3_run_id,
                "Completion tokens",
                self.question_tokens["completion_tokens"],
            )
            self.mlflow_manager.log_metric(
                child_run, parent_track3_run_id, "Run Time", self.track_3a_runtime
            )
            self.mlflow_manager.log_status(
                child_run,
                parent_track3_run_id,
                "success" if not self.query_to_qns.error_message else "failure",
                self.query_to_qns.error_message,
            )

        # Process for 'insight_questions_to_code'
        if hasattr(self, "qns_to_code") and self.qns_to_code:
            # child_run_id = self.mlflow_manager.start_child_run(
            #     parent_track3_run_id, "insight_questions_to_code"
            # )
            # self.mlflow_manager.start_child_run(
            #     parent_track3_run_id, "insight_questions_to_code"
            # )
            mlflow.set_experiment(self.experiment_name_mlflow)
            child_run = self.mlflow_manager.start_child_run(
                parent_track3_run_id, "insight_questions_to_code"
            )
            # Prepare and log parameters, metrics, and artifacts
            self.mlflow_manager.log_param(
                child_run,
                parent_track3_run_id,
                "LLM Call Type",
                "insight_questions_to_code",
            )
            self.mlflow_manager.log_param(
                child_run,
                parent_track3_run_id,
                "LLM Model",
                self.model_config.insight_questions_to_code.model_params["engine"],
            )
            self.mlflow_manager.log_artifact(
                child_run,
                parent_track3_run_id,
                "Prompt_for_LLM_Call.txt",
                self.qns_to_code.prompt,
            )
            self.mlflow_manager.log_artifact(
                child_run, parent_track3_run_id, "Result.txt", ""
            )
            self.mlflow_manager.log_artifact(
                child_run,
                parent_track3_run_id,
                "Result 1 post processing(code generated).txt",
                self.code_suggestion,
            )
            self.mlflow_manager.log_artifact(
                child_run,
                parent_track3_run_id,
                "Result 2 post processing(code generated).py",
                self.track3_trimmed_code,
            )
            # artifacts = {
            #     "LLM_for_static_prompt.txt": self.qns_to_code.prompt,
            #     "Result.txt": "",
            #     "Result 1 post processing(code generated).py": self.code_suggestion,
            #     "Result 2 post processing(code generated).py": self._clean_code(
            #         self.code_suggestion
            #     ),
            # }
            # self.mlflow_manager.log_artifacts(
            #     child_run, parent_track3_run_id, artifacts
            # )
            self.mlflow_manager.log_metric(
                child_run,
                parent_track3_run_id,
                "Prompt tokens",
                self.code_tokens["prompt_tokens"],
            )
            self.mlflow_manager.log_metric(
                child_run,
                parent_track3_run_id,
                "Completion tokens",
                self.code_tokens["completion_tokens"],
            )
            self.mlflow_manager.log_metric(
                child_run, parent_track3_run_id, "Run Time", self.track_3b_runtime
            )
            self.mlflow_manager.log_status(
                child_run,
                parent_track3_run_id,
                "success" if not self.qns_to_code.error_message else "failure",
                self.qns_to_code.error_message,
            )

        # Process for 'summarize_insights' or 'summarize_tables'
        if hasattr(self, "summary") and self.summary:
            run_name = (
                "summarize_insights"
                if not self.track_3c_new_summary
                else "summarize_tables"
            )
            # child_run_id = self.mlflow_manager.start_child_run(
            #     parent_track3_run_id, run_name
            # )
            mlflow.set_experiment(self.experiment_name_mlflow)
            child_run = self.mlflow_manager.start_child_run(
                parent_track3_run_id, run_name
            )

            # Prepare and log parameters, metrics, and artifacts
            self.mlflow_manager.log_param(
                child_run, parent_track3_run_id, "LLM Call Type", run_name
            )
            self.mlflow_manager.log_param(
                child_run,
                parent_track3_run_id,
                "LLM Model",
                self.model_config.get(run_name).model_params["engine"],
            )
            self.mlflow_manager.log_artifact(
                child_run,
                parent_track3_run_id,
                "Prompt_for_LLM_Call.txt",
                self.summary.prompt,
            )
            self.mlflow_manager.log_artifact(
                child_run, parent_track3_run_id, "Result.txt", ""
            )
            self.mlflow_manager.log_artifact(
                child_run,
                parent_track3_run_id,
                "Result 1 post processing(code generated).txt",
                "",
            )
            self.mlflow_manager.log_artifact(
                child_run,
                parent_track3_run_id,
                "Result 2 post processing.txt",
                self.summary_suggestion,
            )
            # artifacts = {
            #     "LLM_for_static_prompt.txt": self.summary.prompt,
            #     "Result.txt": "",
            #     "Result 2 post processing.txt": self.summary_suggestion,
            # }
            # self.mlflow_manager.log_artifacts(
            #     child_run, parent_track3_run_id, artifacts
            # )
            self.mlflow_manager.log_metric(
                child_run,
                parent_track3_run_id,
                "Prompt tokens",
                self.summary_tokens["prompt_tokens"],
            )
            self.mlflow_manager.log_metric(
                child_run,
                parent_track3_run_id,
                "Completion tokens",
                self.summary_tokens["completion_tokens"],
            )
            self.mlflow_manager.log_metric(
                child_run, parent_track3_run_id, "Run Time", self.track_3c_runtime
            )
            self.mlflow_manager.log_status(
                child_run,
                parent_track3_run_id,
                "success" if not self.summary.error_message else "failure",
                self.summary.error_message,
            )

        self.logger.info("mlflow logging for track3 completed")
