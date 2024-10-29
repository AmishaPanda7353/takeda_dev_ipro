import copy
import datetime
import json
import logging
import posixpath as pp
import random
import timeit
import traceback

import openai
import pandas as pd
from core.utils.read_config import cloud_config, config
from src.query_insights.utils.utils import read_data, write_data

from .facilitator import (
    ConfigValidation,
    DataLoader,
    DataProcessor,
    FindUnits,
    FolderManager,
    Generic,
    SimilarityAnalyzer,
    SkipFlag,
)
from .insights_generator.insights_generator import GenerateInsights
from .utils.pre_processing import HybridQuestionClassifier
from .utils.time_logging import timing_decorator
from .utils.utils import (
    SensitiveContentError,
    TimeoutError,
    TokenLimitError,
    upload_data,
)


class TableToInsights:
    def __init__(
        self,
        user_config: dict,
        data_config: dict,
        model_config: dict,
        debug_config: dict,
        question: str = None,
        language: str = "english",
        additional_context: str = None,
        track1_output_table: pd.DataFrame = None,
        track1_output_table_dict: dict = None,
        logging_level: str = "INFO",
    ) -> None:
        """Given user query, generate python code that generates chart and display to the user.
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
        question : str
                New question from the user
        additional_context : str, optional
                Additional context to answer the question, by default None
        language : str, optional
                Language to answer the question in, for example "english", "spanish", "german", by default "english"

        track1_output_table : pd.DataFrame, optional
            Output of text_to_query function by running the SQL generated to answer the ``question``, by default None

        track1_output_table_dict : dict, optional
            Data dictionary of ``track1_output_table`` parameter, by default None


        logging_level : str, optional
            Level or severity of the events they are used to track. Acceptable values are ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], by default "WARNING", by default "WARNING"

        Raises
        ------
        ValueError
            if any of the argument is missing or invalid.

        """

        self.question = question
        self.additional_context = additional_context
        self.language = language
        self.track1_output_table = track1_output_table
        self.track1_output_table_dict = track1_output_table_dict

        self.user_config = user_config
        self.data_config = data_config
        self.model_config = model_config
        self.debug_config = debug_config

        self.logging_level = logging_level

        self.all_tokens = []

        self.MYLOGGERNAME = "QueryInsights"
        self.logger = logging.getLogger(self.MYLOGGERNAME)

    @timing_decorator(track_app_start=True)
    def table_insights(
        self,
        generic_initializations,
        foldercreation,
        dataloader,
        config_validation,
        dataprocessor,
        skip_flag,
    ):
        """This function is responsible for running track3(generate insights)

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

        track3_start_time = timeit.default_timer()

        if cloud_config.cloud_provider != "s3":
            foldercreation._individual_track_folder_creation(
                track="03_table_to_insights"
            )

        if skip_flag.is_followup:
            if hasattr(skip_flag, "model_tokens"):
                if skip_flag.model_tokens is not None:
                    self.all_tokens.append(
                        {"Followup question tag": skip_flag.model_tokens.to_dict()}
                    )

        if self.track1_output_table is None:
            raise ValueError(
                "Data Table is empty. Inorder to run the track individually, Please pass the data table along with the question"
            )
        if self.track1_output_table_dict is None:
            # TODO : We can create the dictionary from the table if dictionary is empty.
            raise ValueError(
                "Data Dictionary is empty. Inorder to run the track individually, Please pass the data dictionary along with the question"
            )
        # check if its a why question.
        if dataprocessor.why_qn_flag:
            self.logger.info(
                f"As given question {self.question} is a why question, track 3 will run on entire data and data dictionary."
            )
            track1_output_table_dict = dataloader.data_dictionary
            track1_output_table = (
                self.track1_output_table
            )  # TODO: Probably need to think what table to use after table selector is ready.
        else:
            # Else, we have generated track 1 output. Fetch it if its not passed directly as arguments.
            # Input data validation
            if self.track1_output_table is None:
                # track1_output_table = self.track1_output_table
                pass
            else:
                # if the dataframe is received as a parameter then we are saving it in the table_to_insights folder
                # This is needed in passing the path to the insights generator(track 3b)
                # track="03_table_to_insights"
                self.text_to_query_path = (
                    f"{foldercreation.output_path}/03_table_to_insights"
                )
                if cloud_config.cloud_provider != "s3":
                    with generic_initializations._fs.open(
                        pp.join(
                            f"{foldercreation.output_path}/03_table_to_insights",
                            "output_table.csv",
                        ),
                        mode="wb",
                        newline="",
                    ) as fp:
                        self.track1_output_table.to_csv(fp, index=False)
                else:
                    write_data(
                        file_path=f"{foldercreation.output_path}/03_table_to_insights/output_table.csv",
                        content=self.track1_output_table,
                    )

            if self.track1_output_table_dict is None:
                # track1_output_table_dict = self.track1_output_table_dict
                pass

        self.logger.info("User query to insight generation started.")

        skip_flag._skip_model_kb()
        self.logger.info(
            f"For Insight generation: existing_question check is {foldercreation.existing_question} and its status in {foldercreation.existing_status}"
        )
        if skip_flag.skip_track3_followup:
            self.logger.info(
                "Insight generation is skipped because this question is identified as a follow up question not related to Inight generation"
            )

        skip_model_flag = skip_flag.skip_model_kb or skip_flag.skip_track3_followup
        gi = GenerateInsights(
            user_config=self.user_config,
            data_config=self.data_config,
            model_config=self.model_config,
            question=self.question,
            dictionary=self.track1_output_table_dict,
            business_overview=generic_initializations.business_overview,
            bot_history=foldercreation.bot_history,
            table=self.track1_output_table,
            skip_model=skip_model_flag,
            output_path=f"{foldercreation.output_path}/03_table_to_insights",
            sql_results_path=self.text_to_query_path,
            language=self.language,
            fs=generic_initializations._fs,
        )
        insights = gi.get_insights(units_to_skip=generic_initializations.units_to_skip)
        self.logger.info("User query to insight generation completed.")
        error_message = gi.error_message

        if hasattr(gi, "question_tokens"):
            if gi.question_tokens is not None:
                token_str = gi.question_tokens
                token_dict = {
                    "completion_tokens": token_str.completion_tokens,
                    "prompt_tokens": token_str.prompt_tokens,
                    "total_tokens": token_str.total_tokens,
                }
                self.all_tokens.append({"Track 3a": token_dict})
        if hasattr(gi, "code_tokens"):
            if gi.code_tokens is not None:
                token_str = gi.code_tokens
                token_dict = {
                    "completion_tokens": token_str.completion_tokens,
                    "prompt_tokens": token_str.prompt_tokens,
                    "total_tokens": token_str.total_tokens,
                }
                self.all_tokens.append({"Track 3b": token_dict})
        if hasattr(gi, "summary_tokens"):
            if gi.summary_tokens is not None:
                token_str = gi.summary_tokens
                token_dict = {
                    "completion_tokens": token_str.completion_tokens,
                    "prompt_tokens": token_str.prompt_tokens,
                    "total_tokens": token_str.total_tokens,
                }
                self.all_tokens.append({"Track 3c": token_dict})

        if error_message is None:
            return_value = {"status": "success", "output": insights}
        else:
            return_value = {"status": "failure", "output": error_message}

        # Creation of Response JSON for track 3
        # Parent keys of json
        response_json_track3 = copy.deepcopy(config_validation.response_json)
        if not config_validation.response_json:
            response_json_track3 = {
                "Request JSON": {"question": self.question},
                "Response JSON": {
                    "error": "",
                    "status": ["", ""],
                    "type": "insights",
                    "data": ["", ""],
                    # "created_time": foldercreation.current_ts,
                    "question_index": foldercreation.question_index,
                    "output_path": foldercreation.output_path,
                    "engines": generic_initializations.model_engine_dict,
                },
            }

        # Creation of Response JSON for track 3

        response_json_track3["Response JSON"]["status"].append(return_value["status"])

        data_dict = {}
        data_dict["insight_type"] = "summary"
        if return_value["status"] == "success":
            data_dict["content"] = return_value["output"]
            data_dict["error"] = ""
            data_dict["showError"] = False
        elif return_value["status"] == "failure":
            data_dict["content"] = ""
            data_dict["error"] = return_value["output"]
            data_dict["showError"] = True
        data_dict["bot_response"] = ""

        response_json_track3["Response JSON"]["data"].append(data_dict)
        # JSON is created to be used for front-end applications.
        try:
            # Save the JSON data to a file
            output_file_path = "response.json"  # Set the desired file path
            if cloud_config.cloud_provider != "s3":
                with generic_initializations._fs.open(
                    pp.join(
                        foldercreation.output_path,
                        "03_table_to_insights",
                        output_file_path,
                    ),
                    "w",
                ) as json_file:
                    json.dump(response_json_track3, json_file, indent=4)
            else:
                write_data(
                    file_path=f"{foldercreation.output_path}/03_table_to_insights/{output_file_path}",
                    content=response_json_track3,
                )
        except Exception as e:
            self.logger.error(f"Response JSON not saved due to an error : {e}")

        track3_end_time = timeit.default_timer()
        self.logger.info(
            f"Time taken to run track 3: {round(track3_end_time - track3_start_time, 2)} seconds."
        )
        completion_success_phrases = [
            "Done! Here are the results.",
            "All set! Here's what I found.",
            "Got it! Here are the results for you.",
            "Finished! Here's what you were looking for.",
            "Here's what I came up with.",
            "Completed! Here's what I found out for you.",
            "Task accomplished! Here are the results you requested.",
            "Task fulfilled! Here's what I found.",
            "Done and done! Here are the results you asked for.",
        ]

        status = response_json_track3["Response JSON"]["status"]
        # if track 1 fails, response_for_history is updated in text to query function with bot response
        # if track 2 fails, table is returned so response can still be success message
        if status[1] == "failure":
            # if track 3 fails, response_for_history is updated with ""
            self.completion_response = random.choice(
                generic_initializations.completion_error_phrases
            )
            response_json_track3["Response JSON"]["response_for_history"] = ""
        else:
            # if none of the tracks fail, response_for_history is updated with success message
            self.completion_response = random.choice(completion_success_phrases)
            response_json_track3["Response JSON"][
                "response_for_history"
            ] = self.completion_response
        response_json_track3["Response JSON"][
            "completion_response"
        ] = self.completion_response
        self.response_track3 = response_json_track3

        if not hasattr(gi, "track3a_latency"):
            lat_3a = 0
        else:
            lat_3a = gi.track3a_latency

        if not hasattr(gi, "track3b_latency"):
            lat_3b = 0
        else:
            lat_3b = gi.track3b_latency

        if not hasattr(gi, "track3c_latency"):
            lat_3c = 0
        else:
            lat_3c = gi.track3c_latency

        if config.user_config_domains.takeda.save_track_output:

            track_status = {
                # "question_id": [self.question_id],
                "question": [self.question],
                "run_ts": [datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")],
                "track": [3],
                "experiment": [self.data_config.path.exp_name],
                "output_path": [foldercreation.table_to_insights_path],
                "engine": [
                    f"{self.user_config.connection_params.api_type.table_to_insight_questions}--{self.user_config.connection_params.api_type.insight_questions_to_code}--{self.user_config.connection_params.api_type.summarize_tables}"
                ],
                "track_latency": [round(track3_end_time - track3_start_time, 2)],
                "LLM_latency": [lat_3a + lat_3b + lat_3c],
                "LLM_Response": [return_value["status"]],
            }
            if cloud_config.cloud_provider != "s3":
                pd.DataFrame(track_status).to_csv(
                    pp.join(
                        foldercreation.output_path,
                        "03_table_to_insights",
                        "track_run_status.csv",
                    ),
                    index=False,
                )
            else:
                write_data(
                    file_path=f"{foldercreation.output_path}/03_table_to_insights/track_run_status.csv",
                    content=pd.DataFrame(track_status),
                )
        return return_value
