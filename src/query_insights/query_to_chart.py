import copy
import datetime
import json
import logging
import posixpath as pp
import random
import timeit
import traceback

import numpy as np
import openai
import pandas as pd
from core.utils.read_config import cloud_config, config
from src.query_insights.utils.utils import write_data

from .chart_generator.chart_generator import GenerateCharts
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
from .utils.time_logging import timing_decorator
from .utils.utils import (
    SensitiveContentError,
    TimeoutError,
    TokenLimitError,
    download_nltk_data,
    download_spacy_data,
    format_dataframe,
    upload_data,
    read_data,

)


class QueryToChart:
    def __init__(
        self,
        user_config: dict,
        data_config: dict,
        model_config: dict,
        debug_config: dict,
        question: str = None,
        question_path: str = None,
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

        track1_output_table = read_data(
                                bucket=cloud_config.domain_storage.account_name,
                                path=(f"{question_path}/01_text_to_query/output_table.csv"))
        track1_output_table_dict = read_data(
                                bucket=cloud_config.domain_storage.account_name,
                                path=(f"{question_path}/01_text_to_query/output_data_dictionary.txt"))

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

    def _check_empty_charts(self, chart_object) -> bool:
        """
        Function to check whether the chart is empty without axis values or all axis values as 0.

        Parameters
        ----------
        chart_object : JSON
            Plotly Chart Object

        Returns
        -------
        bool
            Boolean to check whether the chart is empty without axis values or all axis values as 0.
        """
        all_zeros = False
        self.logger.info("Checking if all the axis values are 0 in the chart.")
        # Initialize the y-values to an empty list and loop over the chart object for all y-values and append to this list.
        axis_values = []
        for trace in chart_object["data"]:
            # Check if x and y numeric values are present in the chart object.
            # For some plotly charts like Tabular views/histogram, y parameter is not present in the JSON.
            for axis in ["x", "y", "z"]:
                if axis in trace:
                    if isinstance(trace[axis], (np.ndarray, pd.Series)):
                        axis_values = axis_values + trace[axis].tolist()
                    elif isinstance(trace[axis], list):
                        axis_values = axis_values + trace[axis]
                    elif isinstance(trace[axis], tuple):
                        axis_values = axis_values + list(trace[axis])
                    else:
                        self.logger.info("Chart object has other data type.")

        # Remove any non-numeric values from the list since we are looking at both x and y axis.
        axis_values = [
            v for v in axis_values if isinstance(v, (int, float)) or v is None
        ]
        # Check if all the values are 0 in the y-values list and change the all_zeros flag.
        if len(axis_values) > 0 and all(
            value == 0 or value is None for value in axis_values
        ):
            all_zeros = True
            self.logger.info(
                "All axis-values are 0 in the chart. So the chart can be skipped."
            )
        else:
            if len(axis_values) == 0:
                self.logger.info("No axis-values present. It could be a tabular view.")
            else:
                self.logger.info(
                    "Axis-values other than 0 present in the chart object."
                )

        return all_zeros

    @timing_decorator(track_app_start=True)
    def query_to_chart(
        self,
        generic_initializations,
        foldercreation,
        config_validation,
        dataprocessor,
        skip_flag,
    ):
        """This function is responsible for running track2(from query to chart)

        Parameters
        ----------
        generic_initializations : object
            object for common attributes

        foldercreation : object
            object for FolderManager class

        config_validation : object
            object for ConfigValidation class

        dataprocessor : object
            object for DataProcessor class

        skip_flag : object
            object for SkipFlag class

        Returns
        -------


        """

        track2_start_time = timeit.default_timer()
        if cloud_config.cloud_provider != "s3":
            foldercreation._individual_track_folder_creation("02_query_to_chart")
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
        # Check if its a why Question.
        if dataprocessor.why_qn_flag:
            skip_reason = f"As given question {self.question} is a why question, chart generation will be skipped."
            self.logger.info(skip_reason)
            return_value = {
                "status": "skip",
                "output": skip_reason,
            }
            return return_value

        # Input data validation
        if self.track1_output_table is None:
            # track1_output_table = self.track1_output_table
            pass
        else:
            # if the dataframe is received as a parameter then we are saving it in the query_to_insights folder
            # This is needed in passing the path to the chart code
            self.text_to_query_path = f"{foldercreation.output_path}/02_query_to_chart"
            if cloud_config.cloud_provider != "s3":
                with generic_initializations._fs.open(
                    pp.join(self.text_to_query_path, "output_table.csv"),
                    mode="wb",
                    newline="",
                ) as fp:
                    self.track1_output_table.to_csv(fp, index=False)
            else:
                write_data(
                    file_path=f"{foldercreation.output_path}/02_query_to_chart/output_table.csv",
                    content=self.track1_output_table,
                )

        # if track1_output_table_dict is None: #optional
        #     track1_output_table_dict = self.track1_output_table_dict

        # Track 2 - to generate chart type/code suggestion from Track 1 results
        # track1_output_table_dict = track1_output_table_dict.replace(

        # )  # replace single quotes with double quotes
        # track1_data_dict = {"columns": json.loads(track1_output_table_dict)}
        alternate_dict = None
        if self.track1_output_table is not None:
            alternate_dict = {
                "columns": {
                    col: dtype.name
                    for col, dtype in self.track1_output_table.dtypes.items()
                }
            }

        # Initializing the track 1 data dict to None.
        track1_data_dict = None
        try:
            self.logger.info("Reading the data dictionary.")
            # track1_data_dict = ast.literal_eval("{'columns':" + track1_output_table_dict + "}")
            # Create a new dictionary with the "columns" key and the list of dictionaries as its value
            # dict_list = _string_to_dict(track1_output_table_dict)
            dict_list = self.track1_output_table_dict
            if len(dict_list) == self.track1_output_table.shape[1]:
                track1_data_dict = {"columns": dict_list}
                self.logger.info(f"data dictionary is read - {track1_data_dict}")
            else:
                self.logger.info(
                    """An error occured while reading data dictionary. It doesn't have all the column details of track 1 result.
                    Changing that to columns list."""
                )
                track1_data_dict = alternate_dict
        except Exception as e:
            self.logger.info(
                f"Some error with the  data dictionary. Changing that to columns list. Error - {e}"
            )
            track1_data_dict = alternate_dict

        # print(track1_data_dict)
        self.logger.info("User query to chart generation started.")
        self.chart_object = None  # Default

        if dataprocessor.multiple_charts:
            self.logger.info(
                "User has opted for more than one chart. Track 2 will be processed accordingly."
            )
        if len(foldercreation.existing_status) == 1:
            foldercreation.existing_status.append([None, None])
        skip_flag._skip_model_kb()
        self.logger.info(
            f"For Chart generation: existing_question check is {foldercreation.existing_question} and its status in {foldercreation.existing_status[1]}"
        )
        if skip_flag.skip_track2_followup:
            self.logger.info(
                "Chart generation is skipped because this question is identified as a follow up question and is not Grapth related"
            )

        skip_model_flag = skip_flag.skip_model_kb or skip_flag.skip_track2_followup
        track2_ins = GenerateCharts(
            user_config=self.user_config,
            data_config=self.data_config,
            model_config=self.model_config,
            question=self.question,
            additional_context=self.additional_context,
            table=self.track1_output_table,
            data_dictionary=track1_data_dict,
            business_overview=generic_initializations.business_overview,
            bot_history=foldercreation.bot_history,
            output_path=foldercreation,
            skip_model=skip_model_flag,
            sql_results_path=self.text_to_query_path,
            multiple_charts=dataprocessor.multiple_charts,
            fs=generic_initializations._fs,
        )
        self.chart_object = track2_ins.process_suggestion()
        self.logger.info("User query to chart generation completed.")
        error_message = track2_ins.error_message

        # Check if all the axis values are 0.
        # Initializing the all_zeros flag to False.
        if self.chart_object is not None:
            self.all_zeros = [False] * len(self.chart_object)
            for i in range(0, len(self.chart_object)):
                if self.chart_object[i] is not None:
                    try:
                        if "Chart Metrics" in self.chart_object[i]:
                            # Since the metrics can have 0 in it, we can keep the metrics as is.
                            # So keeping the Flag as False.
                            self.all_zeros[i] = False
                        else:
                            self.all_zeros[i] = self._check_empty_charts(
                                self.chart_object[i]
                            )
                    except Exception as e:
                        error_string = (
                            f"Error in identifying empty chart for {i}. \n Error: {e}"
                        )
                        self.logger.info(error_string)

        if hasattr(track2_ins, "chart_type_tokens"):
            if track2_ins.chart_type_tokens is not None:
                token_str = track2_ins.chart_type_tokens
                token_dict = {
                    "completion_tokens": token_str.completion_tokens,
                    "prompt_tokens": token_str.prompt_tokens,
                    "total_tokens": token_str.total_tokens,
                }
                self.all_tokens.append({"Track 2a": token_dict})

        if hasattr(track2_ins, "chart_code_tokens"):
            if track2_ins.chart_code_tokens is not None:
                token_str = track2_ins.chart_code_tokens
                token_dict = {
                    "completion_tokens": token_str.completion_tokens,
                    "prompt_tokens": token_str.prompt_tokens,
                    "total_tokens": token_str.total_tokens,
                }
                self.all_tokens.append({"Track 2a": token_dict})

        if error_message is None:
            if self.chart_object is None or all(self.all_zeros):
                # if isinstance(self.chart_object, list):
                #     if len(self.chart_object) == 1:
                #         self.chart_object = self.chart_object[0]
                return_value = {
                    "status": "skip",
                    "output": (self.chart_object, self.track1_output_table),
                }
            else:
                # Remove the list if there is only one element in chart object.
                # if len(self.chart_object) == 1:
                #     self.chart_object = self.chart_object[0]
                return_value = {
                    "status": "success",
                    "output": (self.chart_object, self.track1_output_table),
                }
        else:
            return_value = {
                "status": "failure",
                "output": (error_message, self.track1_output_table),
            }

        # Creation of Response JSON for track 2
        # Parent keys of json
        response_json_track2 = copy.deepcopy(config_validation.response_json)
        if not config_validation.response_json:
            response_json_track2 = {
                "Request JSON": {"question": self.question},
                "Response JSON": {
                    "error": "",
                    "status": [""],
                    "type": "insights",
                    "data": [""],
                    # "created_time": foldercreation.current_ts,
                    "question_index": foldercreation.question_index,
                    "output_path": foldercreation.output_path,
                    "engines": generic_initializations.model_engine_dict,
                },
            }

        response_json_track2["Response JSON"]["status"].append(return_value["status"])
        data_dict = {}
        table = return_value["output"][1]
        if return_value["status"] == "success":
            # when track 2 runs successfully
            data_dict["insight_type"] = "chart"
            if len(self.chart_object) > 1:
                data_dict["insight_type"] = "multi_chart"
            data_dict["content"] = [obj if isinstance(obj,dict) else obj.to_dict() for obj in self.chart_object]
            data_dict["error"] = ""
            data_dict["showError"] = False
            data_dict["table"] = [format_dataframe(table)]

        else:
            # when track 2 has failed or been skipped
            data_dict["insight_type"] = "table"
            if table.shape == (1, 1):
                # if track2 is skipped and returns a scalar value in a 1x1 table
                finding_units = FindUnits(
                    self.track1_output_table_dict, generic_initializations.units_to_skip
                )
                unit = finding_units._find_units_of_measurement()
                data_dict["insight_type"] = "scalar"
                data_dict["content"] = f"{table.values[0][0]} {unit}"
            else:
                # if track2 is skipped and returns table
                data_dict["insight_type"] = "table"
                data_dict["content"] = [format_dataframe(table)]

            if return_value["status"] == "failure":
                # Not showing error message when track 2 fails, as we are displaying the table.
                data_dict["error"] = return_value["output"][0]
                data_dict["showError"] = False
            else:
                # show error message when track 2 gets skipped
                data_dict["error"] = ""
                data_dict["showError"] = False
            data_dict["table"] = ""
        data_dict["bot_response"] = ""

        if return_value["status"] == "failure":
            generic_initializations.completion_response = random.choice(
                generic_initializations.completion_error_phrases
            )

        response_json_track2["Response JSON"]["data"].append(data_dict)
        # JSON is created to be used for front-end applications.
        try:
            # Save the JSON data to a file
            output_file_path = "response.json"  # Set the desired file path
            if cloud_config.cloud_provider != "s3":
                with generic_initializations._fs.open(
                    pp.join(foldercreation.output_path, output_file_path), "w"
                ) as json_file:
                    json.dump(response_json_track2, json_file, indent=4)
            else:
                write_data(
                    file_path=f"{foldercreation.output_path}/02_query_to_chart/{output_file_path}",
                    content=response_json_track2,
                )
        except Exception as e:
            self.logger.error(f"Response JSON not saved due to an error : {e}")
        self.response_track2 = response_json_track2
        track2_end_time = timeit.default_timer()
        self.logger.info(
            f"Time taken to run track 2: {round(track2_end_time - track2_start_time, 2)} seconds."
        )

        # if config.cloud_details.mcd.cloud_provider == "s3":
        #     upload_data(self.data_config.path.exp_name)

        if not hasattr(track2_ins, "track2a_latency"):
            track2a_latency = 0
        else:
            track2a_latency = track2_ins.track2a_latency

        if not hasattr(track2_ins, "track2b_latency"):
            track2b_latency = 0
        else:
            track2b_latency = track2_ins.track2b_latency

        if config.user_config_domains.mcd.save_track_output:

            track_status = {
                # "question_id": [self.question_id],
                "question": [self.question],
                "run_ts": [datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")],
                "track": [2],
                "experiment": [self.data_config.path.exp_name],
                "output_path": [f"{foldercreation.output_path}/02_query_to_chart"],
                "engine": [
                    f"{self.user_config.connection_params.api_type.query_to_chart_type}-{self.user_config.connection_params.api_type.query_to_chart_code}"
                ],
                "track_latency": [round(track2_end_time - track2_start_time, 2)],
                # "LLM_latency": [track2a_latency + track2b_latency],
                "LLM_Response": [return_value["status"]],
            }
            if cloud_config.cloud_provider != "s3":
                pd.DataFrame(track_status).to_csv(
                    pp.join(
                        f"{foldercreation.output_path}/02_query_to_chart",
                        "track_run_status.csv",
                    ),
                    index=False,
                )
            else:
                write_data(
                    file_path=f"{foldercreation.output_path}/02_query_to_chart/track_run_status.csv",
                    content=pd.DataFrame(track_status),
                )

        return return_value
