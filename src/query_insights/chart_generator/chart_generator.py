import io
import json
import logging
import os
import posixpath as pp
import subprocess
import sys
import timeit
import traceback
from io import StringIO
import pandas as pd
import numpy as np
import black
import boto3
import fsspec
import isort
import mlflow
import plotly
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from core.model.model_factory import ModelFactory
from core.utils.client_utils import get_model_type
from core.utils.read_config import cloud_config, cloud_secrets, model
from src.query_insights.utils.utils import read_data, write_data

# log_stage_end, log_stage_start, set_current_track
from ..utils.post_processing import (
    _get_import_statements,
    _uniquecategory_check,
    add_exception_to_code,
    check_for_static_lists,
    clean_chart_code,
    extract_code,
)
from ..utils.time_logging import set_task_label, timing_decorator
from ..utils.utils import generate_env_dict, rate_limit_error_handler, read_text_file
from src.query_insights.utils.local_chart import (
                simple_line_chart,
                simple_bar_chart, 
                simple_scatter_chart,
                group_bar_chart,
                group_scatter_bar_chart,
                group_scatter_chart)

MYLOGGERNAME = "QueryInsights"


class GenerateCharts:
    """
    Generate Chart type and corresponding code from data extracted from Track 1(or Text to Query). It consists of two steps.
    1. Get the chart type suggestion from GPT based on the business question and track 1's data dictionary.
    2. Using the suggestion from Track 1 and get the chart code from GPT.
    3. Using the code to generate the chart.

    Parameters
    ----------
    user_config : dict
        input user_config dictionary for storing and accessing user-specific configurations.
    data_config : dict
        input data_config dictionary contains the paths to the data.
    model_config : dict
        input model_config dictionary for storing and accessing model-related configurations.
    question : str
        User question to be answered
    additional_context : str
        Additional context to answer the question
    table: pandas df
        Ouput from Track 1 (Text to Query)
    data_dictionary : dict
        contains table name, column name and description
    output_path : str
        path to save the results
    skip_model : bool
        condition whether to skip the api call.
    multiple_charts: bool
        condition to indicate if the user needs multiple charts
    fs : fsspec.filesystem, optional
        Filesystem of the url, None will default to local file system, by default ``None``
    language : str
        Language to answer the question in, for example "english", "spanish", "german", by default "english"
    """

    def __init__(
        self,
        user_config,
        data_config,
        model_config,
        question,
        additional_context,
        table,
        data_dictionary,
        business_overview,
        bot_history,
        output_path,
        skip_model: bool,
        sql_results_path: str,
        multiple_charts: bool,
        fs=None,
        language: str = "english",
    ):
        """_summary_"""
        self.user_config = user_config
        self.data_config = data_config
        self.model_config = model_config
        self.ui = user_config.ui
        self.multiple_charts = multiple_charts

        self.question = question
        self.additional_context = additional_context
        self.data_dictionary = data_dictionary
        self.business_overview = business_overview
        self.output_path = f"{output_path.output_path}/02_query_to_chart"
        self.connection_param_dict = user_config.connection_params
        self.input_table = table
        self.skip_model = skip_model
        self.bot_history = bot_history

        # Init for tmp files and names
        self.sql_results_path = pp.join(sql_results_path, "output_table.csv")
        self.track2_code_path = pp.join(self.output_path, "chartcode_<n>.py")
        self.track2_chart_path = pp.join(self.output_path, "chart_<n>.json")
        self.track2_metrics_path = pp.join(self.output_path, "metrics.json")

        self._fs = fs or fsspec.filesystem("file")

        # Required for decorator
        time_delay = user_config.time_delay
        max_retries_2a = model_config.query_to_chart_type.model_params.max_tries
        max_retries_2b = model_config.query_to_chart_code.model_params.max_tries
        self.track_2a_runtime = 0
        self.track_2b_runtime = 0
        self.query_to_chartcode_ins = None
        self.query_to_charttype_ins = None

        # If language is None or empty string, default to "english" language
        if language is None or not bool(language.strip()):
            language = "english"
        language = language.lower().title()

        self.language = language
        self.logger = logging.getLogger(MYLOGGERNAME)
        self.error_message = None

        # Normal way of using decorator as we are getting trouble passing arguments
        # in intended way of "@rate_limit_error_handler(...)"
        self._charttype_apicall = rate_limit_error_handler(
            logger=self.logger, time_delay=time_delay, max_retries=max_retries_2a
        )(self._charttype_apicall)
        self._chartcode_apicall = rate_limit_error_handler(
            logger=self.logger, time_delay=time_delay, max_retries=max_retries_2b
        )(self._chartcode_apicall)

        # mlflow parameters

        # self.experiment_name_mlflow = self.data_config.experiment_params.experiment_name
        # self.mlflow_manager = MLflowManager(experiment_name=self.experiment_name_mlflow)
        return

    @timing_decorator(track_app_start=False)
    def _charttype_apicall(self):
        """Track 2a - Using Business user query and Track 1 (SQL query) results to get a chart-type suggestion from the model.

        This method makes an API call to a model to get a suggestion for the chart type based on the business user query and the results of a SQL query.

        Returns:
            None
        """
        # Load Configuration for Track 2a.
        prompt_dict = self.model_config.query_to_chart_type.prompts
        model_param_dict = self.model_config.query_to_chart_type.model_params
        model_param_dict["history"] = self.bot_history
        self.track = "query_to_chart_type"

        # Updating the prompts based on multiple charts flag.
        if self.multiple_charts:
            prompt_dict["static_prompt"] = prompt_dict["static_prompt_multiplecharts"]

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
            self.data_dictionary,
            self.business_overview,
        )
        self.model_factory = ModelFactory(model_client)
        # GPT Model call for Track 2a
        self.logger.info("Track2 charttype api model call started.")
        start_time_track_chartype_2a = timeit.default_timer()
        (
            self.chart_type_output,
            self.chart_type_finish,
            self.chart_type_tokens,
            self.error_message,
        ) = self.model_factory.model_response(model_param_dict)
        end_time_track_chartype_2a = timeit.default_timer()
        self.logger.info(
                f"Time taken for Track 2a(charttype api model call)): {round(end_time_track_chartype_2a - start_time_track_chartype_2a, 2)} seconds."
            )
        return

    @timing_decorator(track_app_start=False)
    def _chartcode_apicall(self):
        """Track 2b - Using Business user query and Track 1 (SQL query) results and Track 2a (Chart Type Suggestion)
        to get a chart-code suggestion from the model.

        This method performs an API call to retrieve a chart-code suggestion from the model based on the business user query,
        SQL query results, and chart type suggestion. It loads the configuration for Track 2b, updates the prompts based on
        the multiple charts flag, and makes the necessary model calls to generate the chart code.

        Returns:
            None
        """
        # Load Configuration for Track 2b.
        prompt_dict = self.model_config.query_to_chart_code.prompts
        model_param_dict = self.model_config.query_to_chart_code.model_params
        model_param_dict["history"] = self.bot_history
        self.track = "query_to_chart_code"
        # Updating the prompts based on multiple charts flag.
        if self.multiple_charts:
            prompt_dict["static_prompt"] = prompt_dict["static_prompt_multiplecharts"]
            prompt_dict["guidelines"] = prompt_dict["guidelines_multiplecharts"]

        # llm model call
        config = dict()
        config["llm_model_type"] = model
        model_client = get_model_type(
            config,
            prompt_dict,
            self.question,
            self.chart_type_output,
            self.connection_param_dict,
            self.track,
            self.user_config,
            self.language,
            self.data_dictionary,
            self.business_overview,
        )
        self.model_factory = ModelFactory(model_client)
        # GPT model call for Track 2b
        self.logger.info("Track2 chartcode api model call started.")
        start_time_track_chartcode_2a = timeit.default_timer()

        (
            self.chart_code_output,
            self.chart_code_finish,
            self.chart_code_tokens,
            self.error_message,
        ) = self.model_factory.model_response(model_param_dict)

        end_time_track_chartcode_2a = timeit.default_timer()
        self.logger.info(
                f"Time taken for Track 2a(chartcode api model call)): {round(end_time_track_chartcode_2a - start_time_track_chartcode_2a, 2)} seconds."
            )
        return

    @timing_decorator(track_app_start=False)
    def _post_processing(self, track: str):
        """
        Save the files based on the track details.

        Parameters
        ----------
        track : str
            Either Track2a or Track2b
        """
        if track == "Track2a":
            # Save the prompt
            self._save_outputs(
                file_type="text",
                output_folder=self.output_path,
                file_name="charttype_prompt.txt",
                content=self.model_factory.model_type.prompt,
            )

            # sometimes code gives an error that GPTModelCall doesnot have attribute current_message (reason unknown). so adding an if condition
            if hasattr(self.model_factory.model_type, "current_message"):
                self._save_outputs(
                    file_type="json",
                    output_folder=self.output_path,
                    file_name="track2a_current_message.json",
                    content=self.model_factory.model_type.current_message,
                )

            # Save the chart-type suggestion
            self.logger.debug("Saving chart type suggestion.")
            self._save_outputs(
                file_type="text",
                output_folder=self.output_path,
                file_name="charttype_suggestion.txt",
                content=self.chart_type_output,
            )

        elif track == "Track2b":
            # Save the prompt
            self.logger.debug("Saving chart code suggestion prompt.")
            self._save_outputs(
                file_type="text",
                output_folder=self.output_path,
                file_name="chartcode_prompt.txt",
                content=self.model_factory.model_type.prompt,
            )

            # sometimes code gives an error that GPTModelCall doesnot have attribute current_message (reason unknown). so adding an if condition
            if hasattr(self.model_factory.model_type, "current_message"):
                self._save_outputs(
                    file_type="json",
                    output_folder=self.output_path,
                    file_name="track2b_current_message.json",
                    content=self.model_factory.model_type.current_message,
                )

            # Save the chart-code suggestion
            self.logger.debug("Saving chart code suggestion.")
            if self.question is None:
                self._save_outputs(
                    file_type="text",
                    output_folder=self.output_path,
                    file_name="chartcode_suggestion_wo_question.txt",
                    content=self.chart_code_output,
                )
            else:
                self._save_outputs(
                    file_type="text",
                    output_folder=self.output_path,
                    file_name="chartcode_suggestion.txt",
                    content=self.chart_code_output,
                )

            # Extracting the codes from the suggestion and executing it.
            # If there are more than one start substrings and one element is a substring of other, please order it in a way the first element is a subset of other element.
            # For example - If the start elements are <start> and <begin>, they can be given in any order.
            # If start elements are "```python" and "```", one is a substring of the other. "```" should be specified before "```python".
            self.chart_code_list = extract_code(
                string_input=self.chart_code_output,
                start=["```", "```python", "<start>", "```\n```"],
                end=["```", "<end>", "```\n```"],
                extract="all",
            )

            # Get all the import statements from the initial code.
            import_statements = _get_import_statements(self.chart_code_list[0])
            # Post process all the codes in the for loop and create separate chart code files.
            for i in range(0, len(self.chart_code_list)):
                self.chart_code = self.chart_code_list[i]
                self.chart_code = self.chart_code.replace(
                    "metrics.json", self.track2_metrics_path
                )

                # Update the file names based on the iteration.
                # Suffix is not necessary for the first code.
                if i == 0:
                    track2_chart_path = self.track2_chart_path.replace("_<n>", "")
                    track2_code_file = "chartcode.py"
                    add_import_statements = None
                else:
                    track2_chart_path = self.track2_chart_path.replace("<n>", str(i))
                    track2_code_file = "chartcode_" + str(i) + ".py"
                    add_import_statements = import_statements

                # Post process the code
                # fs_key is used only when fs_connection_dict.platform is not None
                self.chart_code = clean_chart_code(
                    full_code_str=self.chart_code,
                    input_file_path=self.sql_results_path,
                    chart_save_path=track2_chart_path,
                    import_statements=add_import_statements,
                )
                # Add exception to the code
                self.chart_code = add_exception_to_code(
                    full_code_str=self.chart_code,
                    include_pattern=("fig."),
                    exclude_pattern=("fig.write_json"),
                )
                # Save the chart code
                self.logger.debug(f"Saving chart code to {self.track2_code_path}")

                # Format the code using Black.
                self.chart_code = black.format_str(self.chart_code, mode=black.Mode())

                # Format the code using isort.
                self.chart_code = isort.code(self.chart_code, float_to_top=True)

                self._save_outputs(
                    file_type="text",
                    output_folder=self.output_path,
                    file_name=track2_code_file,
                    content=self.chart_code,
                )

    @timing_decorator(track_app_start=False)
    def _run_chart_code(self, file_suffix):
        """
        The python code is executed. The code will create either plotly fig object or Metrics JSON.
        If the code has plotly fig object, it is saved as html and png files.
        Finally the JSON is returned from the function (Plotly fig JSON or Metrics JSON).

        Returns
        -------
        fig
            Figure Object / Metrics object - JSON
        """
        # Update the file names based on the iteration.
        # Suffix is not necessary for the first code.
        # set_current_track("track2")
        # log_stage_start("Time taken to run chart code","Track 2")
        if file_suffix == "0":
            track2_code_path = self.track2_code_path.replace("_<n>", "")
            track2_chart_path = self.track2_chart_path.replace("_<n>", "")
            chart_html_filename = "chart.html"
            chart_png_filename = "chart.png"
        else:
            track2_code_path = self.track2_code_path.replace("<n>", file_suffix)
            track2_chart_path = self.track2_chart_path.replace("<n>", file_suffix)
            chart_html_filename = "chart_" + file_suffix + ".html"
            chart_png_filename = "chart_" + file_suffix + ".png"

        # Run the chart code using subprocess.
        # subprocess_args = ["python", self.track2_code_path]

        # sys.executable is used as sometimes when setting env variables
        # python path is not being recognized
        subprocess_args = [sys.executable, "-c"]
        try:
            if cloud_config.cloud_provider != "s3":
                with self._fs.open(track2_code_path, "r") as f:
                    code = f.read()
            else:
                code = read_data(
                    track2_code_path,
                    cloud_config.domain_storage.account_name,
                )
            # print(code)
            # Sometimes LLM is hallucinating data (initializing some random data in the code itself) and using that for generating the graph. This is dangerous as this will silently produce incorrect answers
            # If it is a df hallucination, it's okay as long as it assigns the dataframe to the variable name `df` since we replace them in post processing with our data
            # If it is a list, it's hard because the chart trace code itself needs to change
            # We added some prompts to control this behaviour to avoid list hallucination. It might still happen sometimes. The following code is for the cases when the GPT hallucinates list even after the updated prompts
            # We identify if there are any static numeric lists in the code and stop chart generation if there is any
            if check_for_static_lists(code):
                self.logger.error(
                    "Static numeric lists are created in the chart code as a proxy for data."
                )
                raise ValueError(
                    "Static numeric lists should not be created in chart code."
                )
            subprocess_args.append(code)

            # Generate env dictionary from cloud storage parameters to be passed to subprocess as env variables
            # env will be None in case no cloud storage parameters are specified

            env = generate_env_dict(
                cloud_storage_dict=cloud_config,
                # account_key=None
                account_key=cloud_secrets.domain_storage.connection_key,  # os.getenv("BLOB_ACCOUNT_KEY"),
            )
            # subprocess output is of type - https://docs.python.org/3/library/subprocess.html#subprocess.CompletedProcess
            subprocess_result = subprocess.run(
                args=subprocess_args,
                text=True,
                capture_output=True,  # For populating stderr
                env=env,  # Sets environment variables to run the code
            )
            self.logger.debug(
                f"Chart code run return status = {subprocess_result.returncode}"
            )
            subprocess_result.check_returncode()
            # If code runs successfully, code will resume running below statements
            # yet to change
            if "metrics.json" in self.chart_code_list[int(file_suffix)]:
                with self._fs.open(self.track2_metrics_path, "r") as file:
                    metrics_json = file.read()
                # Parse the JSON data into a dictionary
                self.return_object = {"Chart Metrics": json.loads(metrics_json)}
            else:
                if cloud_config.cloud_provider != "s3":
                    with self._fs.open(track2_chart_path, mode="r") as fp:
                        self.return_object = plotly.io.read_json(fp)
                else:
                    s3_client = boto3.client("s3")
                    response = s3_client.get_object(
                        Bucket=cloud_config.domain_storage.account_name,
                        Key=track2_chart_path,
                    )
                    file_content = response["Body"].read().decode("utf-8")
                    json_data = io.StringIO(file_content)
                    self.return_object = plotly.io.read_json(json_data)

                self.logger.debug("Saving chart.")
                self._save_outputs(
                    file_type="plotly_fig",
                    output_folder=self.output_path,
                    file_name=[chart_html_filename, chart_png_filename],
                    content=self.return_object,
                )
        # If there is an error while running the chart code, log the error and raise an exception
        except subprocess.CalledProcessError as e:
            self.logger.error(
                f"Error while running the chart code. Error description:\n{e.stderr}"
            )
            self.return_object = None
            raise ValueError(e.stderr)
        except Exception as e:
            self.logger.error(
                f"Error while running the chart code. Error description:\n{e}"
            )
            self.return_object = None
            raise ValueError(e)
        # log_stage_end("Time taken to run chart code","Track 2")

        return self.return_object

    def _save_outputs(
        self, file_type: str, output_folder: str, file_name: str, content
    ) -> None:
        """
        Save the outputs in the respective folders based on file types.

        Args:
            file_type (str): The type of the file to be saved.
            output_folder (str): The path to the output folder where the file will be saved.
            file_name (str): The name of the file to be saved.
            content: The content to be saved in the file.

        Returns:
            None
        """
        self.logger.info(f"saving output {output_folder,file_type,file_name}")
        try:
            if file_type == "text":
                # save chart code suggestion
                if cloud_config.cloud_provider != "s3":
                    with self._fs.open(pp.join(output_folder, file_name), "w") as f:
                        f.writelines(content)
                else:
                    write_data(file_path=f"{output_folder}/{file_name}", content=content)
            if file_type == "json":
                self.logger.debug("Saving the current_message input to GPT.")
                if cloud_config.cloud_provider != "s3":
                    with self._fs.open(pp.join(output_folder, file_name), "w") as f:
                        json.dump(content, f, indent=4)
                else:
                    write_data(file_path=f"{output_folder}/{file_name}", content=content)
            elif file_type == "plotly_fig":
                # save chart figure as html
                # TODO: Add a comment indicating it should be a list with 0th index needs to be plotly and 1st is a png (in the place where this is called)
                if cloud_config.cloud_provider != "s3":
                    with self._fs.open(
                        pp.join(output_folder, file_name[0]), mode="w"
                    ) as fp:
                        plotly.io.write_html(content, file=fp)
                    with self._fs.open(
                        pp.join(output_folder, file_name[1]), mode="wb"
                    ) as fp:
                        # content.write_image(fp)
                        plotly.io.write_image(content, file=fp)
                else:
                    html_string = content.to_html(full_html=True)
                    write_data(
                        file_path=f"{output_folder}/{file_name[0]}", content=html_string
                    )
                    img_bytes = content.to_image(format="png")
                    write_data(
                        file_path=f"{output_folder}/{file_name[1]}", content=img_bytes
                    )
                    # s3_client.put_object(Bucket=bucket_name, Key=file_name, Body=html_string, ContentType='text/html')
        except Exception as e:
            self.logger.error(f"Error in saving data. Path: {output_folder} file: {file_name[1]} eror: {str(e)}")

    @timing_decorator(track_app_start=False)
    def _get_chart_object(self) -> list:
        """
        Run chart code(s) and return chart objects.

        Returns
        -------
        list
            Chart Object - JSON
        """
        # set_current_track("track2")
        # log_stage_start("API call for Track 2b","Track 2")
        self._chartcode_apicall()
        # log_stage_end("API call for Track 2b","Track 2")

        # set_current_track("track2")
        # log_stage_start("Post processing for Track 2b","Track 2")
        set_task_label("Track2b")
        self._post_processing(track="Track2b")
        # log_stage_end("Post processing for Track 2b","Track 2")
        self.logger.info(f"Running charts code. total charts to be created : {len(self.chart_code_list)}")
        chart_object = []
        for i in range(0, len(self.chart_code_list)):
            chart_object.append(self._run_chart_code(str(i)))
        return chart_object
    

    @timing_decorator(track_app_start=False)
    def _get_chart_object_locally(self) -> list:
        """
        Creating local chart code object (s) and return chart objects.
        Returns
        -------
        list
            Chart Object - JSON
        """ 
        self.chart_type_dict = self._clean_gpt_response(
            raw_response=self.chart_type_output,
            start_pattern=["<start>"],
            end_pattern=["<end>"],
        )
        self.chart_type_dict = json.loads(self.chart_type_dict)

        # loading chart data from s3
        self.df = self._load_chart_data_csv_from_s3()
        """To-Do remove the line 601 after the s2 output  missing data issue fixed """
        self.df = self.df.replace({np.nan: None})
        chart = self.generic_local_chart(self.df, self.chart_type_dict)
        """conveting chart details to plotly fig object 
                for saving chart.html and chart.png"""
        fig = go.Figure(chart)
        self._save_outputs(
                        file_type="plotly_fig",
                        output_folder=self.output_path,
                        file_name=["chart.html", "chart.png"],
                        content=fig,
                    )
        return [chart]

    @timing_decorator(track_app_start=False)
    def _get_chart_suggestion(self) -> None:
        """
        Model call and post processing for Chart code suggestion - Track 2b.

        Returns
        -------
        fig
            Chart Object - JSON
        """
        try:

            self.logger.info("Track 2a started.")
            start_time_track_2a = timeit.default_timer()
            # Getting the chart type suggestion.
            self._charttype_apicall()
            set_task_label("Track2a")
            self._post_processing(track="Track2a")
            # log_stage_end("Postprocessing Track 2a","Track 2")

            end_time_track_2a = timeit.default_timer()
            self.track_2a_runtime = round(end_time_track_2a - start_time_track_2a, 2)
            self.logger.info(
                f"Time taken for Track 2a(chart type suggestion): {round(end_time_track_2a - start_time_track_2a, 2)} seconds."
            )
            self.logger.info("Track 2a completed")

            self.logger.info("Track 2b started.")
            start_time_track_2b = timeit.default_timer()
            chart_object = None
            if self.user_config.chart_generation_type == 'llm': 
                try:
                    self.logger.info("Trying an iteration with the user question with llm call ")
                    chart_object = self._get_chart_object()
                except Exception as e:
                    error_string = f"Iteration with question returned an error. Error - {e}"
                    self.logger.info(error_string)
                    self.logger.info("Trying an iteration without the user question.")
                    self.question = None
                    self.additional_context = None
                    chart_object = self._get_chart_object()
            else:
                self.logger.info("Starting local chart generation")
                try:
                    chart_object = self._get_chart_object_locally()
                    self.logger.info("local chart generation Completed ")
                except Exception as e:
                    error_string = f"Expetion occured during local chart generation. Error - {e}"
                    self.logger.info(error_string)
                    self.logger.info("Trying llm chart generation after local chart generation failed")
                    try:
                        chart_object = self._get_chart_object()
                    except Exception as e:
                        error_string = f" Exception occure dunring llm chart generation. Error - {e}"
                        self.logger.info(error_string)
                        self.logger.info("Trying llm chart  without the user question.")
                        self.question = None
                        self.additional_context = None
                        chart_object = self._get_chart_object()

            end_time_track_2b = timeit.default_timer()
            self.track_2b_runtime = round(end_time_track_2b - start_time_track_2b, 2)
            # log_stage_end("Track 2b","Track 2")
            self.logger.info(
                f"Time taken for Track 2b(chart code generation): {round(end_time_track_2b - start_time_track_2b, 2)} seconds."
            )
            self.logger.info("Track 2b completed.")
        except Exception as e:
            error_msg = f"""
            Error while generating Chart Type/Code Suggestion, error: {e}
            """
            self.logger.error(error_msg)
            self.logger.error(f"Error Traceback:\n{traceback.format_exc()}")
            raise ValueError(error_msg)

        return chart_object

    @timing_decorator(track_app_start=False)
    def process_suggestion(self):
        """
        This is the main function which runs the Track 2 process in order.

        Returns
        -------
        fig
            Figure Object - JSON if it's a success.
            None - if Track 2 is skipped if the input table passed has just 1 row.
        """
        chart_object = None
        # Check if the output from Track 1 is suitable to generate a chart.
        # If the output table has only one row/scalar value, then it is not necessary to generate chart.
        if self.input_table.shape[0] <= 1:
            self.logger.info(
                "The output from Track 1 has just 1 row, which is not suitable to generate a chart."
            )
        # If the output table has only one column, check whether it is an ID or unique categorical column.
        elif self.input_table.shape[1] == 1:
            self.uniqueCategory_flag = _uniquecategory_check(self.input_table)
            # If it's an ID column or unique categorical column, can skip the histogram since it is not needed.
            if (("id" in self.data_dictionary["columns"][0].keys() 
            and self.data_dictionary["columns"][0]["id"] == "Yes") 
            or self.uniqueCategory_flag):
                self.logger.info(
                    """Output from track 1 has just one ID column or unique categorical column.
                    List or Table view should be appropriate."""
                )
            else:
                # If it's not an ID or unique categorical column, try creating a histogram.
                try:
                    df = self.input_table.copy()
                    if "description" in self.data_dictionary["columns"][0].keys():
                        description = self.data_dictionary["columns"][0]["description"]
                    else:
                        description = "Distribution"
                    # print(type(self.data_dictionary['columns'][0]['column_name']))
                    fig = px.histogram(
                        df, x=self.data_dictionary["columns"][0]["column_name"]
                    )
                    fig.update_layout(
                        xaxis_title=description.title(),
                        yaxis_title="Frequency",
                        title="Distribution of Values",
                    )

                    self.logger.info("converting the fig x axis data into list formate")
                    fig.data[0]['x'] = []
                    fig.data[0]['x'] = df[self.data_dictionary["columns"][0]["column_name"]].tolist()
                    self._save_outputs(
                        file_type="plotly_fig",
                        output_folder=self.output_path,
                        file_name=["chart.html", "chart.png"],
                        content=fig,
                    )
                    chart_object = [fig]
                except Exception as e:
                    error_msg = f"""Output from track 1 has just one column. List or Table view should be appropriate.
                    If the output has more rows, use the 'Export Output as csv' option.
                    Error while generating Chart Type/Code Suggestion, error: {e}
                    """
                    self.logger.error(error_msg)
                    self.logger.error(f"Error Traceback:\n{traceback.format_exc()}")
                    raise ValueError(error_msg)
        else:
            if self.skip_model:
                try:
                    self.logger.info(
                        "Question already exists in the Knowledge base. Extracting the code."
                    )
                    # Get the list of chart code files from the Track 2 path.
                    if cloud_config.cloud_provider != "s3":
                        file_list = self._fs.ls(self.output_path)
                    else:
                        file_list = self._fs.listdir(self.output_path)
                    chart_codes = sorted(
                        pp.basename(file)
                        for file in file_list
                        if pp.basename(file).startswith("chartcode")
                        and pp.basename(file).endswith(".py")
                    )

                    # Read the chart code(s) from the Track 2 path.
                    self.chart_code_list = []
                    for i in range(0, len(chart_codes)):
                        chart_code = chart_codes[i]

                        chartcode_path = pp.join(self.output_path, chart_code)
                        chartcode_str = read_text_file(chartcode_path, fs=self._fs)
                        # TODO: clean chart code function here is only relevant for old KB files. can be removed after KB is updated
                        if i == 0:
                            track2_chart_path = self.track2_chart_path.replace(
                                "_<n>", ""
                            )
                            track2_file_name = "chartcode.py"
                        else:
                            track2_chart_path = self.track2_chart_path.replace(
                                "<n>", str(i)
                            )
                            track2_file_name = f"chartcode_{i}.py"

                        if "df = pd.read_csv" not in chartcode_str:
                            self.logger.info(
                                "cleaning the chart code present in knowledge base"
                            )
                            # fs_key is used only when fs_connection_dict.platform is not None
                            chartcode_str = clean_chart_code(
                                chartcode_str,
                                self.sql_results_path,
                                chart_save_path=track2_chart_path,
                                import_statements=None,
                            )
                            self._save_outputs(
                                file_type="text",
                                output_folder=self.output_path,
                                file_name=track2_file_name,
                                content=chartcode_str,
                            )
                        self.chart_code_list.append(chartcode_str)

                    # Loop through the chart codes and get the final chart object (JSON).
                    chart_object = []
                    for i in range(0, len(self.chart_code_list)):
                        chart_object.append(self._run_chart_code(str(i)))
                    self.logger.info(
                        "Chart generated using existing code from Knowledge base."
                    )
                except Exception as e:
                    self.logger.info(
                        f"Error in retreiving details from Knowledge base. API Call will happen. Error - {e}"
                    )
                    chart_object = self._get_chart_suggestion()
            else:
                chart_object = self._get_chart_suggestion()
        if (not isinstance(chart_object, list)) and (chart_object is not None):
            chart_object = [chart_object]
        return chart_object

    def create_child_run_track2(self, parent_track2_run_id):
        self.logger.info("mlflow logging for track2 started")

        # Check if Chart Type Instance is available
        if hasattr(self, "query_to_charttype_ins") and self.query_to_charttype_ins:
            # Start a child run for 'query_to_chart_type'
            mlflow.set_experiment(self.experiment_name_mlflow)
            child_run = self.mlflow_manager.start_child_run(
                parent_track2_run_id, "query_to_chart_type"
            )
            # Log parameters and metrics
            self.mlflow_manager.log_param(
                child_run, parent_track2_run_id, "LLM Call Type", "query_to_chart_type"
            )
            self.mlflow_manager.log_param(
                child_run,
                parent_track2_run_id,
                "LLM Model",
                self.model_config.query_to_chart_type.model_params["engine"],
            )

            # Prepare artifacts to log
            self.mlflow_manager.log_artifact(
                child_run,
                parent_track2_run_id,
                "Prompt_for_LLM_Call.txt",
                self.query_to_charttype_ins.prompt,
            )
            self.mlflow_manager.log_artifact(
                child_run, parent_track2_run_id, "Result.txt", self.chart_type_output
            )
            self.mlflow_manager.log_artifact(
                child_run,
                parent_track2_run_id,
                "Result 1 post processing(code generated).txt",
                "",
            )
            self.mlflow_manager.log_artifact(
                child_run,
                parent_track2_run_id,
                "Result 2 post processing(charttype generated).txt",
                self.chart_type_output,
            )

            # Log metrics
            self.mlflow_manager.log_metric(
                child_run,
                parent_track2_run_id,
                "Prompt tokens",
                self.chart_type_tokens["prompt_tokens"],
            )
            self.mlflow_manager.log_metric(
                child_run,
                parent_track2_run_id,
                "Completion tokens",
                self.chart_type_tokens["completion_tokens"],
            )
            self.mlflow_manager.log_metric(
                child_run, parent_track2_run_id, "Run Time", self.track_2a_runtime
            )

            # Log status
            status = (
                "failure"
                if self.query_to_charttype_ins.error_message is not None
                else "success"
            )
            self.mlflow_manager.log_status(
                child_run,
                parent_track2_run_id,
                status,
                self.query_to_charttype_ins.error_message,
            )

        # Check if Chart Code Instance is available
        if hasattr(self, "query_to_chartcode_ins") and self.query_to_chartcode_ins:
            mlflow.set_experiment(self.experiment_name_mlflow)
            child_run = self.mlflow_manager.start_child_run(
                parent_track2_run_id, "query_to_chart_code"
            )

            # Log parameters and metrics
            self.mlflow_manager.log_param(
                child_run, parent_track2_run_id, "LLM Call Type", "query_to_chart_code"
            )
            self.mlflow_manager.log_param(
                child_run,
                parent_track2_run_id,
                "LLM Model",
                self.model_config.query_to_chart_code.model_params["engine"],
            )

            # Prepare artifacts to log
            self.mlflow_manager.log_artifact(
                child_run,
                parent_track2_run_id,
                "Prompt_for_LLM_Call.txt",
                self.query_to_chartcode_ins.prompt,
            )
            self.mlflow_manager.log_artifact(
                child_run, parent_track2_run_id, "Result.txt", self.chart_code_output
            )
            self.mlflow_manager.log_artifact(
                child_run,
                parent_track2_run_id,
                "Result 1 post processing(code generated).py",
                self.chart_code,
            )
            self.mlflow_manager.log_artifact(
                child_run,
                parent_track2_run_id,
                "Result 2 post processing.png",
                self.return_object,
            )

            # Log metrics
            self.mlflow_manager.log_metric(
                child_run,
                parent_track2_run_id,
                "Prompt tokens",
                self.chart_code_tokens["prompt_tokens"],
            )
            self.mlflow_manager.log_metric(
                child_run,
                parent_track2_run_id,
                "Completion tokens",
                self.chart_code_tokens["completion_tokens"],
            )
            self.mlflow_manager.log_metric(
                child_run, parent_track2_run_id, "Run Time", self.track_2b_runtime
            )

            # Log status
            status = (
                "failure"
                if self.query_to_chartcode_ins.error_message is not None
                else "success"
            )
            self.mlflow_manager.log_status(
                child_run,
                parent_track2_run_id,
                status,
                self.query_to_chartcode_ins.error_message,
            )

        self.logger.info("mlflow logging for track2 completed")
    

    @timing_decorator(track_app_start=False)
    def generic_local_chart(self, df, chart_type_dict):
        """
        Generates a chart based on the type specified.

        Parameters:
        - df: Pandas DataFrame
        - chart_type: Type of the chart (e.g., 'line', 'bar', 'scatter', etc.)
        - x_col: Column name for the x-axis
        - y_col: Column name for the y-axis
        chart_type, x_axis, y_axis,x_label,y_label,title
        calling the type of chart as requested by llm 
        """
        #fetching which type of bar chart it is
        #loading chart details 
        chart_type = chart_type_dict["chart_type"]
        if "horizontal" in chart_type.lower():
            orientation = 'h'
        else:
            orientation = 'v'
        if 'line' in chart_type.lower():
            if len(self.chart_type_dict["y_axis"]) == 1:
                chart_details = simple_line_chart(df, chart_type_dict, orientation)

        elif "bar" in chart_type.lower():
            """ need  to make a chake for n. of axis for y axis if secondary axis there"""
            if len(self.chart_type_dict["y_axis"]) > 1:
                if "dual-axis" in chart_type.lower():
                    chart_details = group_scatter_bar_chart(df, chart_type_dict, orientation)
                elif "group" in chart_type.lower():
                    chart_details = group_bar_chart(df, chart_type_dict, orientation)
                else:
                    chart_details = group_bar_chart(df, chart_type_dict, orientation)
            else:
                chart_details = simple_bar_chart(df, chart_type_dict, orientation)

        elif "scatter" in chart_type.lower():
            if len(self.chart_type_dict["y_axis"]) > 1:
                chart_details = group_scatter_chart(df, chart_type_dict, orientation)
            else:
                chart_details = simple_scatter_chart(df, chart_type_dict, orientation)

        else:
            raise ValueError(f"Functionality to generate Chart type: {chart_type} is not yet design")

        print("Local Chart details  successfully.")

        return chart_details

    def _load_chart_data_csv_from_s3(self):
        """loading chart data from s3 query to chart output table"""
        input_file_path = self.sql_results_path
        input_file_path = input_file_path.replace(os.path.sep, pp.sep)
        s3 = boto3.client('s3')
        bucket_name = cloud_config.domain_storage.account_name
        file_key = f"{input_file_path}"
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        csv_string = response['Body'].read().decode('utf-8')
        csv_data = StringIO(csv_string)
        df = pd.read_csv(csv_data) 

        return df
    
    def _clean_gpt_response(
        self,
        raw_response: str,
        start_pattern: list = ["<start>"],
        end_pattern: list = ["<end>"],
    ) -> str:
        """Cleans GPT response by trimming prefixes and suffixes

        Parameters
        ----------
        raw_response : str
            GPT response
        start_pattern : list, optional
            prefix to be removed, by default ["<start>"]
        end_pattern : list, optional
            suffix to be removed, by default ["<end>"]

        Returns
        -------
        str
            trimmed response
        """
        cleaned_response = extract_code(
            string_input=raw_response,
            start=start_pattern,
            end=end_pattern,
            extract="first",
        )
        if cleaned_response is None:
            if end_pattern[0] == "<end_dict>":
                cleaned_response = ""
            else:
                cleaned_response = raw_response

        return cleaned_response
