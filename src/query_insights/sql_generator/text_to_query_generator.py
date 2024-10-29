import json
import logging
import posixpath as pp
import re
import timeit
from typing import Tuple

import fsspec
import mlflow
import numpy as np
import pandas as pd
import sqlparse
from core.model.model_factory import ModelFactory
from core.utils.client_utils import get_model_type
from core.utils.read_config import cloud_config, db_config, model
from src.query_insights.utils.time_logging import timing_decorator
from src.query_insights.utils.utils import write_data
from sqlalchemy.engine import Connection
# log_stage_start,log_stage_end, set_current_track
from ..utils.post_processing import (
    _complete_data_dict,
    _detect_alter_drop_table,
    _extract_queries,
    _string_to_dict,
    _update_percent_sign,
    extract_code,
    round_float_numbers_in_dataframe,
)
from ..utils.utils import rate_limit_error_handler, read_text_file

# from ..benchmarking import MLflowManager

# import sys


# from .utils import MyLogger

MYLOGGERNAME = "QueryInsights"


class TextToQuery:
    """
    Track 1 - to generate SQL query from the user question

    Parameters
    ----------
    user_config : dict
        input user_config dictionary for storing and accessing user-specific configurations.
    model_config : dict
        input model_config dictionary for storing and accessing model-related configurations.
    data_config : dict
        input data_config dictionary contains the paths to the data
    debug_config: dict
        Debug config dictionary for using appropriate prompts to make requests for debugging to GPT.
    question : str
        User question to be answered
    additional_context : str
        Additional context to answer the question
    data_dictionary : dict
        contains table name, column name and description
    db_connection :
        to connect to SQL DB
    output_path : str
        path to save the results
    language : str
        Language to answer the question in, for example "english", "spanish", "german", by default "english"
    skip_model : bool
        condition whether to skip the api call.
    fs : fsspec.filesystem, optional
        Filesystem of the url, None will default to local file system, by default ``None``
    similarity: list
        [Boolean, (Question index, Similarity score, Question)]
        Boolean represents whether the user question is similar to any existing question.
        If First value is True, Second element (Tuple) will be populated. Otherwise it will have None values.
        Question index is the index of the similar question.
        Similarity score represents how similar the user question is to the existing question.
    """

    def __init__(
        self,
        user_config,
        model_config,
        data_config,
        debug_config,
        question,
        additional_context,
        data_dictionary,
        business_overview,
        bot_history,
        db_factory,
        foldercreation,
        similarity,
        language: str = "english",
        skip_model: bool = False,
        fs=None,
    ) -> None:
        """Class constructor"""
        # Config related
        self.prompt_dict = model_config.text_to_query.prompts
        self.model_param_dict = model_config.text_to_query.model_params
        self.model_param_dict["history"] = bot_history

        self.model_config = model_config
        self.data_config = data_config
        self.user_config = user_config
        self.connection_param_dict = user_config.connection_params
        self.text_to_query_debug_dict = debug_config.text_to_query
        self.ui = user_config.ui

        self.db_params = db_config

        # Business user query related
        self.question = question
        self.additional_context = additional_context
        self.business_overview = business_overview
        self.raw_data_dictionary = data_dictionary
        self.db_factory = db_factory
        self.output_path = foldercreation.output_path
        self.skip_model = skip_model
        self.similarity = similarity
        self.track1_runtime = 0.0

        # Logger
        self.logger = logging.getLogger(MYLOGGERNAME)

        # Mlflow experiment
        # self.experiment_name_mlflow = self.data_config.experiment_params.experiment_name

        # Init some instance vars to None
        self.output_table = None
        self.output_query = None
        self.output_table_dict = None
        self.text_to_query_ins = None
        self.query_model_output = None
        self.query_model_tokens = None
        self.query_error_message = None

        # SQL query debugging params
        self.sql_query_debug_retry_attempts = self.user_config.sql_debug_retry_attempts
        self.logger.info(
            f"Configuring text to query module with {self.sql_query_debug_retry_attempts} attempts."
        )

        # If language is None or empty string, default to "english" language
        if language is None or not bool(str(language).strip()):
            language = "english"
        language = language.lower().title()

        self.language = language

        self._fs = fs or fsspec.filesystem("file")
        self.prompt_dict["static_prompt_original"] = self.prompt_dict["static_prompt"]

        # Check if there is a similar question identified.
        if self.similarity[0]:
            # Create a filtered data dictionary based on the columns identified using the similar query.
            self.data_dictionary = {
                table_name: {
                    "table_name": table_data["table_name"],
                    "columns": [
                        {
                            key: value
                            for key, value in column.items()
                            if key != "similar"
                        }
                        for column in table_data["columns"]
                        if column.get("id") == "Yes" or column.get("similar") == "Yes"
                    ],
                }
                for table_name, table_data in self.raw_data_dictionary.items()
                if any(
                    column.get("similar") == "Yes" for column in table_data["columns"]
                )
            }
            # Update the prompt for similar questions and assign the similar question/response as the sample input
            self.prompt_dict["static_prompt"] = self.prompt_dict[
                "static_prompt_similar_question"
            ]
            sample_input = self.similarity[1]
        else:
            # Use the original data dictionary
            self.data_dictionary = self.raw_data_dictionary
            # No sample input since similarity is False
            sample_input = None
        self.track = "text_to_query"
        # llm model call
        config = dict()
        config["llm_model_type"] = model
        model_client = get_model_type(
            config,
            self.prompt_dict,
            self.question,
            self.additional_context,
            self.connection_param_dict,
            self.track,
            self.user_config,
            self.language,
            self.data_dictionary,
            self.business_overview,
            self.db_params,
            sample_input,
        )
        self.model_factory = ModelFactory(model_client)

        # Save prompts if we are going to run track 1.
        if not self.skip_model:
            self.logger.debug("Saving prompts.")
            # from io import BytesIO

            # import boto3

            # s3 = boto3.client("s3")
            # bucket_name = "takeda-ipro"
            # file_path = f"{self.output_path}/prompt.txt"
            # buffer = BytesIO(self.model_factory.model_type.prompt.encode("utf-8"))
            # s3.upload_fileobj(buffer, bucket_name, file_path)
            # print("promtttttt")
            if cloud_config.cloud_provider != "s3":
                with self._fs.open(pp.join(self.output_path, "prompt.txt"), "w") as f:
                    f.writelines(self.model_factory.model_type.prompt)
            else:
                write_data(
                    file_path=f"{self.output_path}/01_text_to_query/prompt.txt",
                    content=self.model_factory.model_type.prompt,
                )

        # Required for decorator
        time_delay = user_config.time_delay
        max_retries = model_config.text_to_query.model_params.max_tries

        # Normal way of using decorator as we are getting trouble passing arguments
        # in intended way of "@rate_limit_error_handler(...)"
        self._call_model_api = rate_limit_error_handler(
            logger=self.logger, time_delay=time_delay, max_retries=max_retries
        )(self._call_model_api)

        # self.mlflow_manager = MLflowManager(experiment_name=self.experiment_name_mlflow)
        return

    @timing_decorator(track_app_start=False)
    def _call_model_api(self, with_history: bool = False, history_kwargs: dict = None):
        """
        call_model_api
        Get model response from GPT model
        """
        self.logger.info("Track1 api model call started.")
        start_time_track_model_api = timeit.default_timer()
        if not with_history:
            (
                query_model_output,
                query_model_finish,
                query_model_tokens,
                query_error_message,
            ) = self.model_factory.model_response(self.model_param_dict)
        else:
            self.logger.debug("Requesting GPT to debug generated SQL.")
            (
                query_model_output,
                query_model_finish,
                query_model_tokens,
                query_error_message,
            ) = self.model_factory.model_response(**history_kwargs)
        end_time_track_model_api = timeit.default_timer()
        self.logger.info(
                f"Time taken for Track 1(api model call)): {round(end_time_track_model_api - start_time_track_model_api, 2)} seconds."
            )
        if (not self.skip_model) and hasattr(
            self.model_factory.model_type, "current_message"
        ):
            self.logger.debug("Saving the current_message input to GPT.")
            with self._fs.open(
                pp.join(self.output_path, "current_message_input.json"), "w"
            ) as f:
                json.dump(self.model_factory.model_type.current_message, f, indent=4)

        self.logger.info(
            f"Track 1:-\n finish token - {query_model_finish},\n token information - {query_model_tokens}"
        )
        self.logger.debug(f"Model output : \n{query_model_output}")
        return (
            query_model_output,
            query_model_finish,
            query_model_tokens,
            query_error_message,
        )

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

    def _clean_sql(self, raw_sql: str) -> str:
        """Post processing of GPT generated SQL

        Parameters
        ----------
        raw_sql : str
            SQL that needs to be cleaned

        Returns
        -------
        str
            cleaned SQL

        Raises
        ------
        pd.io.sql.DatabaseError
            when we encounter ALTER/CREATE SQL we raise this error
        """
        # Look for multiple SQL queries and check if all are starting with Select or with
        self.logger.debug("Checking for multiple SQL query.")
        self.queries_list = _extract_queries(raw_sql)
        self.logger.info(f"Number of SQL Queries present - {len(self.queries_list)}")

        # TODO: Look for alternate ways the SQL comments can come and handle in the post processing.
        # Once that is handled, the detect_alter_drop_table is redundant. Then the logic can be changed to
        # run the latest SELECT statement, even though ALTER/UPDATE statements are present.
        if _detect_alter_drop_table(raw_sql):
            error_msg = f"Execution failed on sql '{raw_sql}': ALTER or DROP TABLE query detected."
            self.logger.error(f"{error_msg}")

            # Assign this var in case of alter table
            self.alter_query = raw_sql

            raise pd.io.sql.DatabaseError(error_msg)
        # Check if all SQL Queries present are starting with SELECT or WITH
        elif len(self.queries_list) > 1 and all(item[1] for item in self.queries_list):
            self.logger.info(
                "Multiple select statements found. Extracting the latest Select query."
            )
            cleaned_query = self.queries_list[-1][0]
        else:
            self.logger.info(
                "Single SQL query present. No ALTER/UPDATE/DROP/CREATE statements found."
            )
            cleaned_query = raw_sql

        # Removing percent sign from the final executable query and executing the query.
        cleaned_query = _update_percent_sign(cleaned_query, update="remove")

        return cleaned_query

    def _sql_query_append_where_cols(self, sql_query: str) -> str:
        """
        Appending columns in SQL Query that are present in where clause.

        Parameters
        ----------
        sql_query : str
            SQL Query

        Returns
        -------
        str
            SQL query after appending columns that are present in where clause.
        """
        cols = extract_columns_in_where(sql_query)
        query_ls = sql_query.split("FROM")
        cols = [i for i in cols if i not in query_ls[0]]

        if len(cols) == 0:
            return sql_query
        else:
            query_ls[0] = query_ls[0] + ", " + ", ".join(cols)
            new_sql_query = query_ls[0] + " \nFROM" + query_ls[1]
        return new_sql_query

    def split_query_at_select(self, query: str) -> str:
        query_upper = (
            query.upper()
        )  # Convert the query to uppercase to make it case-insensitive
        select_keyword = "SELECT"
        select_positions = []

        # Find positions of each "SELECT" keyword in the query
        pos = query_upper.find(select_keyword)
        while pos != -1:
            select_positions.append(pos)
            pos = query_upper.find(select_keyword, pos + len(select_keyword))

        # Split the query at each "SELECT" position
        split_parts = []
        previous_position = 0

        for pos in select_positions:
            if previous_position < pos:
                split_parts.append(query[previous_position:pos].strip())
            previous_position = pos

        # Append the final part after the last SELECT
        if previous_position < len(query):
            split_parts.append(query[previous_position:].strip())

        return split_parts

    def find_where_clauses(self, query: str) -> str:
        """Find all WHERE clauses in the query."""
        where_clauses = re.findall(
            r"\bWHERE\b.*?(?=\bGROUP\b|\bORDER\b|\bHAVING\b|\bLIMIT\b|$)",
            query,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return where_clauses

    def extract_columns_from_where(self, where_clauses: str) -> list:
        """Extract column names from WHERE clauses."""
        columns = set()
        for clause in where_clauses:
            # Updated regex to capture column names with optional table aliases
            matches = re.findall(
                r"(\b\w+\.\w+|\b\w+\b)\s*(?:=|>|<|>=|<=|<>|!=|LIKE|IN|BETWEEN|IS\s+NULL)",
                clause,
                flags=re.IGNORECASE,
            )
            columns.update(matches)
        return list(columns)

    def add_columns_to_select(self, query: str) -> str:
        where_clauses = self.find_where_clauses(query=query)

        # Extract columns from the WHERE clauses
        columns = self.extract_columns_from_where(where_clauses=where_clauses)
        query_ls = query.split("FROM")
        columns = [i for i in columns if i not in query_ls[0]]
        if len(columns) == 0:
            new_sql_query = query
        elif len(columns) == 1:
            query_ls[0] = query_ls[0] + ", " + " ".join(columns)
            new_sql_query = query_ls[0] + " \nFROM" + query_ls[1]
        else:
            query_ls[0] = query_ls[0] + ", " + ", ".join(columns)
            new_sql_query = query_ls[0] + " \nFROM" + query_ls[1]
        return new_sql_query

    def _post_process_response(self, db_conn, raw_response: str) -> str:
        """Post processing that cleans the response to extract SQL and post process the SQL to be ready to run.

        Parameters
        ----------
        raw_response : str
            GPT response

        Returns
        -------
        str
            cleaned SQL and data dictionary
        """
        # Extract query and data dict from response
        # sometimes, response will also contain ``` instead of tags.
        output_query = self._clean_gpt_response(
            raw_response=raw_response,
            start_pattern=["```", "```python", "<start>", "```\n```","```sql"],
            end_pattern=["```", "<end>", "```\n```"],
        )

        output_table_dict = self._clean_gpt_response(
            raw_response=raw_response,
            start_pattern=["<start_dict>"],
            end_pattern=["<end_dict>"],
        )
        self.logger.info("SQL and data dict extracted.")

        cleaned_query = self.split_query_at_select(query=output_query)
        # Find all WHERE clauses
        final_query = ""
        for i in range(0, len(cleaned_query)):
            select_count = cleaned_query[i].lower().count("select")
            if select_count == 1 and "*" not in cleaned_query[i]:
                if i > 0:
                    p_query = cleaned_query[i - 1]
                    list1 = ["=", ">", "<", ">=", "<=", "<>", "!="]
                    if p_query[-7:-2].strip() != "JOIN" and p_query[-3] not in list1:
                        # print("jj")
                        new_sql_query = self.add_columns_to_select(cleaned_query[i])
                        final_query = final_query + " " + new_sql_query
                    else:
                        # print("cleaned_query[i]",cleaned_query[i])
                        final_query = final_query + " " + cleaned_query[i]
                else:
                    new_sql_query = self.add_columns_to_select(cleaned_query[i])
                    final_query = final_query + "" + new_sql_query
            else:
                final_query = final_query + "" + cleaned_query[i]

        final_opt_query = output_query  # self.optimize_query(final_query, db_conn)

        # save model response and SQL query before running so that we can use it for debugging if something errors out in the middle.
        self.logger.debug("Saving query.")
        # from io import BytesIO

        # import boto3

        # s3 = boto3.client("s3")
        # bucket_name = "takeda-ipro"
        # file_path_sql = f"{self.output_path}/01_text_to_query/sql_query.sql"
        # buffer = BytesIO(output_query.encode("utf-8"))
        # s3.upload_fileobj(buffer, bucket_name, file_path_sql)
        if cloud_config.cloud_provider != "s3":
            with self._fs.open(pp.join(self.output_path, "sql_query.sql"), "w") as f:
                f.writelines(final_query)
            with self._fs.open(
                pp.join(self.output_path, "optimized_sql_query.sql"), "w"
            ) as f:
                f.writelines(final_opt_query)
        else:
            write_data(
                file_path=f"{self.output_path}/01_text_to_query/sql_query.sql",
                content=final_query,
            )
            write_data(
                file_path=f"{self.output_path}/01_text_to_query/optimized_sql_query.sql",
                content=final_opt_query,
            )
        # with self._fs.open(pp.join(self.output_path, "sql_query.sql"), "w") as f:
        #     f.writelines(final_query)
        # with self._fs.open(
        #     pp.join(self.output_path, "optimized_sql_query.sql"), "w"
        # ) as f:
        #     f.writelines(final_opt_query)

        self.logger.debug("Saving model response.")
        if cloud_config.cloud_provider != "s3":
            with self._fs.open(
                pp.join(self.output_path, "model_response.txt"), "w"
            ) as f:
                f.writelines(raw_response)
        else:
            write_data(
                file_path=f"{self.output_path}/01_text_to_query/model_response.txt",
                content=raw_response,
            )

        # file_path_response = f"{self.output_path}/01_text_to_query/model_response.txt"
        # buffer = BytesIO(raw_response.encode("utf-8"))
        # s3.upload_fileobj(buffer, bucket_name, file_path_response)
        # with self._fs.open(pp.join(self.output_path, "model_response.txt"), "w") as f:
        #     f.writelines(raw_response)

        self.logger.debug("Cleaning SQL")
        final_opt_query = self._clean_sql(raw_sql=final_opt_query)

        return final_query, final_opt_query, output_table_dict

    def optimize_query(self, final_query, db_conn):

        replace_with = self.data_config.path.replace_in_query
        store_master = self.db_factory.execute_query(
            db_conn,
            "SELECT id, user_id, legacy_id, takeda_store_id, takeda_store_name FROM "
            + self.data_config.path.store_master,
        )
        user_master = self.db_factory.execute_query(
            db_conn,
            "SELECT user_id, first_name, last_name FROM "
            + self.data_config.path.user_master,
        )
        store_master = store_master.merge(user_master, on="user_id", how="inner")
        store_master["operator_name"] = (
            store_master["first_name"] + " " + store_master["last_name"]
        )
        store_master["takeda_store_name"] = store_master["takeda_store_name"].str.lower()
        store_master["operator_name"] = store_master["operator_name"].str.lower()

        # TODO: to be implemented based on database connection
        # if source!=='athena':
        #     store_id_map = dict(zip(store_master['legacy_id'].astype(str), store_master['id'].astype(str)))
        # else:
        #     store_id_map = dict(zip(store_master['legacy_id'].astype(str), store_master['global_store_id'].astype(str)))
        # for now replacing legacy_id with takeda_store_id
        store_id_map = dict(
            zip(
                store_master["takeda_store_id"].astype(str), store_master["id"].astype(str)
            )
        )  # remove this line after implementing above

        store_name_map = dict(
            zip(store_master["takeda_store_name"], store_master["id"].astype(str))
        )
        store_operator_map = dict(
            zip(store_master["operator_name"], store_master["id"].astype(str))
        )
        final_query = final_query.lower()
        final_query = sqlparse.format(final_query, keyword_case="upper")
        all_filters = []
        where_clauses = re.findall(
            rf"\bWHERE\b.*?(?=\bGROUP\b|\bSELECT\b|\bORDER\b|\bHAVING\b|\bLIMIT\b|$)",
            final_query,
            flags=re.IGNORECASE | re.DOTALL,
        )

        updated_where_clause = []
        for clause in where_clauses:
            s1 = clause.replace("WHERE", "")
            s1 = re.split(" AND | OR ", s1)
            for s in s1:
                for k in replace_with.keys():
                    if k in s:
                        all_filters.append(s)
            for filter in all_filters:
                before = filter
                if "store_id" in filter or "global_store_id" in filter:
                    filter = filter.replace("store_id", replace_with["store_id"])
                    rhs = (
                        re.split("=|LIKE|IN", filter)[1]
                        .replace("'", "")
                        .replace('"', "")
                    )
                    rhs = re.findall(r"\d+", rhs)
                    for store in rhs:
                        store = store.strip()
                        filter = filter.replace(str(store), str(store_id_map[store]))
                elif "store_name" in filter or "store_address" in filter:
                    filter = filter.replace("store_name", replace_with["store_name"])
                    rhs = (
                        re.split("=|LIKE|IN", filter)[1]
                        .replace("'", "")
                        .replace('"', "")
                    )
                    print(rhs)
                    rhs = rhs.strip()
                    filter = filter.replace(rhs, store_name_map[rhs])
                elif "operator_name" in filter:
                    filter = filter.replace(
                        "operator_name", replace_with["operator_name"]
                    )
                    rhs = (
                        re.split("=|LIKE|IN", filter)[1]
                        .replace("'", "")
                        .replace('"', "")
                        .strip()
                    )
                    rhs1 = rhs.replace("%", "").strip()
                    filter = filter.replace("LIKE", "IN").replace("=", "IN")
                    filter = filter.replace(
                        "'" + str(rhs) + "'", str(tuple(store_operator_map[rhs1]))
                    )
                clause = clause.replace(before, filter)
            updated_where_clause.append(clause)

        for i in range(len(where_clauses)):
            final_query = final_query.replace(where_clauses[i], updated_where_clause[i])

        print("Optimized Query:\n")
        print(final_query)
        return final_query

    def _post_process_output(self) -> None:
        """Post proess the output of GPT after running the SQL

        Raises
        ------
        ValueError
            if any of the column has invalid format.
        """
        # Replace string 'None' witn nan
        self.output_table = self.output_table.replace("None", np.nan)
        self.logger.info("Results from SQL is extracted.")

        if self.output_table.empty:
            error_msg = "No data is fetched while running the SQL.\nPlease change the user question or the data."
            self.logger.error(error_msg + "query:\n" + self.output_query)
            # raise ValueError(error_msg)

        # Post process data dictionary
        self.output_table_dict = _complete_data_dict(
            output_table=self.output_table,
            raw_dict=self.data_dictionary,
            result_dict=_string_to_dict(self.output_table_dict),
        )

        # Round float numbers to consistent format
        self.output_table = round_float_numbers_in_dataframe(self.output_table)

        # Identify numeric and datetime columns
        num_cols = []
        date_cols = []
        for col in self.output_table.columns:
            try:
                self.output_table[col] = pd.to_numeric(
                    self.output_table[col], errors="raise"
                )
                num_cols.append(col)
            except ValueError:
                # Identify datetime columns
                try:
                    self.output_table[col] = pd.to_datetime(
                        self.output_table[col], errors="raise"
                    )
                    date_cols.append(col)
                except ValueError:
                    pass
        print("-----------")
        print(self.output_table)
        return

    def _extract_sql_and_error(self, error_msg: str) -> Tuple[str, str]:
        """Based on error pattern, extracts the failed SQL and column name/error description

        Parameters
        ----------
        error_msg : str
            Error message for which we are extracting

        Returns
        -------
        Tuple[str, str]
            Error SQL and column name/error description
        """
        # Extract column that is invalid
        error_parts = error_msg.split(":")
        error_parts = [part.strip() for part in error_parts]

        # First part is SQL, capture everything that is between two single quotes.
        error_sql = re.findall(pattern=r"'([^']*)'", string=error_parts[0])

        # Second part is actual error message in case of 'no such column'
        # Final part is invalid column name with alias or actual error message for 'syntax error'
        final_part = error_parts[-1].split(".")[-1]

        return error_sql, final_part

    def _construct_debug_prompt_sql(
        self, debug_handler_dict: dict, error_msg: str
    ) -> str:
        """Constructs debug prompt for SQL.

        Parameters
        ----------
        debug_handler_dict : dict
            debug dictionary
        error_msg : str
            Actual error message for which we are debugging

        Returns
        -------
        str
            newly constructed debug prompt.
        """
        # Identify the error based on history
        error_under_consideration = None
        for error_type, error_params in debug_handler_dict.items():
            if error_params["error_pattern"] in error_msg:
                error_under_consideration = error_type
                break

        # error_under_consideration = "wrongerror"  # Test
        # Track 1 errors
        if error_under_consideration in [
            "error_invalid_column",
            "error_ambiguous_column",
            "error_unknown_column",
        ]:
            _, error_column_name = self._extract_sql_and_error(error_msg=error_msg)
            debug_prompt = debug_handler_dict[error_under_consideration][
                "debug_prompt"
            ].replace("<column_name>", error_column_name)
        elif error_under_consideration in ["error_syntax", "error_alter"]:
            _, _ = self._extract_sql_and_error(error_msg=error_msg)
            debug_prompt = debug_handler_dict[error_under_consideration]["debug_prompt"]
        else:
            raise ValueError(
                f"Below SQL debugging is not handled.\n\nSQL error description:\n{error_msg}"
            )

        return debug_prompt

    @timing_decorator(track_app_start=False)
    def _run_sql(self, db_conn):
        """
        post_processing
        Process the model result to extract the SQL query and the data dictionary to use them the following steps. Also fix data type issues in the code
        """
        try:
            # SQL execution block
            self.output_query, self.opt_query, self.output_table_dict = (
                self._post_process_response(
                    db_conn, raw_response=self.query_model_output
                )
            )
            print("****************************************")
            print(self.output_query)
            self.logger.info("Running SQL.")
            if hasattr(db_conn, "is_connected"):
                if not db_conn.is_connected():
                    self.logger.info("MySQL connection is closed. Reopening the connection.")
                    db_conn.reconnect()
                else:
                    self.logger.info("MySql connection is open. Reopening the connection. not nessasary.")

            elif isinstance(db_conn, Connection): # Check if the connection object is sqlalchemy type for Athena

                if db_conn.closed:
                    self.logger.info("SQLAlchemy connection is closed. Reopening the connection.")
                    db_conn = db_conn.engine.connect()
                else:
                    self.logger.info("SQLAlchemy connection is open Reopening the connection. not nessasary.")

            else:
                self.logger.info("Unknow connection  Type Opened. Reopening the connection. not nessasary.")

            self.output_table = self.db_factory.execute_query(db_conn, self.opt_query)

            # Check if the table has 0 rows. If so, add % to the LIKE if any and re-execute the query.
            if self.output_table.shape[0] == 0 and "LIKE" in self.opt_query:
                self.logger.info(
                    "The table returned has 0 rows. Trying another iteration by adding % to the LIKE paramater if any."
                )
                self.opt_query = _update_percent_sign(self.opt_query, update="add")
                self.output_table = self.db_factory.execute_query(
                    db_conn, self.opt_query
                )

        except pd.io.sql.DatabaseError as err_msg:
            err_msg = str(err_msg)  # Convert to string from error object.
            self.logger.error(
                f"Exception in executing SQL. Error description: {err_msg}"
            )
            if self.sql_query_debug_retry_attempts < 3:
                self.sql_query_debug_retry_attempts += 1
                self.logger.info(
                    f"Asking GPT to debug the SQL. Retry attempt : {self.sql_query_debug_retry_attempts}"
                )
                self._debug_sql(
                    err_msg, db_conn
                )  # calling this function to debug the generated sql
                self.logger.info("check the code")
            else:
                raise ValueError(
                    f"SQL could not be debugged within debug limits. Limits configured : {self.sql_query_debug_retry_attempts}"
                )

        except Exception as e:
            error_msg = f"""Error description - {e}
            Error occurred while running the GPT generated SQL for procuring the data.\nPlease change the user question or the data.
            """
            self.logger.error(error_msg + "query:\n" + self.output_query)
            raise ValueError(error_msg)

        finally:
            db_conn.close()
            self.logger.info("Closed Database Connection successfully.")
        if self.output_table.size > 10000:
            self.logger.info(f"Query received a table of size {self.output_table.size}. Trimming to top 50 rows.")
            self.output_table = self.output_table.head(50)
        self._post_process_output()

        return

    def _debug_sql(self, err_msg, db_conn):
        """
        Debug the sql generated in the SQL and try running again the sql.
        """

        debug_prompt = self._construct_debug_prompt_sql(
            debug_handler_dict=self.text_to_query_debug_dict, error_msg=err_msg
        )

        if hasattr(self, "alter_query"):  # Alter query detected
            history_kwargs = {
                "model_param_dict": self.model_param_dict,
                "debug_prompt": debug_prompt,
                "history": "Received error while running this SQL:\n"
                + self.alter_query,
            }
            new_prompt = (
                self.model_factory.model_type.prompt
                + "\n\n"
                + "Received error while running this SQL:\n"
                + self.alter_query
                + "\n\n"
                + debug_prompt
            )
        else:  # column not found and syntax error case.
            history_kwargs = {
                "model_param_dict": self.model_param_dict,
                "debug_prompt": debug_prompt,
                "history": "Received error while running this SQL:\n"
                + self.output_query,
            }
            new_prompt = (
                self.model_factory.model_type.prompt
                + "\n\n"
                + "Received error while running this SQL:\n"
                + self.output_query
                + "\n\n"
                + debug_prompt
            )

        with self._fs.open(pp.join(self.output_path, "prompt.txt"), "w") as f:
            f.writelines(new_prompt)

        if (not self.skip_model) and (
            hasattr(self.model_factory.model_type, "current_message")
        ):
            self.logger.debug("Saving the current_message input to GPT.")
            with self._fs.open(
                pp.join(self.output_path, "current_message_input.json"), "w"
            ) as f:
                json.dump(self.model_factory.model_type.current_message, f, indent=4)

        (
            self.query_model_output,
            self.query_model_finish,
            self.query_model_tokens,
            self.query_error_message,
        ) = self._call_model_api(with_history=True, history_kwargs=history_kwargs)

        self.logger.info("Received modified SQL after GPT debugging.")

        # here

        self._run_sql(db_conn)

        return

    @timing_decorator(track_app_start=False)
    def _save_outputs(self):
        """
        save_outputs
        Save the outputs in the respective folders
        """
        # save output table
        self.logger.debug("Saving table.")
        # import json
        # from io import BytesIO, StringIO

        # import boto3

        # s3_client = boto3.client("s3")
        # csv_buffer = StringIO()
        # self.output_table.to_csv(csv_buffer, index=False)
        # s3_file_path = f"{self.output_path}/01_text_to_query/output_table.csv"
        # s3_client.put_object(
        #     Bucket="takeda-ipro", Key=s3_file_path, Body=csv_buffer.getvalue()
        # )
        if cloud_config.cloud_provider != "s3":
            with self._fs.open(
                pp.join(self.output_path, "output_table.csv"), mode="wb", newline=""
            ) as fp:
                self.output_table.to_csv(fp, index=False)
        else:
            write_data(
                file_path=f"{self.output_path}/01_text_to_query/output_table.csv",
                content=self.output_table,
            )

        # save output table data dictionary
        self.logger.debug("Saving data dictionary.")
        if cloud_config.cloud_provider != "s3":
            with self._fs.open(
                pp.join(self.output_path, "output_data_dictionary.txt"), "w"
            ) as f:
                f.writelines(f"{self.output_table_dict}")
        else:
            write_data(
                file_path=f"{self.output_path}/01_text_to_query/output_data_dictionary.txt",
                content=self.output_table_dict,
            )
        # file_path_data_dict = (
        #     f"{self.output_path}/01_text_to_query/output_data_dictionary.txt"
        # )

        # # print(self.output_table_dict)
        # buffer = StringIO()
        # json.dump(self.output_table_dict, buffer, indent=4)
        # buffer.seek(0)
        # # buffer = BytesIO(self.output_table_dict.encode("utf-8"))
        # s3_client.put_object(
        #     Bucket="takeda-ipro", Key=file_path_data_dict, Body=buffer.getvalue()
        # )
        # with self._fs.open(
        #     pp.join(self.output_path, "output_data_dictionary.txt"), "w"
        # ) as f:
        #     f.writelines(f"{self.output_table_dict}")

        return

    @timing_decorator(track_app_start=False)
    def get_query_suggestion(self, db_conn):
        """
        This is the main function call for SQL query and table generation.

        Flow -
            1. If it's the same question from KB, model response is retrieved from KB folders.
                - If it results in an error, normal API call happens.
            2. If the user question is similar to existing question in KB, model call happens with modified prompt.
                - If it results in an error, normal API call happens.
            3. In case of a new unrelated question, normal API call happens.

        Returns:
            None
        """
        # set_current_track("track1")
        # log_stage_start("Checks if the question present in KB or a similar question", "Track1")
        if False:  # self.skip_model:
            try:
                log_stage_start("Process Existing Question", "Track1")
                self.logger.info(
                    "Question already exists. Getting SQL from knowledge base."
                )
                model_response = pp.join(self.output_path, "model_response.txt")
                self.query_model_output = read_text_file(model_response, fs=self._fs)
                self.logger.info("SQL Query retreived from Knowledge Base.")
                log_stage_end("Process Existing Question", "Track1")
                self.process_query(skip_api_call=True)
            except Exception as e:
                self.logger.info(
                    f"Knowledge Retrieval failed with error {e}. Making the GPT call again."
                )
                self.process_query()
        elif False:  # self.similarity[0]:
            try:
                log_stage_start("Running GPT -  Similar question", "Track1")
                self.logger.info(
                    "Running GPT request for SQL generation for similar question."
                )
                log_stage_end("Running GPT -  Similar question", "Track1")
                self.process_query()
            except Exception as e:
                log_stage_start("Running GPT -  Similar question", "Track1")
                self.logger.info(f"Similar question api call ended with error - {e}.")
                self.logger.info("Running the usual GPT request for SQL generation.")
                # Updaing the static prompt to the usual one.
                self.prompt_dict["static_prompt"] = self.prompt_dict[
                    "static_prompt_original"
                ]
                # Passing the raw data dictionary without sample input.
                config = dict()
                config["llm_model_type"] = model
                model_client = get_model_type(
                    config,
                    self.prompt_dict,
                    self.question,
                    self.additional_context,
                    self.connection_param_dict,
                    self.track,
                    self.user_config,
                    self.language,
                    self.db_params,
                    self.data_dictionary,
                    self.business_overview,
                    sample_input=None,
                )
                self.model_factory = ModelFactory(model_client)

                self.process_query()
        else:  # Normal run
            # log_stage_start("Get query Suggestion", "Track1")
            self.process_query(db_conn)
        # log_stage_end("Checks if the question present in KB or a similar question", "Track1")
        return

    @timing_decorator(track_app_start=False)
    def process_query(self, db_conn, skip_api_call: bool = False):
        """Generates SQL from GPT and executes it. Returns the output table and its data dictionary if SQL execution yielded results.
        GPT call call can be skipped using the skip_api_call parameter if self.query_model_output is obtained from KB (applicable for same questions in KB).

        Parameters
        ----------
        skip_api_call : boolean (Default - False)
            Skips the GPT api call if this parameter is True.

        Returns:
            None
        """
        start_time_to_call_api_for_sql = timeit.default_timer()
        if not skip_api_call:
            # set_current_track("track1")
            self.logger.info("Calling the API for SQL generation.")
            (
                self.query_model_output,
                self.query_model_finish,
                self.query_model_tokens,
                self.query_error_message,
            ) = self._call_model_api()

        end_time_to_call_api_for_sql = timeit.default_timer()

        self.logger.info(
            f"Time taken to generate the SQL: {round(end_time_to_call_api_for_sql - start_time_to_call_api_for_sql, 2)} seconds."
        )
        start_time_to_run_sql = timeit.default_timer()
        self.logger.info("Executing the SQL to generate output.")
        # set_current_track("track1")
        self._run_sql(db_conn)
        end_time_to_run_sql = timeit.default_timer()
        self.logger.info(
            f"Time taken to execute the SQL: {round(end_time_to_run_sql - start_time_to_run_sql, 2)} seconds."
        )
        start_time_to_save_outputs = timeit.default_timer()
        self.logger.info("Saving the output to predefined paths.")
        # set_current_track("track1")
        # log_stage_start("Saving the output to predefined paths.", "Track1")
        self._save_outputs()
        end_time_to_save_outputs = timeit.default_timer()
        # log_stage_end("Saving the output to predefined paths.", "Track1")
        self.logger.info(
            f"Time taken to save the output table: {round(end_time_to_save_outputs - start_time_to_save_outputs, 2)} seconds."
        )
        # end_time_to_call_api_for_sql = start_time_to_call_api_for_sql = timeit.default_timer()
        self.track1_runtime = (
            round(end_time_to_call_api_for_sql - start_time_to_call_api_for_sql, 2)
            + round(end_time_to_run_sql - start_time_to_run_sql, 2)
            + round(end_time_to_save_outputs - start_time_to_save_outputs, 2)
        )

        self.logger.info("Text to Query is completed.")
        return


class BotResponse:
    def __init__(
        self,
        user_config=None,
        model_config=None,
        conversation_history=None,
        error_message=None,
        skip_model=False,
        mode="rule_based",
        language: str = "english",
    ):
        if mode == "model_based":
            self.prompt_dict = model_config.bot_response.prompts
            self.model_param_dict = model_config.bot_response.model_params
            self.connection_param_dict = user_config.connection_params
            self.ui = user_config.ui
            self.conversation_history = ""
            for conv in conversation_history[:-1]:
                self.conversation_history += f"user: {conv[0]}\n"
                self.conversation_history += f"bot: {conv[1]}\n"
            self.conversation_history += f"user: {conversation_history[-1][0]}"
            self.initialize_model_attr()
            # Required for decorator
            time_delay = user_config.time_delay
            max_retries = model_config.text_to_query.model_params.max_tries

            # Normal way of using decorator as we are getting trouble passing arguments
            # in intended way of "@rate_limit_error_handler(...)"
            self._call_model_api = rate_limit_error_handler(
                logger=self.logger, time_delay=time_delay, max_retries=max_retries
            )(self._call_model_api)

            # If language is None or empty string, default to "english" language
            if language is None or not bool(str(language).strip()):
                language = "english"
            self.language = language.lower().title()

            # Required for decorator
            time_delay = user_config.time_delay
            max_retries = model_config.text_to_query.model_params.max_tries

            # Normal way of using decorator as we are getting trouble passing arguments
            # in intended way of "@rate_limit_error_handler(...)"
            self._call_model_api = rate_limit_error_handler(
                logger=self.logger, time_delay=time_delay, max_retries=max_retries
            )(self._call_model_api)

        self.error_message = error_message

        self.skip_model = skip_model

        self.logger = logging.getLogger(MYLOGGERNAME)

        return

    def _call_model_api(self):
        """
        call_model_api
        Get model response from GPT model
        """
        self.logger.debug("call model api reached for bot response")
        config = dict()
        config["llm_model_type"] = model
        model_client = get_model_type(
            config,
            self.prompt_dict,
            question=None,
            additional_context=None,
            connection_param_dict=self.connection_param_dict,
            track=self.track,
            user_config=self.user_config,
            language=self.language,
            data_dictionary=None,
            business_overview=None,
            db_param_dict=self.db_params,
            sample_input=None,
            code_result=None,
            table=None,
            history=self.conversation_history,
            error_message=self.error_message,
        )
        self.model_factory = ModelFactory(model_client)
        (
            self.bot_response_output,
            self.bot_response_finish,
            self.bot_response_tokens,
            error_message,
        ) = self.model_factory.model_response(self.model_param_dict)
        self.bot_response_output = self.bot_response_output.replace("\n", "")
        self.logger.info(
            f"Track 1:-\n finish token - {self.bot_response_finish},\n token information - {self.bot_response_tokens}"
        )
        self.logger.debug(f"Model output\n{self.bot_response_output}")

    def process_sql_error(self):
        """calls the API to generate appropriate bot response when the generated SQL query fails.

        Returns:
            None
        """

        if self.skip_model:
            try:
                self.logger.info(
                    "Question already exists. Getting bot response from from knowledge base."
                )
                # TODO: handle skip model
                # model_response = os.path.join(self.output_path, "model_response.txt")
                # self.query_model_output = read_text_file(model_response)
                # print(self.query_model_output)
                # self.logger.info("SQL Query retreived from Knowledge Base.")
            except Exception as e:
                self.skip_model = False
                self.logger.info(
                    f"Knowledge Retrieval failed with error {e}. Making the GPT call again."
                )
                self._call_model_api()
        else:
            self.logger.info("Making the GPT request for bot response generation.")
            self._call_model_api()

        # self.logger.info("Saving the output to predefined paths.")
        # self._save_outputs()

        self.logger.info("Bot response generation is completed.")
        return

    def extract_key_error_from_message(self, error_message, key_phrase):
        """
        To extract the column/table/funtion name that caused the SQL error

        Parameters
        ----------
        error_message : str
            contains the error message genrated by pandas read_sql_query
        key_phrase : str
            contains the phrase (like: no such column, ambiguous column name) that can be used for splitting the string

        Returns
        -------
        str
            column/table/funtion name that caused the error
        """
        return (
            error_message.split(key_phrase)[1]
            .strip()
            .split(" ")[0]
            .strip()
            .split(".")[-1]
        )

    def get_bot_error_message(self, error_message):
        """
        Generates the bot response without any model using some custom messages for each error.

        Parameters
        ----------
        error_message : str
            contains the error message genrated by pandas read_sql_query

        Returns
        -------
        str
            custom bot response specific to the given error
        """
        error_message = error_message.replace("\n", " ")

        # OperationalError: no such column: <column_name>: This error occurs when the specified column does not exist in the table.
        if ": no such column:" in error_message:
            error = self.extract_key_error_from_message(
                error_message, ": no such column:"
            )
            return f"I didn't get what you mean by this column: '{error}'. Help me in understand where to get it."

        # OperationalError: no such table: <table_name>: This error indicates that the specified table does not exist in the database.
        elif ": no such table:" in error_message:
            error = self.extract_key_error_from_message(
                error_message, ": no such table:"
            )
            return f"I couldn't find the table: '{error}'. Help me understand which table to use."

        # OperationalError: near "<syntax_error>": syntax error: This error indicates a syntax error in the SQL query, where <syntax_error> represents the specific syntax element that caused the issue.
        elif ": syntax error:" in error_message:
            return "Looks like there is a syntax error. Please check the below query and help me correct it."

        # OperationalError: ambiguous column name: <column_name>: This error occurs when the specified column name is ambiguous and exists in multiple tables referenced in the query. To resolve this, you can specify the table name or alias along with the column name in the query to disambiguate it.
        elif ": ambiguous column name:" in error_message:
            error = self.extract_key_error_from_message(
                error_message, ": ambiguous column name:"
            )
            return f"The column: '{error}' is present in more than one table. Help me identify which one to use."

        # OperationalError: no such function: <function_name>: This error occurs when the query references a function that does not exist in SQLite. Make sure that the function name is spelled correctly and that the function is supported by SQLite.
        elif ": no such function:" in error_message:
            error = self.extract_key_error_from_message(
                error_message, ": no such function:"
            )
            return (
                f"The fuction: '{error}' is not found. Can you ask me something else?"
            )

        # OperationalError: unrecognized token: <token>: This error indicates that the query contains an unrecognized or invalid token. Check the query syntax and ensure that all keywords, operators, and identifiers are correctly specified.
        elif ": unrecognized token:" in error_message:
            error = self.extract_key_error_from_message(
                error_message, ": unrecognized token:"
            )
            return f"The token: '{error}' is invalid. Check the query syntax and try rephrasing the question."

        # OperationalError: No data is fetched while running the SQL query
        elif "No data is fetched" in error_message:
            return "No data fetched. Please check the filters in the SQL query below and suggest any changes (if required)."

        else:
            return None
        # OperationalError: table <table_name> has no column named <column_name>: This error message indicates that the specified table does not have a column with the given name. Double-check the column name for typos or verify the table schema to ensure the column exists.
        # OperationalError: too many terms in compound SELECT: This error message suggests that the compound SELECT statement has an excessive number of terms, exceeding the limits set by SQLite. Review and simplify the compound SELECT statement to resolve this error.
        # OperationalError: database is locked: This error occurs when multiple processes or threads are attempting to access the database simultaneously and one of them has a lock on the database. It indicates a concurrency issue and typically resolves when the lock is released.
        # OperationalError: no such module: <module_name>: This error message suggests that the specified SQLite module is not available or not installed. SQLite allows the use of external modules, and this error occurs when trying to access a module that is not present.
        # OperationalError: disk I/O error: This error indicates a problem with the disk I/O operations, such as reading or writing to the database file. It can occur due to disk failures, lack of disk space, or file permission issues.
        # ProgrammingError: Incorrect number of bindings supplied: This error occurs when the number of provided parameter bindings in the SQL query does not match the number of placeholders. Ensure that the number of bindings matches the number of placeholders in the query.
        # ProgrammingError: Incorrect type of bindings supplied: This error indicates that the data types of the provided parameter bindings do not match the expected data types in the query. Ensure that the data types of the bindings match the corresponding placeholders in the query.
        # DatabaseError: file is encrypted or is not a database: This error occurs when the specified file is either encrypted or not a valid SQLite database file.
        # DatabaseError: unable to open database file: This error typically indicates that the specified database file cannot be opened, either due to incorrect file path or insufficient permissions.
        # DatabaseError: unable to open database file: This error typically indicates that the specified database file cannot be opened, either due to incorrect file path or insufficient permissions.
