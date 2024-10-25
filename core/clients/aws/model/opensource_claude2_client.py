import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from core.utils.read_config import db_config
from core.model.model_client import Model
from dynaconf import Dynaconf
import timeit
import openai
import boto3

from src.query_insights.utils.utils import parse_today_date

# import logging


# MYLOGGERNAME = "QueryInsights"


class OpenSourceClaude2ModelCall(Model):
    """
    A class that generates prompts and performs insights generation using Mixtral(LLM model).

    Parameters
    ----------
    prompt_dict : dict
        A dictionary containing the prompt template and other static content.
    question : str
        A string representing the user's question.
    additional_context : str
        A string representing additional context to include in the prompt.
    connection_param_dict : dict
        A dictionary containing the parameters needed for the Azure's OpenAI API.
    user_config : dict
        Input user_config dictionary for storing and accessing user-specific configurations.
    language : str
        Language to answer the question in, for example "english", "spanish", "german", by default "english"
    db_param_dict : dict
        A dictionary containing the database information
    dictionary : dict, optional
        A dictionary representing the data dictionary of the data specified in the config file. Defaults to None.
    suggestion : str, optional
        A string representing a suggestion for the user. Defaults to None.
    table : str, optional
        A string representing a table to include in the prompt. Defaults to None.
    sample_input: list [sample_question, sample_response], optional
        A list having a sample question and response for GPT's reference

    Attributes
    ----------
    api_key : str
        A string representing the OpenAI API key.
    dictionary : str
        A dictionary  representing the data dictionary of the data specified in the config file.
    prompt_dict : dict
        A dictionary containing the prompt template and other static content.
    question : str
        A string representing the user's question.
    additional_context : str
        A string representing additional context to include in the prompt.
    suggestion : str
        A string representing a suggestion for the user.
    table : str
        A string representing a table to include in the prompt.
    connection_param_dict : dict
        A dictionary containing the parameters needed for the Azure's OpenAI API.
    prompt : str
        A string representing the final prompt.
    db_param_dict : dict
        A dictionary containing the database information
    language : str
        Language to answer the question in

    Methods
    -------

    generate_prompt():
        Generates the final prompt from the user inputs.

    model_response(model_param_dict):
        Performs insights generation using the OpenAI API and returns the output.

    extract_code(string_input, start, end):
        Extracts code from the ourput of OpenAI API response.
    """

    def __init__(
        self,
        prompt_dict,
        question,
        additional_context,
        connection_param_dict,
        track: str,
        user_config: dict,
        language: str = "english",
        db_param_dict=None,
        dictionary=None,
        business_overview=None,
        suggestion=None,
        table=None,
        history=None,
        error_message=None,
        sample_input=None,
    ):
        super().__init__(
            prompt_dict,
            question,
            additional_context,
            connection_param_dict,
            track,
            user_config,
            language,
            db_param_dict,
            dictionary,
            business_overview,
            suggestion,
            table,
            history,
            error_message,
            sample_input,
        )
        self.model_dict = {
            "text_to_query": user_config.connection_params.api_type.text_to_query,
            "query_to_chart_type": user_config.connection_params.api_type.query_to_chart_type,
            "query_to_chart_code": user_config.connection_params.api_type.query_to_chart_code,
            "table_to_insight_questions": user_config.connection_params.api_type.table_to_insight_questions,
            "insight_questions_to_code": user_config.connection_params.api_type.insight_questions_to_code,
            "summarize_tables": user_config.connection_params.api_type.summarize_tables,
        }
        self.generate_prompt()
        self.set_connection_params()

    def set_connection_params(self):
        """
        Set the Azure's OpenAI API connection parameters.

        Parameters
        ----------
        self : GPTModelCall
            An instance of the GPTModelCall class.

        Raises
        ------
        KeyError
            If the 'api_type' key is not found in the connection_param_dict dictionary.

        Returns
        -------
        None
        """
        try:
            if self.connection_param_dict["api_type"] == "azure":
                openai.api_type = self.connection_param_dict["api_type"]
                openai.api_base = self.connection_param_dict["api_base"]
                openai.api_version = self.connection_param_dict["api_version"]
            elif self.connection_param_dict["api_type"] == "aws-bedrock":
                self.bedrock = boto3.client(
                    service_name="bedrock-runtime", region_name="us-east-1"
                )
        except KeyError as e:
            raise KeyError(
                f"""An error occurred during the setting the connection parameters.
                                The error message is: {e}."""
            )

    def generate_prompt(self):
        try:
            static_prompt = self.prompt_dict.static_prompt.replace(
                "Question:", "Human:"
            )
            # print("#############",static_prompt)
            # static_prompt = f"{static_prompt}\nAssistant:"
            # print("**************",static_prompt)
            self.prompt_dict["static_prompt"] = static_prompt
            prompt = self.prompt_dict["static_prompt"].replace(
                "<date>", parse_today_date(self.user_config.today)
            )
            if self.sample_question is not None:
                if self.connection_param_dict["api_type"] == "aws_claude_2":
                    prompt = prompt.replace(
                        "<sample_question>", f"Human: {self.question}\nAssistant:"
                    )
                    prompt = prompt.replace("Question: ", "")
            else:
                prompt = "\n".join(
                    [
                        line
                        for line in prompt.split("\n")
                        if "<sample_question>" not in line
                    ]
                )
            if self.sample_response is not None:
                prompt = prompt.replace(
                    "<sample_response>", f"{self.sample_response}"
                )
            else:
                prompt = "\n".join(
                    [
                        line
                        for line in prompt.split("\n")
                        if "<sample_response>" not in line
                    ]
                )
            if self.question is not None:
                prompt = prompt.replace("<question>", f"{self.question}")
            else:
                prompt = "\n".join(
                    [
                        line
                        for line in prompt.split("\n")
                        if "<question>" not in line
                    ]
                )
            if self.additional_context is not None:
                prompt = f"{prompt}\n{self.prompt_dict['additional_context']}"
                prompt = prompt.replace(
                    "<additional_context>", f"{self.additional_context}"
                )
            if self.suggestion is not None:
                prompt = prompt.replace("<suggestion>", f"{self.suggestion}")

            prompt = f"{prompt}\n{self.prompt_dict['guidelines']}"
            if self.db_param_dict is not None:
                if db_config.domain_database is None:
                    # If db_name is None, default to MySql
                    sql_string = "MySql"
                else:
                    # Else sql string will be same as specified in config
                    sql_string = db_config.domain_database
                prompt = prompt.replace("<sql>", sql_string)

            data_dict_json_str = json.dumps(self.dictionary, indent=4)
            prompt = prompt.replace("<data_dictionary>", f"{data_dict_json_str}")

            if self.table is not None:
                prompt = prompt.replace("<table>", self.table)
            if self.history is not None:
                prompt = prompt.replace("<history>", self.history)
            if self.error_message is not None:
                prompt = prompt.replace("<error_message>", self.error_message)
            # if bool(self.prompt_dict['additional_context']):
            #    prompt = prompt + f"\n{self.prompt_dict['additional_context']}"
            if self.business_overview is not None:
                prompt = f"{prompt}\n{self.prompt_dict['business_overview']}"
                prompt = prompt.replace(
                    "<business_overview>", f"{self.business_overview}"
                )
            self.prompt = prompt
            self.logger.info(f"Prompt generated successfully.")

        except KeyError as e:
            raise KeyError(
                f"An error occurred during the creating the prompt. The error message is: {e}."
            )

    def model_response(
        self, model_param_dict: dict, debug_prompt: str = None, history: str = None
    ):
        output = ""
        finish_reason = ""
        tokens = dict()

        #api_type = self.connection_param_dict["api_type"]
        api_type = self.model_dict[self.track]

        if api_type == "aws_claude_2":
            if history is not None:
                self.prompt = self.prompt + "\n\n"
                self.prompt += f"claude response:\n\n{history}\n\n"
                self.prompt += f"New Question: {debug_prompt}"

            bedrock = boto3.client(
                service_name="bedrock-runtime", region_name="us-east-1"
            )
            #print(self.prompt)
            body = json.dumps(
                {
                    "prompt": self.prompt,
                    "anthropic_version": model_param_dict.anthropic_version,  # "bedrock-2023-05-31"
                    "max_tokens_to_sample": model_param_dict.max_tokens_to_sample,  # 300,
                    "temperature": model_param_dict.temperature,  # 0.5,
                    "top_k": model_param_dict.top_k,  # 250,
                    "top_p": model_param_dict.top_p,  # 1,
                }
            )

            api_response = bedrock.invoke_model(
                modelId=model_param_dict.engine,
                contentType="application/json",
                accept="application/json",
                body=body,
            )
            self.logger.info(
                f"API Response error code : {api_response['ResponseMetadata']['HTTPStatusCode']}"
            )

            response = json.loads(api_response.get("body").read())
            prompt_tokens = int(
                api_response["ResponseMetadata"]["HTTPHeaders"][
                    "x-amzn-bedrock-input-token-count"
                ]
            )
            completion_tokens = int(
                api_response["ResponseMetadata"]["HTTPHeaders"][
                    "x-amzn-bedrock-output-token-count"
                ]
            )
            total_tokens = prompt_tokens + completion_tokens
            tokens_dict = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
            settings = Dynaconf(settings_files=[], environments=True, load_dotenv=False)
            settings.set('MY_DICT', tokens_dict)
            tokens = settings.MY_DICT
            output = response["completion"]
            finish_reason = response["stop_reason"]
            # tokens = {"total_tokens": 0}  # response["usage"]["output_tokens"]}

            self.logger.info(f"Generated Resposne : {output}")

        return output, finish_reason, tokens, None
            
            