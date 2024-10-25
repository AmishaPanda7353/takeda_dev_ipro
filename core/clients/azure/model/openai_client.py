import json
import traceback

import openai
from openai import AzureOpenAI

from core.model.model_client import Model
from core.utils.read_config import db_config
from src.query_insights.utils.utils import (SensitiveContentError, TimeoutError,
                                          TokenLimitError, parse_today_date,
                                          timeout)

MYLOGGERNAME = "QueryInsights"


class GPTModelCall(Model):
    """
    A class that generates prompts and performs insights generation using OpenAI.

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
    conversation_questions : str, optional
        List of recent user question.

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
    set_connection_params():
        Sets the connection parameters for the Azure's OpenAI API.

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
            api_type = self.model_dict[self.track]
            if api_type == "openai":
                self.client = AzureOpenAI(
                    azure_endpoint=self.connection_param_dict["api_base"],
                    api_key=self.api_key,
                    api_version=self.connection_param_dict["api_version"],
                )
        except KeyError as e:
            raise KeyError(
                f"""An error occurred during the setting the connection parameters.
                                The error message is: {e}."""
            )

    def generate_prompt(self):
        """
        Generate the final prompt from the user inputs.

        Parameters
        ----------
        self : GPTModelCall
            An instance of the GPTModelCall class.

        Raises
        ------
        KeyError
            If the key is not found in the prompt_dict dictionary.

        Returns
        -------
        None
        """
        try:
            prompt = self.prompt_dict["static_prompt"].replace(
                "<date>", parse_today_date(self.user_config.today)
            )
            if self.sample_question is not None:
                prompt = prompt.replace("<sample_question>", f"{self.sample_question}")
            else:
                prompt = "\n".join(
                    [
                        line
                        for line in prompt.split("\n")
                        if "<sample_question>" not in line
                    ]
                )
            if self.sample_response is not None:
                prompt = prompt.replace("<sample_response>", f"{self.sample_response}")
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
                    [line for line in prompt.split("\n") if "<question>" not in line]
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
            # self.logger.info(f"PROMPT:\n\n{prompt}\n")
        except KeyError as e:
            raise KeyError(
                f"An error occurred during the creating the prompt. The error message is: {e}."
            )

    @timeout()
    def model_response(
        self, model_param_dict: dict, debug_prompt: str = None, history: str = None
    ):
        """
        Generates a response from the provided model parameters and prompt.

        Parameters:
        -----------
        model_param_dict : dict
            A dictionary containing the parameters for the OpenAI Completion and Chat Completion API.

        debug_prompt : str
            Debug prompt

        history : str
            Previous response by GPT.

        Returns:
        --------
        tuple
            A tuple of output, finish reason and tokens from the OpenAI response.

        Raises:
        -------
        None
        """
        api_type = self.model_dict[self.track]
        if api_type == "openai":
            if model_param_dict["function"].lower() == "chatcompletion":
                current_message = [
                    {
                        "role": "system",
                        "content": self.prompt_dict["system_role"].replace(
                            "<language>", self.language
                        ),
                    },
                ]

                self.logger.debug(
                    f"system_prompt: {self.prompt_dict['system_role'].replace('<language>', self.language)}"
                )
                bot_history = model_param_dict["history"]

                data = {"role": "user", "content": self.prompt}
                current_message.append(data)
                if bool(bot_history) and len(bot_history) > 1:
                    if bot_history[0][1] is not None:
                        current_message.append(
                            {"role": "assistant", "content": bot_history[0][1]}
                        )
                    for conv in bot_history[1:]:
                        current_message.append({"role": "user", "content": conv[0]})
                        if (conv[1] is not None) and (conv[1].strip() != ""):
                            current_message.append(
                                {"role": "assistant", "content": conv[1]}
                            )

                if not (debug_prompt is None and history is None):
                    current_message = current_message + [
                        {"role": "assistant", "content": history},
                        {
                            "role": "user",
                            "content": debug_prompt,
                        },
                    ]
                    self.logger.debug(f"debug prompt:-\n\n{current_message}")
                self.current_message = current_message
                try:
                    if api_type == "openai":
                        response = self.client.chat.completions.create(
                            model=model_param_dict["engine"],
                            messages=current_message,
                            temperature=model_param_dict["temperature"],
                            max_tokens=model_param_dict["max_tokens"],
                            n=model_param_dict["n"],
                            stop=model_param_dict["stop"],
                        )
                    else:
                        response = self.client.chat.completions.create(
                            model=model_param_dict["engine"],
                            messages=current_message,
                            temperature=model_param_dict["temperature"],
                            max_tokens=model_param_dict["max_tokens"],
                            n=model_param_dict["n"],
                            stop=model_param_dict["stop"],
                        )
                    error_message = None

                except TokenLimitError:
                    error_message = f"Possible answer for the user query {self.question} exceeded the token limits. Please change the user query or the data."
                    self.logger.error(error_message)
                    return error_message

                except SensitiveContentError:
                    error_message = "The question is flagged as sensitive content by the OpenAI. Please change the language in the question or the data."
                    self.logger.error(error_message)

                except TimeoutError:
                    error_message = "The request to GPT model timed out (even after retries). Please resubmit the question after sometime or check with your IT team."
                    self.logger.error(error_message)

                except openai.APIError as e:
                    error_message = f"Something went wrong on the OpenAI side. Please resubmit the question.\nError:{e}"
                    self.logger.error(error_message)

                except openai.Timeout as e:
                    error_message = f"Request to GPT timed out. Please resubmit the question.\nError:{e}"
                    self.logger.error(error_message)

                except openai.RateLimitError as e:
                    error_message = f"Ratelimit exceeded. Please resubmit the question after one minute.\nError:{e}"
                    self.logger.error(error_message)

                except openai.APIConnectionError as e:
                    error_message = f"Could not establish connection to OpenAI's services. Please check with your IT team.\nError:{e}"
                    self.logger.error(error_message)
                except openai.AuthenticationError as e:
                    error_message = f"The API key may have been expired. Please check with your IT team.\nError:{e}"
                    self.logger.error(error_message)

                except openai.ServiceUnavailableError as e:
                    if "Service Unavailable" in str(e):
                        error_message = f"OpenAI's services are not available at the moment. Please resubmit your question after sometime. If problem still persists, please check with your IT team.\nError:{e}"
                        self.logger.error(error_message)

                except Exception as e:
                    error_message = f"t, error:\n{e}"
                    self.logger.error(error_message)
                    self.logger.error(traceback.format_exc())

                output = response.choices[0].message.content

                finish_reason = response.choices[0].finish_reason
                tokens = response.usage

                return output, finish_reason, tokens, error_message

            elif model_param_dict["function"].lower() == "completion":
                if debug_prompt is not None and history is not None:
                    self.prompt = self.prompt + "\n\n"
                    self.prompt += f"GPT response:\n\n{history}\n\n"
                    self.prompt += f"New Question: {debug_prompt}"

                try:
                    response = self.client.completions.create(
                        model=model_param_dict["engine"],
                        prompt=self.prompt,
                        temperature=model_param_dict["temperature"],
                        max_tokens=model_param_dict["max_tokens"],
                        n=model_param_dict["n"],
                        stop=model_param_dict["stop"],
                    )
                except TokenLimitError:
                    error_message = f"Possible answer for the user query {self.question} exceeded the token limits. Please change the user query or the data."
                    self.logger.error(error_message)
                    return error_message

                except SensitiveContentError:
                    error_message = "The question is flagged as sensitive content by the OpenAI. Please change the language in the question or the data."
                    self.logger.error(error_message)

                except TimeoutError:
                    error_message = "The request to GPT model timed out (even after retries). Please resubmit the question after sometime or check with your IT team."
                    self.logger.error(error_message)

                except openai.APIError as e:
                    error_message = f"Something went wrong on the OpenAI side. Please resubmit the question.\nError:{e}"
                    self.logger.error(error_message)

                except openai.Timeout as e:
                    error_message = f"Request to GPT timed out. Please resubmit the question.\nError:{e}"
                    self.logger.error(error_message)

                except openai.RateLimitError as e:
                    error_message = f"Ratelimit exceeded. Please resubmit the question after one minute.\nError:{e}"
                    self.logger.error(error_message)

                except openai.APIConnectionError as e:
                    error_message = f"Could not establish connection to OpenAI's services. Please check with your IT team.\nError:{e}"
                    self.logger.error(error_message)
                except openai.AuthenticationError as e:
                    error_message = f"The API key may have been expired. Please check with your IT team.\nError:{e}"
                    self.logger.error(error_message)

                except openai.ServiceUnavailableError as e:
                    if "Service Unavailable" in str(e):
                        error_message = f"OpenAI's services are not available at the moment. Please resubmit your question after sometime. If problem still persists, please check with your IT team.\nError:{e}"
                        self.logger.error(error_message)

                except Exception as e:
                    error_message = f"t, error:\n{e}"
                    self.logger.error(error_message)
                    self.logger.error(traceback.format_exc())

                output = response.choices[0].text
                finish_reason = response.choices[0].finish_reason
                tokens = response.usage
                return output, finish_reason, tokens, error_message
            else:
                raise ValueError(
                    f"Invalid function {model_param_dict['function']} is passed in the config. Acceptable values are 'Completion' and 'ChatCompletion'"
                )
