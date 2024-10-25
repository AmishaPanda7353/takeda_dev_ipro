import logging
import re

from core.utils.client_utils import get_model_type
from core.utils.read_config import model

from core.model.model_factory import ModelFactory

MYLOGGERNAME = "QueryInsights"


class followup_q_tag:
    def __init__(
        self,
        user_config,
        model_config,
        question,
        output_path=None,
        skip_model: bool = False,
    ) -> None:
        """Class constructor"""
        # Config related
        self.prompt_dict = model_config.followup_question_tagging.prompts
        self.model_param_dict = model_config.followup_question_tagging.model_params

        self.connection_param_dict = user_config.connection_params
        self.ui = user_config.ui

        # Business user query related
        self.question = question
        self.output_path = output_path
        self.skip_model = skip_model

        # Logger
        self.logger = logging.getLogger(MYLOGGERNAME)

        # GPT Model Call object
        config = dict()
        config["llm_model_type"] = model
        model_client = get_model_type(
            config=None,
            prompt_dict=self.prompt_dict,
            question=self.quesiton,
            additional_context=None,
            connection_param_dict=self.connection_param_dict,
            user_config=user_config,
            language=None,
            data_dictionary=None,
            business_overview=None,
            db_param_dict=None,
            sample_input=None,
            code_result=None,
            table=None,
        )
        # GPT Model Call object
        self.model_factory = ModelFactory(model_client)

    def _call_model_api(self):
        """
        call_model_api
        Get model response from GPT model
        """
        self.logger.debug("Sending a new request to GPT")
        (
            model_output,
            model_finish,
            model_tokens,
        ) = self.model_factory.model_response(self.model_param_dict)

        self.logger.info(
            f"Followup question tagging:-\n finish token - {model_finish},\n token information - {model_tokens}"
        )
        self.logger.debug(f"Model output\n{model_output}")

        return model_output, model_finish, model_tokens

    def process_followup_question(self, skip_api_call: bool = False):
        if not skip_api_call:
            self.logger.info("Calling the API for SQL generation.")
            (
                self.model_output,
                self.model_finish,
                self.model_tokens,
            ) = self._call_model_api()

        self.model_output = re.sub(r"[^a-zA-Z\s]", "", self.model_output)

        if self.model_output.lower() == "sql related":
            # followup question is SQL related. Only track 1 will be re-run
            skip_track1 = False
            skip_track2 = False
            skip_track3 = False
        elif self.model_output.lower() == "graph related":
            # followup question is graph related. Only track 2 will be re-run
            skip_track1 = True
            skip_track2 = False
            skip_track3 = True
        elif self.model_output.lower() == "insights related":
            # followup question is insights related. Only track 3 will be re-run
            skip_track1 = True
            skip_track2 = True
            skip_track3 = False
        elif self.model_output.lower() == "greeting":
            # Greeting message. No need to rerun any track
            skip_track1 = True
            skip_track2 = True
            skip_track3 = True
        else:
            # else is for general followup question. all 3 tracks should be re-run.
            # Also if GPT response has something other than the expected tags (sql related / graph related / ...), all 3 tracks will be re-run
            skip_track1 = False
            skip_track2 = False
            skip_track3 = False

        self.logger.info(f"The followup question is tagged as '{self.model_output}'")
        return skip_track1, skip_track2, skip_track3
