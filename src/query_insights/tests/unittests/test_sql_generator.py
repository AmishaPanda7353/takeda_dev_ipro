import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from query_insights.text_to_sqlquery import TextToSqlQuery


class TestTextToSqlQuery(unittest.TestCase):
    @patch("query_insights.utils.utils.download_nltk_data", return_value=False)
    @patch("query_insights.utils.utils.download_spacy_data", return_value=False)
    def test_init(self, mock_download_nltk_data, mock_download_spacy_data):
        # self,user_config,data_config,model_config,debug_config,question,additional_context,en_core_web_model,logging_level,nltk_error_flag,spacy_error_flag,mock_download_nltk_data,mock_download_spacy_data
        user_config = {}
        data_config = {}
        model_config = {}
        debug_config = {}
        question = "Create a chart to compare on-time dispatch for different source locations.?"
        additional_context = None
        language = "english"
        en_core_web_model = "en_core_web_lg"
        logging_level = "INFO"

        track1_ins = TextToSqlQuery(
            user_config=user_config,
            data_config=data_config,
            model_config=model_config,
            debug_config=debug_config,
            question=question,
            additional_context=additional_context,
            language=language,
            logging_level="INFO",
        )

        if not track1_ins.nltk_error_flag:
            mock_download_nltk_data = True

        if not track1_ins.spacy_error_flag:
            mock_download_spacy_data = True

        # Assert that the instance is initialized correctly
        self.assertTrue(isinstance(track1_ins.user_config, dict))
        self.assertTrue(isinstance(track1_ins.data_config, dict))
        self.assertTrue(isinstance(track1_ins.model_config, dict))
        self.assertTrue(isinstance(track1_ins.debug_config, dict))
        self.assertEqual(track1_ins.question, question)
        self.assertEqual(track1_ins.additional_context, additional_context)
        self.assertEqual(en_core_web_model, "en_core_web_lg")
        self.assertEqual(track1_ins.logging_level, "INFO")
        self.assertTrue(mock_download_nltk_data)
        self.assertTrue(mock_download_spacy_data)

    def test_text_to_query(self):
        # Mock dependencies
        mock_user_config = {}
        mock_data_config = {}
        mock_model_config = {}
        mock_debug_config = {}
        mock_foldercreation = MagicMock()
        mock_dataloader = MagicMock()
        mock_config_validation = MagicMock()
        mock_dataprocessor = MagicMock()
        mock_generic_initializations = MagicMock()
        mock_skip_flag = MagicMock()

        # Initialize TextToSqlQuery object
        text_to_sql_query = TextToSqlQuery(
            mock_user_config, mock_data_config, mock_model_config, mock_debug_config
        )

        # Mock method calls and set return values as needed

        # Call the method being tested
        result = text_to_sql_query.text_to_query(
            mock_generic_initializations,
            mock_foldercreation,
            mock_dataloader,
            mock_config_validation,
            mock_dataprocessor,
            mock_skip_flag,
        )

        # Assert the result
        self.assertTrue(result["status"] in ["success", "failure", "skip"])
        self.assertTrue(isinstance(result["output"][0], str))
        self.assertTrue(isinstance(result["output"][1], pd.DataFrame))
        self.assertTrue(isinstance(result["output"][2][0], dict))

        # Add more assertions as needed


if __name__ == "__main__":
    unittest.main()
