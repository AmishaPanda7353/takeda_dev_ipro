import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from query_insights.query_to_chart import QueryToChart


class TestQueryToChart(unittest.TestCase):
    def test_init(self):
        # self,user_config,data_config,model_config,debug_config,question,additional_context,en_core_web_model,logging_level,nltk_error_flag,spacy_error_flag,mock_download_nltk_data,mock_download_spacy_data
        user_config = {}
        data_config = {}
        model_config = {}
        debug_config = {}
        question = "Create a chart to compare on-time dispatch for different source locations.?"
        additional_context = None
        language = "english"
        track1_output_table = pd.DataFrame()
        track1_output_table_dict = {}
        logging_level = "INFO"

        track2_ins = QueryToChart(
            user_config=user_config,
            data_config=data_config,
            model_config=model_config,
            debug_config=debug_config,
            question=question,
            additional_context=additional_context,
            language=language,
            logging_level="INFO",
            track1_output_table=track1_output_table,
            track1_output_table_dict=track1_output_table_dict,
        )

        # Assert that the instance is initialized correctly
        self.assertTrue(isinstance(track2_ins.user_config, dict))
        self.assertTrue(isinstance(track2_ins.data_config, dict))
        self.assertTrue(isinstance(track2_ins.model_config, dict))
        self.assertTrue(isinstance(track2_ins.debug_config, dict))
        self.assertEqual(track2_ins.question, question)
        self.assertEqual(track2_ins.additional_context, additional_context)
        self.assertEqual(track2_ins.logging_level, "INFO")
        self.assertTrue(isinstance(track2_ins.track1_output_table, pd.DataFrame))
        self.assertTrue(isinstance(track2_ins.track1_output_table_dict, dict))

    def test_text_to_query(self):
        # Mock dependencies
        mock_user_config = {}
        mock_data_config = {}
        mock_model_config = {}
        mock_debug_config = {}
        mock_foldercreation = MagicMock()
        mock_config_validation = MagicMock()
        mock_dataprocessor = MagicMock()
        mock_generic_initializations = MagicMock()
        mock_skip_flag = MagicMock()

        # Initialize TextToSqlQuery object
        query_to_chart_ins = QueryToChart(
            mock_user_config, mock_data_config, mock_model_config, mock_debug_config
        )

        # Mock method calls and set return values as needed

        # Call the method being tested
        result = query_to_chart_ins.query_to_chart(
            mock_generic_initializations,
            mock_foldercreation,
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
