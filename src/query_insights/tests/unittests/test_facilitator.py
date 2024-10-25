import os
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
from sentence_transformers import SentenceTransformer, util

from query_insights.facilitator import DataLoader, FolderManager, Generic
from query_insights.utils.pre_processing import HybridQuestionClassifier
from query_insights.utils.utils import config_init


class TestFacilitator(unittest.TestCase):
    def test_check_similarity_and_get_question_index(self):
        similarity_threshold = 0.95
        model = SentenceTransformer("all-MiniLM-L6-v2")

        questions_list = [
            "What is the average unloading time at Location 14?",
            "What can be done to improve loading efficiency at Aberdeen?",
        ]
        question = "What is the average unloading time at Location 21?"

        question_list_tensor = model.encode(
            [
                "What is the average unloading time at Location 14?",
                "What can be done to improve loading efficiency at Aberdeen?",
            ],
            convert_to_tensor=True,
        )
        question1_tensor = model.encode(
            "What is the average unloading time at Location 21?", convert_to_tensor=True
        )

        cos_similarities = util.pytorch_cos_sim(
            question1_tensor, question_list_tensor
        ).tolist()[0]
        max_similarity = max(cos_similarities)
        max_index = "Q_20240325115058660577_1_4579"
        new_folder_index = "Q_20240325115058660577_2_1274"
        if max_similarity > similarity_threshold:  # Threshold for similarity
            self.assertEqual(max_index[:22], new_folder_index[:22])

        question2_tensor = model.encode(
            "Find out the mean unloading time for location 21?", convert_to_tensor=True
        )
        cos_sim_bw_2_ques = util.pytorch_cos_sim(
            question1_tensor, question2_tensor
        ).tolist()[0]
        max_similarity = max(cos_sim_bw_2_ques)
        print(max_similarity)
        if max_similarity < similarity_threshold:
            # Threshold for similarity
            new_folder_index = "Q_20240325180557882252_1_9239"
            self.assertEqual(max_index[:22], new_folder_index[:22])

    def test_post_process_output(self):
        output_table = pd.read_csv(
            "../../../../data/supply_chain_management/output_folder/streamlit_knowledge_base1/Q_20240325150759019891_1_7673/01_text_to_query/output_table.csv"
        )

        no_data_flag = False

        if output_table.isna().all(axis=1)[0]:
            no_data_flag = True
        self.assertEqual(no_data_flag, output_table.empty)

    def test_find_reason_based_questions(self):
        question = (
            "What factors contribute to Carrier 1 being considered the best carrier?"
        )
        classifier = HybridQuestionClassifier(embedding_model="bert")
        reason_based_questions = classifier.find_reason_based_questions(
            [question.split(";")[-1].strip()], 0.8
        )
        self.assertTrue(len(reason_based_questions) > 0)

    def test_folder_creation_for_new_question(self):
        # Domain name
        domain_name = "supply_chain_management"

        # Config path
        data_config_path = (
            f"../../../../azure_function/configs/{domain_name}/local/data_config.yaml"
        )
        user_config_path = (
            f"../../../../azure_function/configs/{domain_name}/local/user_config.yaml"
        )
        model_config_path = (
            f"../../../../azure_function/configs/{domain_name}/model_config.yaml"
        )
        debug_config_path = (
            f"../../../../azure_function/configs/{domain_name}/debug_code_config.yaml"
        )

        user_config, data_config, model_config, debug_config = config_init(
            user_config_path, data_config_path, model_config_path, debug_config_path
        )

        question = "what is the quantity exported from location 12?"

        folder_manager = FolderManager(
            user_config, data_config, model_config, debug_config, question, None
        )

        folder_manager._folder_creation_for_each_question()
        self.assertTrue(os.path.exists(folder_manager.output_path))

    def test_individual_track_folder_creation(self):
        table_to_insights_path = "../../../../data/supply_chain_management/output_folder/streamlit_knowledge_base1/Q_20240325150759019891_1_7673/table_to_insights"
        text_to_query_path = "../../../../data/supply_chain_management/output_folder/streamlit_knowledge_base1/Q_20240325150759019891_1_7673/text_to_query"
        query_to_chart_path = "../../../../data/supply_chain_management/output_folder/streamlit_knowledge_base1/Q_20240325150759019891_1_7673/query_to_chart"

        question = (
            "What factors contribute to Carrier 1 being considered the best carrier?"
        )
        classifier = HybridQuestionClassifier(embedding_model="bert")
        reason_based_questions = classifier.find_reason_based_questions(
            [question.split(";")[-1].strip()], 0.8
        )
        why_qn_flag = False
        for qn, _ in reason_based_questions:
            if qn == question:
                why_qn_flag = True
            else:
                why_qn_flag = False

        track = "01_text_to_query"
        if why_qn_flag:
            folders_to_create = table_to_insights_path
        else:
            if track == "01_text_to_query":
                folders_to_create = text_to_query_path
            elif track == "02_query_to_chart":
                folders_to_create = query_to_chart_path
            elif track == "03_table_to_insights":
                folders_to_create = table_to_insights_path
            else:
                raise ValueError(
                    "Invalid track provided. Valid values are '01_text_to_query', '02_query_to_chart', and '03_table_to_insights'."
                )

        self.assertEqual(folders_to_create, table_to_insights_path)


if __name__ == "__main__":
    unittest.main()
