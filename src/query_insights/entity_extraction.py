import ast
import itertools
import json
import logging
import re
import sys
import timeit

import pandas as pd
from fuzzywuzzy import fuzz
from keybert import KeyBERT
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util

from core.utils.client_utils import get_entity_extraction_client
from src.query_insights.utils.utils import DataConverter


class EntityExtraction:
    def __init__(self):
        """A class for getting union of results generated from using different methods of entity extraction methods

        Parameters
        ----------
        data_dictionary: str
            data dictionary
        question : str
            Question that is asked by user

        Raises
        ------
        ValueError
            if any of the argument is missing or invalid.
        """

        self.logger = logging.getLogger("entity_extraction_logger")
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            stream=sys.stdout,
        )
        self.question_entities_list = []
        self.lemmatizer = WordNetLemmatizer()
        self.date_columns_flag = False
        self.date_entities = []
        self.non_date_entities = []
        self.chunk_embeddings_path = "../../data/chunk_embeddings.xlsx"
        self.chunk_embeddings = pd.read_excel(self.chunk_embeddings_path)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.question_extension = ""
        self.final_data_dictionary_output = {}
        self.duplicate_tables_dict = {}
        formulas_json_path = "../../data/output_folder/formulas.json"
        self.formulas_json = self._load_json(formulas_json_path)

    def _load_json(self, path):
        """Loads a single JSON file.

        This function opens and reads a JSON file specified by the path.

        Parameters:
        path (str): Path to the JSON file to be loaded.

        Returns:
        dict: The content of the JSON file.

        Raises:
        Exception: If an error occurs during file reading or JSON parsing.

        """
        try:
            with open(path, "r") as file:
                json_data = json.load(file)
            return json_data
        except Exception as e:
            self.logger.error(f"Error while loading json. Error : {e}")

    def _lemmatization(
        self,
        text,
    ):
        """
        Processes a given text string for word normalization and lemmatization.

        This method converts the text to lowercase, replaces hyphens and underscores
        with spaces, lemmatizes each word and replace id string with space if the text ends with id.

        Parameters:
        text (str): The text to be processed.

        Returns:
        str: The processed and lemmatized text.

        Raises:
        Exception: If an error occurs during the processing.
        """

        text = text.lower()
        text = text.replace("-", " ")
        text = text.replace("_", " ")
        lemmatized_texts = " ".join(
            [
                self.lemmatizer.lemmatize(re.sub(r"[^a-zA-Z0-9]", " ", txt).strip())
                for txt in text.split(" ")
            ]
        )
        return lemmatized_texts

    def _contain_datetime(self, string):
        """
        This function is used to find whether a string contains date pattern using some expression.

        Parameters:
        text (str): The text to be processed.

        Returns:
        bool: True if the text string contains date pattern, False otherwise.

        Raises:
        Exception: If an error occurs during finding date pattern.
        """
        try:
            combined_pattern = r"\b(?:\d{1,2}(?:st|nd|rd|th)(?:\d{2}-\d{2}-\d{4}|\d{4}-\d{2}-\d{2}|\d{4}-\d{2}|\d{2}-\d{4}|\d{2}[\/\\-]\d{2}[\/\\-]\d{4}|\d{4}[\/\\-]\d{2}[\/\\-]\d{2}|\d{4}[\/\\-]\d{2}|\d{2}[\/\\-]\d{4}|\d{2}[\/\\-]\d{2})\)(?:\d{1,2}(?:[-\/\s*]\d{2,4})?|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?(?:[-\/\s*,]*\s*(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?))?|today|tomorrow|yesterday|now|tonight|morning|afternoon|evening|night|midnight|noon|day|week|month|year|wk|mnth|mth|yr|decade|date|dt|weekly|monthly|yearly)\b"
            if re.search(combined_pattern, string):
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error while finding date pattern. Error : {e}")

    def _removing_only_containing_id_columns(self, entity_extraction_data_dict):
        """
        This function is used to remove id columns if the table contains only id column.

        Parameters:
        -----------

        Returns:
        list: list of tuple of entities

        Raises:
        Exception: If an error occurs during finding date pattern.
        """

        try:
            remove_id_columns = []
            for table_name, table_data in entity_extraction_data_dict.items():
                if len(table_data["columns"]) == 1:
                    if (
                        "id" in table_data["columns"][0].keys()
                        and table_data["columns"][0]["id"] == "Yes"
                    ):
                        remove_id_columns.append(
                            (table_name, table_data["columns"][0]["name"])
                        )
            for tuple_ in remove_id_columns:
                entity_extraction_data_dict.pop(tuple_[0])
                # for key,value in self.match_list.items():
                #     for tab_col in value:
                #         if tuple_[0] == tab_col[0] and tuple_[1] == tab_col[1]:
                #             self.match_list[key].remove(tab_col)

            self.logger.info(
                f"Successfully removed only containing id columns in each table.Error"
            )
            return entity_extraction_data_dict
        except Exception as e:
            self.logger.error(
                f"Error while removing only containing id columns in each table.Error : {e}"
            )

    def get_data_dictionary(self, data_dictionary, date_columns_flag):
        """
        Function to convert into data dictionary from list of tuples.
        Parameters
        ----------
        None

        Returns
        -------

        Raises:
        --------
        Exception: If an error occurs converting data dictionary.

        """

        try:
            entity_extraction_data_dict = {
                table_name: {
                    "table_name": table_data["table_name"],
                    "columns": [
                        {
                            key: value
                            for key, value in column.items()
                            if key in ["name", "description", "id"]
                        }
                        for column in table_data["columns"]
                        if column.get("id") == "Yes"
                        or column.get("match") == "Yes"
                        or (
                            column.get("entity_datecolumn") == "Yes"
                            and date_columns_flag
                        )
                    ],
                }
                for table_name, table_data in data_dictionary.items()
                if any(column.get("match") == "Yes" for column in table_data["columns"])
            }

            self.logger.info("Successfully converted into data dictionary")
            return entity_extraction_data_dict
        except Exception as e:
            self.logger.error(
                f"Error while converting into data dictionary. Error : {e}"
            )

    def _finding_date_entities(self, question_entities, lemmatization=True):
        """
        This function is used to find date_entities_flag , date_entiites and non_date_entities.

        Parameters:
        -----------

        Returns:
        -------

        Raises:
        Exception: If an error occurs during finding date_entities_flag , date_entiites and non_date_entities.
        """

        try:
            date_columns_flag = False
            date_entities = []
            non_date_entities = []
            for entity in question_entities:
                if lemmatization:
                    entity_text = self._lemmatization(entity)
                else:
                    entity_text = entity
                if self._contain_datetime(entity_text):
                    date_columns_flag = True
                    date_entities.append(entity)
                else:
                    non_date_entities.append(entity)

            self.logger.info(
                f"Successfully found date_entities_flag, date entities and non date entities."
            )
            return non_date_entities, date_entities, date_columns_flag

        except Exception as e:
            self.logger.error(
                f"Error while finding date_entities_flag, date entities and non date entities. Error: {e}"
            )

    def add_logic_or_glossary(self, question, match_entity, data_dictionary):
        """
        This function is used to add columns based on the logic or add chunks from glossary to question.

        Parameters:
        -----------

        Returns:
        -------
        match_dictionary (dict) : output data dictionary with key as match_entity and value is dictionary with count matches as key(int, number of columns matches) and matches(list of tuples, each tuple with contain iformation about table_name, column_name, logic no, fuzzy score)
        question (str) : user question
        data_dictionary (dict) : data dictionary added match key with value as 'Yes' if it matches with match entity

        Raises:
        Exception: If an error occurs during adding of columns based on the logic or add chunks from glossary to question.
        """

        try:
            match_dictionary = {}
            no_digit_question = re.sub(r"\d+", "", question)
            question_embeddings = self.model.encode(no_digit_question)
            logic_list = []
            add_logic_columns = []
            logic_flag = False
            match_dictionary[match_entity] = {}
            match_dictionary["logics_text"] = {}
            for i in range(self.chunk_embeddings.shape[0]):
                similarity_scores = util.pytorch_cos_sim(
                    question_embeddings.tolist(),
                    ast.literal_eval(self.chunk_embeddings["embeddings"][i]),
                )

                if similarity_scores.numpy().item() > 0.75 and self.chunk_embeddings[
                    "chunk_type"
                ][i].startswith("Glossary"):
                    question = (
                        question
                        + "."
                        + " "
                        + self.chunk_embeddings["chunk_type"][i].split(":")[1].strip()
                    )

                elif (
                    (similarity_scores.numpy().item() > 0.75)
                    and not (
                        self.chunk_embeddings["chunk_type"][i].startswith("Glossary")
                    )
                    and (
                        self.chunk_embeddings["chunk_type"][i].split(":")[0].strip()
                        not in logic_list
                    )
                ):

                    logic = self.chunk_embeddings["chunk_type"][i].split(":")[0].strip()
                    table_column = self.chunk_embeddings["tables_columns"][i].split(",")
                    logic_flag = True
                    for j in range(len(table_column)):
                        table_name, column_name = (
                            table_column[j].split(".")[0].strip(),
                            table_column[j].split(".")[1].strip(),
                        )
                        if "count_matches" not in match_dictionary[match_entity].keys():
                            match_dictionary[match_entity]["count_matches"] = 0
                            match_dictionary[match_entity]["matches"] = []
                            if logic not in match_dictionary["logics_text"].keys():
                                match_dictionary["logics_text"][logic] = None
                                match_dictionary["logics_text"][logic] = (
                                    self.chunk_embeddings["chunk_type"][i]
                                )
                        add_logic_columns.extend(table_column)

                        match_dictionary[match_entity]["matches"].append(
                            (
                                table_name,
                                column_name,
                                "<" + logic + ">",
                                similarity_scores.numpy().item(),
                            )
                        )
                        for row in data_dictionary[table_name]["columns"]:
                            if row["name"] == column_name:
                                row["match"] = "Yes"

                if i == (self.chunk_embeddings.shape[0] - 1) and logic_flag:

                    match_dictionary[match_entity]["count_matches"] += len(
                        set(add_logic_columns)
                    )

            self.logger.info(
                f"Successfuly added  columns based on the logic or add chunks from glossary to question."
            )

            return match_dictionary, question, data_dictionary
        except Exception as e:
            self.logger.error(
                f"Error while adding  columns based on the logic or glossary chunk to the question. Error : {e}"
            )

    def _find_duplicate_column_tables(self, tables):
        # Step 1: Create a dictionary to hold sets of columns for each table
        table_columns = {table: set(columns) for table, columns in tables.items()}

        # Step 2: Create a reverse dictionary to map column sets to lists of tables
        columns_table_map = {}
        for table, columns in table_columns.items():
            columns_tuple = tuple(sorted(columns))  # Use tuple of sorted columns as key
            if columns_tuple not in columns_table_map.keys():
                columns_table_map[columns_tuple] = []
            columns_table_map[columns_tuple].append(table)

        # Step 3: Filter to find sets of columns associated with more than one table
        duplicate_columns_tables = {
            cols: tbls for cols, tbls in columns_table_map.items() if len(tbls) > 1
        }
        duplicate_tables = [val for val in columns_table_map.values() if len(val) > 1]

        return duplicate_columns_tables, duplicate_tables

    def _add_formulas_and_categorical_values(self, match_entities, match_dictionary):

        try:
            if "metrics" in match_entities:
                match_dictionary["formulas"] = {}
                for col_name, matches in match_dictionary["metrics"].items():
                    if match_dictionary["metrics"]:
                        for match in matches["matches"]:

                            if match and match[2][1:-1] in list(
                                self.formulas_json.keys()
                            ):
                                match_dictionary["formulas"].update(
                                    {match[2][1:-1]: self.formulas_json[match[2][1:-1]]}
                                )
            if "entity_categorical_values" in match_entities:
                match_dictionary["categorical_values"] = {}
                for col_name, matches in match_dictionary[
                    "entity_categorical_values"
                ].items():
                    for match in matches["matches"]:
                        if (
                            match[1]
                            not in match_dictionary["categorical_values"].keys()
                        ):
                            match_dictionary["categorical_values"][match[1]] = []
                        match_dictionary["categorical_values"][match[1]].extend(
                            (match[2])
                        )
                remove_col = []
                for col_name, cat_val in match_dictionary["categorical_values"].items():
                    match_dictionary["categorical_values"][col_name] = list(
                        set(match_dictionary["categorical_values"][col_name])
                    )
                    if len(match_dictionary["categorical_values"][col_name]) <= 5:
                        remove_col.append(col_name)

                for rc in remove_col:
                    match_dictionary["categorical_values"].pop(rc)
            return match_dictionary
        except Exception as e:
            self.logger.info(e)

    def get_matches_from_entities(
        self,
        entities_list,
        method,
        match_entity,
        data_dictionary,
        lemmatization,
        fuzzy_threshold=None,
        fuzzy_method=None,
    ):
        """
        This function is used to add columns based on the match_entity for question entities.

        Parameters:
        -----------

        Returns:
        -------
        match_dictionary (dict) : output data dictionary with key as match_entity and value is dictionary with question_entity as key and again value as dictionary with count matches as key(int, number of columns matches) and matches(list of tuples, each tuple with contain iformation about table_name, column_name, logic no, fuzzy score)
        question (str) : user question
        data_dictionary (dict) : data dictionary added match key with value as 'Yes' if it matches with match entity

        Raises:
        Exception: If an error occurs during adding of columns based on the match_entity for question entities.
        """

        try:
            match_dictionary = {}
            match_dictionary[match_entity] = {}
            matched_entities = []
            original_method = method
            for question_entity in entities_list:
                if lemmatization:
                    entity_text = self._lemmatization(question_entity)
                else:
                    entity_text = question_entity
                method = (
                    ["regex"]
                    if (
                        len(question_entity.split(" ")) == 1
                        and match_entity == "entity_categorical_values"
                        and method == ["fuzzy"]
                    )
                    else original_method
                )
                match_dictionary[match_entity][question_entity] = {}

                for table_name, table_data in data_dictionary.items():

                    for dictionary in table_data["columns"]:

                        if "exact" in method and (
                            entity_text in dictionary[match_entity]
                        ):

                            if (
                                "count_matches"
                                not in match_dictionary[match_entity][
                                    question_entity
                                ].keys()
                            ) and (
                                "matches"
                                not in match_dictionary[match_entity][
                                    question_entity
                                ].keys()
                            ):
                                match_dictionary[match_entity][question_entity][
                                    "count_matches"
                                ] = 0
                                match_dictionary[match_entity][question_entity][
                                    "matches"
                                ] = []

                            match_dictionary[match_entity][question_entity][
                                "count_matches"
                            ] += 1
                            match_dictionary[match_entity][question_entity][
                                "matches"
                            ].append((table_name, dictionary["name"]))

                            dictionary["match"] = "Yes"

                            matched_entities.append(question_entity)

                        if "fuzzy" in method:
                            for entity in dictionary[match_entity]:
                                fuzzy_score = fuzzy_method(entity, entity_text)
                                if fuzzy_score > fuzzy_threshold:
                                    if (
                                        "count_matches"
                                        not in match_dictionary[match_entity][
                                            question_entity
                                        ].keys()
                                    ) and (
                                        "matches"
                                        not in match_dictionary[match_entity][
                                            question_entity
                                        ].keys()
                                    ):
                                        match_dictionary[match_entity][question_entity][
                                            "count_matches"
                                        ] = 0
                                        match_dictionary[match_entity][question_entity][
                                            "matches"
                                        ] = []
                                    match_dictionary[match_entity][question_entity][
                                        "count_matches"
                                    ] += 1
                                    match_dictionary[match_entity][question_entity][
                                        "matches"
                                    ].append(
                                        (
                                            table_name,
                                            dictionary["name"],
                                            "<" + entity + ">",
                                            round(fuzzy_score, 2),
                                        )
                                    )

                                    dictionary["match"] = "Yes"
                                    matched_entities.append(question_entity)

                        if "regex" in method:
                            pattern = re.compile(entity_text, re.IGNORECASE)
                            matching_items = []

                            if not entity_text.isnumeric():
                                matching_items = [
                                    item
                                    for item in dictionary["entity_categorical_values"]
                                    if pattern.search(item)
                                ]

                            else:
                                matching_items = [
                                    item
                                    for item in dictionary["entity_categorical_values"]
                                    if pattern.search(item)
                                    and fuzzy_method(entity_text, item)
                                    > fuzzy_threshold
                                ]
                            if matching_items:
                                if (
                                    "count_matches"
                                    not in match_dictionary[match_entity][
                                        question_entity
                                    ].keys()
                                ) and (
                                    "matches"
                                    not in match_dictionary[match_entity][
                                        question_entity
                                    ].keys()
                                ):
                                    match_dictionary[match_entity][question_entity][
                                        "count_matches"
                                    ] = 0
                                    match_dictionary[match_entity][question_entity][
                                        "matches"
                                    ] = []
                                match_dictionary[match_entity][question_entity][
                                    "count_matches"
                                ] += 1
                                match_dictionary[match_entity][question_entity][
                                    "matches"
                                ].append(
                                    (table_name, dictionary["name"], matching_items)
                                )

                                dictionary["match"] = "Yes"
                                matched_entities.append(question_entity)

                if match_dictionary[match_entity][question_entity] == {}:
                    match_dictionary[match_entity].pop(question_entity, None)

            unmatched_entities = set(entities_list) - set(matched_entities)
            self.logger.info(
                f"Successfuly added  columns based on the match_entity for question entities"
            )

            return (
                unmatched_entities,
                matched_entities,
                match_dictionary,
                data_dictionary,
            )
        except Exception as e:
            self.logger.error(
                f"Error while adding  columns based on the match_entity for question entities. Error : {e}"
            )

    def _remove_duplicate_tables(self, tables, data_dictionary):

        try:
            sorted_tables = dict(sorted(tables.items(), key=lambda item: len(item[1])))

            # Initialize an empty dictionary to store results
            results = {}

            # Step 2: Iterate through the sorted dictionary
            for table, columns in sorted_tables.items():

                # Check if all columns are present in any of the larger tables
                is_duplicate = False
                for other_table, other_columns in sorted_tables.items():
                    if table != other_table and len(columns) < len(other_columns):
                        if all(column in other_columns for column in columns):
                            is_duplicate = True
                            break

                # Step 3: Mark the table as 'duplicate' if criteria met
                results[table] = "duplicate" if is_duplicate else "unique"
            for table_name, duplicate in results.items():
                if duplicate == "duplicate":
                    data_dictionary.pop(table_name)
            return data_dictionary, results

        except Exception as e:
            self.logger.error(e)

    def _remove_entities_superset_of_matched_entities(
        self, matched_entities, unmatched_entities
    ):
        try:
            remove_entities = []
            for match_entity in matched_entities:
                for unmatch_entity in unmatched_entities:
                    if match_entity in unmatch_entity.split(" "):
                        remove_entities.append(unmatch_entity)

            return remove_entities
        except Exception as e:
            self.logger.error(e)

    def remove_matched_entities(self, matched_entities, unmatched_entities):
        remove_entities = set()
        add_entities = set()
        matched_add_entities = set()
        for unmatched in unmatched_entities:
            unmatched_words = set(unmatched.split(" "))
            for match in matched_entities:
                match_words = set(match.split(" "))
                common_words = unmatched_words & match_words
                if common_words:
                    for word in common_words:
                        if word != unmatched:
                            ent = unmatched.replace(word, "").strip()
                            if word not in matched_entities:
                                add_entities.add(ent)
                            remove_entities.add(unmatched)
                        else:
                            remove_entities.add(word)
                            matched_add_entities.add(word)

        matched_entities = set(matched_entities).union(matched_add_entities)
        unmatched_entities = list(set(unmatched_entities) - remove_entities)
        unmatched_entities = list(set(unmatched_entities).union(add_entities))
        return unmatched_entities, matched_entities

    def keybert_entity_extraction(
        self, entity_extraction_config, question, data_dictionary
    ):
        """
        Function to get one list having column indexes using keybert entity extraction method for table selection.
        These indexes can be used to get column related information from the dataframe having unique index
        for unique column name and other information related to that column.

        Parameters
        ----------
        None

        Returns
        -------
        match_dictionary (dict) : output data dictionary with key as match_entity and value is dictionary with question_entity as key and again value as dictionary with count matches as key(int, number of columns matches) and matches(list of tuples, each tuple with contain iformation about table_name, column_name, logic no, fuzzy score)
        question (str) : user question
        data_dictionary (dict) : data dictionary added match key with value as 'Yes' if it matches with match entity
        """

        try:
            question_entities = []
            matched = set()

            output_dict_step1, self.question_extension, data_dictionary = (
                self.add_logic_or_glossary(
                    question, match_entity="logic", data_dictionary=data_dictionary
                )
            )

            entity_extraction = get_entity_extraction_client(
                entity_extraction_config, data_dictionary
            )

            for question_chunk in self.question_extension.split("."):

                entities = entity_extraction.get_entities(question_chunk)

                question_entities.extend(entities)

            self.question_entities_list = [tup[0] for tup in question_entities]
            self.question_entities_list = list(set(self.question_entities_list))

            non_date_entities, date_entities, date_columns_flag = (
                self._finding_date_entities(
                    self.question_entities_list, lemmatization=True
                )
            )
            matched = matched.union(set(date_entities))

            unmatched_entities, matched_entities, output_dict_step2, data_dictionary = (
                self.get_matches_from_entities(
                    non_date_entities,
                    method=["exact"],
                    match_entity="entity_name",
                    data_dictionary=data_dictionary,
                    lemmatization=True,
                )
            )

            matched = matched.union(set(matched_entities))

            unmatched_entities, matched_entities, output_dict_step3, data_dictionary = (
                self.get_matches_from_entities(
                    unmatched_entities,
                    method=["fuzzy"],
                    fuzzy_method=fuzz.token_sort_ratio,
                    data_dictionary=data_dictionary,
                    lemmatization=True,
                    match_entity="metrics",
                    fuzzy_threshold=75,
                )
            )

            matched = matched.union(set(matched_entities))

            unmatched_entities, matched = self.remove_matched_entities(
                matched, unmatched_entities
            )

            max_len = max(
                len(unmatch_ent.split()) for unmatch_ent in unmatched_entities
            )
            output_dict_step4 = {}
            output_dict_step5 = {}
            for i in range(max_len, 0, -1):

                unmatched_entities_list = [
                    ent for ent in unmatched_entities if len(ent.split(" ")) == i
                ]

                (
                    unmatched_entities_list,
                    matched_entities,
                    output_dict_step,
                    data_dictionary,
                ) = self.get_matches_from_entities(
                    unmatched_entities_list,
                    method=["regex"],
                    fuzzy_method=fuzz.token_sort_ratio,
                    data_dictionary=data_dictionary,
                    lemmatization=False,
                    match_entity="entity_categorical_values",
                    fuzzy_threshold=60,
                )

                if "entity_categorical_values" in output_dict_step4.keys():
                    output_dict_step4["entity_categorical_values"].update(
                        output_dict_step["entity_categorical_values"]
                    )
                else:
                    output_dict_step4 = output_dict_step
                matched = matched.union(set(matched_entities))
                # unmatched_entities,matched = self.remove_matched_entities(matched,unmatched_entities)
                (
                    unmatched_entities_list,
                    matched_entities,
                    output_dict_step,
                    data_dictionary,
                ) = self.get_matches_from_entities(
                    unmatched_entities_list,
                    method=["exact", "fuzzy"],
                    match_entity="entity_description",
                    data_dictionary=data_dictionary,
                    fuzzy_method=fuzz.token_sort_ratio,
                    lemmatization=True,
                    fuzzy_threshold=75,
                )

                matched = matched.union(set(matched_entities))

                if "entity_description" in output_dict_step5.keys():
                    output_dict_step5["entity_description"].update(
                        output_dict_step["entity_description"]
                    )
                else:
                    output_dict_step5 = output_dict_step

                unmatched_entities = list(set(unmatched_entities) - matched)

                unmatched_entities, matched = self.remove_matched_entities(
                    matched, unmatched_entities
                )

            data_dictionary_filtered = self.get_data_dictionary(
                data_dictionary, date_columns_flag
            )

            tables = {}
            for table_name, table_data in data_dictionary_filtered.items():
                for dictionary in table_data["columns"]:
                    if table_name not in tables.keys():
                        tables[table_name] = []
                    tables[table_name].append(dictionary["name"])

            data_dictionary_filtered, self.duplicate_tables_dict = (
                self._remove_duplicate_tables(tables, data_dictionary_filtered)
            )

            data_dictionary_filtered = self._removing_only_containing_id_columns(
                data_dictionary_filtered
            )

            self.final_data_dictionary_output = dict(
                **output_dict_step1,
                **output_dict_step2,
                **output_dict_step3,
                **output_dict_step4,
                **output_dict_step5,
            )
            self.final_data_dictionary_output = (
                self._add_formulas_and_categorical_values(
                    ["metrics", "entity_categorical_values"],
                    self.final_data_dictionary_output,
                )
            )
            self.final_data_dictionary_output["unmatched_entities"] = unmatched_entities
            return data_dictionary_filtered, self.final_data_dictionary_output
        except Exception as e:
            self.logger.error(
                f"Error while performing keybert_entity_extraction. Error : {e}"
            )

    def get_clarification_dictionary_and_columns_dictionary(
        self, match_detail_dictionary, data_dictionary
    ):

        clarification_dict = {}
        clarification_dict["columns_dont_need_clarification"] = []
        clarification_dict["keywords_which_need_clarification_on_columns"] = {}
        for match_entity, dict1 in match_detail_dictionary.items():
            if match_entity.startswith("entity"):
                for entity, dict2 in dict1.items():
                    columns_list = set()
                    for tuple_ in dict2["matches"]:
                        columns_list.add(tuple_[1])
                    if len(columns_list) >= 3 and match_entity == "entity_description":
                        clarification_dict[
                            "keywords_which_need_clarification_on_columns"
                        ][entity] = list(columns_list)
                    elif len(columns_list) <= 3:
                        clarification_dict["columns_dont_need_clarification"].extend(
                            list(columns_list)
                        )
                clarification_dict["columns_dont_need_clarification"] = list(
                    set(clarification_dict["columns_dont_need_clarification"])
                )

        columns_description = {}
        for table_name, table_data in data_dictionary.items():
            for dict1 in table_data["columns"]:
                columns_description[dict1["name"]] = dict1["description"]

        match_detail_dictionary["column_description"] = columns_description

        return clarification_dict, match_detail_dictionary
