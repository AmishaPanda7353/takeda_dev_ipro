import os
import re
import time
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def extract_tables_from_query(sql_query):
    table_names = []
    real_table = [
        "recommendation_table",
        "playbook_table",
        "snapshot_table",
        "gc_table",
        "eva_store_item_report",
        "summary_store_table",
        "summary_item_table",
        "summary_category_table",
        "summary_price_table",
    ]
    pattern = r"\b(?:FROM|JOIN)\s+([^\s]+)"
    matches = re.findall(pattern, sql_query, re.IGNORECASE)
    for match in matches:
        table_name = match.split(".")[-1].split(" ")[0]
        table_names.append(table_name)
    table_names_all = [
        name.lower() for name in table_names if name.lower() not in "(select"
    ]
    table_names_all = [tab for tab in table_names_all if tab in real_table]
    return list(set(table_names_all))


def check_sql_match(question_id, output_path, question, expected_query_file_name):
    contains_status = {}
    contains_sql = {}
    flag = True
    try:
        expected_sql = pd.read_excel(
            f"../../data/expected_sql/{expected_query_file_name}"
        )
        expected_sql = expected_sql.query(f"Question_ID=={question_id}")[
            "Expected_SQL_Query"
        ]
        clean_sql_query = re.sub(
            r"\s+",
            " ",
            expected_sql.values[0].replace("\n", " ").replace("\r", " ").strip(),
        )
        expected_table = extract_tables_from_query(clean_sql_query)
        contains_status["Expected_query"] = "True"
    except Exception:
        contains_status["Expected_query"] = "False"

    try:
        with open(f"{output_path}/sql_query.sql", "r") as file:
            llm_sql = file.read()
        clean_llm_query = re.sub(
            r"\s+",
            " ",
            llm_sql.replace("\n", " ").replace("\r", " ").strip(),
        )
        llm_table = extract_tables_from_query(clean_llm_query)
        contains_status["llm_query"] = "True"
    except Exception:
        contains_status["llm_query"] = "False"

    try:
        if "previous" in question or "prior" in question or "last" in question:
            if "round_id < " in clean_llm_query.lower():
                print("uuuu")
                flag = True
            else:
                raise ValueError("The query does not contain 'previous round details.")
        else:
            if (
                "max(round_id)" in clean_llm_query.lower()
                and "round_id < " not in clean_llm_query.lower()
            ):
                flag = True
            else:
                raise ValueError("The query does not contain 'previous round details.")
    except Exception:
        flag = False

    contains_sql["Expected_sql"] = expected_sql.values[0]
    contains_sql["llm_sql"] = llm_sql
    contains_status["Expected_tables"] = [expected_table]
    contains_status["llm_tables"] = [llm_table]
    return contains_status, contains_sql, flag


def multiple_dfs(df_list, sheets, file_name, spaces, sql_queries):
    """
    Combine ideal and llm output for failled experiment and generate the file for manual reveiew.
    """

    directory_path = "../../data/review/"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    writer = pd.ExcelWriter(f"{directory_path}{file_name}", engine="xlsxwriter")
    row = 0

    for index, dataframe in enumerate(df_list):
        if index == 0:
            dataframe["Expected_sql"] = pd.NA
            dataframe.at[0, "Expected_sql"] = sql_queries["Expected_sql"]
            row = row + 2
        else:
            dataframe["llm_sql"] = pd.NA
            dataframe.at[0, "llm_sql"] = sql_queries["llm_sql"]
            row = row + 2

        dataframe.to_excel(
            writer, sheet_name=sheets, startrow=row, startcol=2, index=False
        )
        row = row + len(dataframe.index) + spaces + 1
        if index == 0:
            ws = writer.sheets[sheets]
            ws.write(0, 0, "Expected_output")
            ws.write(row, 0, "llm_output")
    writer.save()


def check_range(df, sub):
    contain_col = []
    for col in df.columns:
        if df[col].apply(lambda x: sub in str(x)).any():
            contain_col.append(col)

    return contain_col


def check_count(df, columns_to_check, sub):
    to_counts = {}
    for column in columns_to_check:
        to_counts[column] = df[column].str.count(sub).sum()
    return to_counts


def compare_dash_to(dash):
    for key, value in dash.items():
        pass


def check_distribution(llm):
    columns_with_range_count = [
        col
        for col in llm.columns
        if "range" in col or "count" in col or "percentage" in col
    ]
    check_to = check_range(llm, " to ")
    check_dash = check_range(llm, " - ")

    if columns_with_range_count and (len(check_to) > 0 or len(check_dash) > 0):
        return True
    else:
        return False


def contains_all_elements(ideal, llm):
    count_ideal = Counter(ideal)
    count_llm = Counter(llm)
    for key, value in count_llm.items():
        # print(key, value)
        if count_ideal[key] > value:
            return False

    return True


def can_subtract(column):
    return np.issubdtype(column.dtype, np.number)


def check_in_llm(ideal, llm):
    """To check the data availibity from expected dataset to llm dataset"""

    # if can_subtract(ideal) and can_subtract(llm):
    #     mask = (llm - ideal).abs() <= 0.0001
    #     return mask.tolist()

    if True:
        mask = ideal.apply(lambda x: x in llm.values)
        return mask.tolist()


def get_iteration(df):
    """Add the iteration column into existing dataframe if single folder contains multiple iteration."""

    df["iteration"] = df["question_folder"].rank(method="min")
    df_sorted = df.sort_values(by="iteration")
    return df_sorted


def sql_evaluation(output_path, question, expected_query_file_name):
    """Evalaute the sql output of llm and expected query.
    Parameters
    ----------
    output_path : string
        output path of llm experiment for evalation
    Returns:
        dataframe containing the status of evaluation.
    """

    if os.path.exists(output_path):
        overall_output = defaultdict(list)
        llm_check = True
        check_identifier = True
        check_merge = True
        try:
            question_folder = os.path.basename(os.path.dirname(output_path))
            question_id = question_folder.split("_")[1]
            sql_tables, sql_queries, flag = check_sql_match(
                question_id, output_path, question, expected_query_file_name
            )
            overall_output["question_id"].append(question_id)
            overall_output["question_folder"].append(question_folder)
            overall_output.update(sql_tables)
            master_list = ["store_id", "item_id", "channel", "date", "round_id"]
            try:
                ideal_output = pd.read_csv(
                    f"../../data/expected_output/Q{question_id}_output.csv"
                )
                overall_output["Expected_output"].append("True")
            except Exception:
                overall_output["Expected_output"].append("False")
            try:
                llm_output = pd.read_csv(f"{output_path}/output_table.csv")

                overall_output["llm_output"].append("True")
            except Exception:
                llm_check = False
                overall_output["llm_output"].append("False")
            if flag:
                overall_output["round_details"].append("present")
            else:
                overall_output["round_details"].append("not present")
            ideal_decimal_columns = {
                col: pd.api.types.is_float_dtype(ideal_output[col])
                for col in ideal_output.columns
            }
            ideal_decimal_columns = [
                key for key, value in ideal_decimal_columns.items() if value
            ]
            ideal_output[ideal_decimal_columns] = ideal_output[
                ideal_decimal_columns
            ].round(4)

            llm_decimal_columns = {
                col: pd.api.types.is_float_dtype(llm_output[col])
                for col in llm_output.columns
            }
            llm_decimal_columns = [
                key for key, value in llm_decimal_columns.items() if value
            ]
            llm_output[llm_decimal_columns] = llm_output[llm_decimal_columns].round(4)

            identifier_column = list(
                set(ideal_output.columns.str.lower()) & set(master_list)
            )
            ideal_number = len(ideal_output.columns)
            llm_output_copy = llm_output.copy(deep=True)
            if len(ideal_output) < len(llm_output):
                if len(identifier_column):
                    merged_df = pd.merge(
                        ideal_output,
                        llm_output,
                        how="inner",
                        on=identifier_column,
                        suffixes=("_drop", ""),
                    )
                else:
                    try:
                        merged_df = pd.merge(
                            ideal_output,
                            llm_output,
                            how="inner",
                            suffixes=("_drop", ""),
                        )
                    except Exception:
                        check_merge = False
                if check_merge:
                    columns_to_drop = merged_df.filter(regex="_drop$").columns
                    merged_df.drop(columns=columns_to_drop, inplace=True)
                    llm_output = merged_df

            try:
                ideal_output = ideal_output.sort_values(
                    by=identifier_column
                ).reset_index(drop=True)
                llm_output = llm_output.sort_values(by=identifier_column).reset_index(
                    drop=True
                )
            except Exception:
                check_identifier = False
                # overall_output["error"].append("identifier column is not present")
                # overall_output["result"].append("not matched")
                # return overall_output
            if len(ideal_output) == len(llm_output):
                check_col = 0
                check_column = llm_output.columns.values
                for col in ideal_output.columns:
                    # if col not in identifier_column:
                    comparison_results = {}
                    if col in llm_output.columns:
                        comparison_results[col] = check_in_llm(
                            ideal_output[col], llm_output[col]
                        )
                        # if column to column comparison failled
                        one_all_true = any(
                            all(values) for values in comparison_results.values()
                        )
                        if one_all_true:
                            check_column = [
                                coler for coler in check_column if coler != col
                            ]
                        else:
                            comparison_results = {}
                            for column in check_column:
                                comparison_results[column] = check_in_llm(
                                    ideal_output[col], llm_output[column]
                                )
                    else:
                        comparison_results = {}
                        for column in check_column:
                            comparison_results[column] = check_in_llm(
                                ideal_output[col], llm_output[column]
                            )
                    one_all_true = any(
                        all(values) for values in comparison_results.values()
                    )
                    if one_all_true:
                        # #for key, values in comparison_results.items():
                        #     if all(values):
                        #         if contains_all_elements(
                        #             ideal_output[col].to_list(),
                        #             llm_output[key].to_list(),
                        #         ):
                        check_col = check_col + 1
                if not check_identifier:
                    ideal_number = ideal_number - (
                        len(identifier_column)
                        - len(
                            set(identifier_column).intersection(set(llm_output.columns))
                        )
                    )
                if ideal_number <= check_col and check_identifier:
                    overall_output["result"].append("match")
                    # overall_output['result'].append('')
                    overall_output["remarks"].append(
                        "match found for identifier and non identifier columns"
                    )
                elif ideal_number <= check_col and not check_identifier:
                    overall_output["result"].append("match")
                    overall_output["remarks"].append(
                        "no match found for identifier column but match for non identifier columns"
                    )
                elif not ideal_number <= check_col and check_identifier:
                    overall_output["result"].append("no match")
                    overall_output["remarks"].append(
                        "match found for identifier column and no match for non identifier columns"
                    )
                else:
                    overall_output["result"].append("no match")
                    overall_output["remarks"].append(
                        "no match found for identifier and non identifier columns"
                    )
            else:
                overall_output["result"].append("no match")
                if check_identifier:
                    overall_output["remarks"].append(
                        "match found for identifier column and extra or missing records present"
                    )
                else:
                    overall_output["remarks"].append(
                        "no match found for identifier column and extra or missing records present"
                    )
            if check_distribution(llm_output_copy):
                overall_output["remarks"][0] = (
                    overall_output["remarks"][0]
                    + " Distribution question verify manually"
                )
            if overall_output["result"][0] == "no match":
                dfs = [ideal_output, llm_output]
                multiple_dfs(dfs, "Validation", f"{question_id}.xlsx", 1, sql_queries)

            return overall_output
        except Exception as e:
            overall_output["result"].append("not match")
            if not llm_check:
                overall_output["remarks"].append(
                    "Either expected or llm output is not present"
                )
            else:
                overall_output["remarks"].append(e)
            return overall_output


# provide the experiment name to evaluate the query
exp_name = "claude_3_5_0715_run1"
# file that contains expected query all the required questions
expected_query_file_name = "question_bank_0718_expected_query.xlsx"

directory_path = f"../../data/output_folder/{exp_name}"
folders = [
    f
    for f in os.listdir(directory_path)
    if os.path.isdir(os.path.join(directory_path, f))
]

start = time.time()
all_data = pd.DataFrame()

for question_id in range(12, 223):
    for fol in folders:
        if f"_{question_id}_" in fol:
            fold = fol
            output_path = f"{directory_path}/{fold}/01_text_to_query"
            question_path = f"{directory_path}/{fold}/question.txt"
            with open(question_path, "r") as file:
                question = file.read()
            all_data = pd.concat(
                [
                    all_data,
                    pd.DataFrame(
                        sql_evaluation(output_path, question, expected_query_file_name)
                    ),
                ]
            )

all_data["question_id"] = pd.to_numeric(all_data["question_id"])
all_data = all_data.groupby("question_id").apply(get_iteration)
all_data.drop("question_folder", errors="ignore", inplace=True, axis=1)
all_data.to_csv("updated_round_variants_2.csv")
print(f"execution time : {time.time()-start} seconds")
