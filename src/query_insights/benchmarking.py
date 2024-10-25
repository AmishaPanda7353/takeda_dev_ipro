import datetime
import json
import logging
import os
import re
import time
from datetime import timedelta

import mlflow
import mlflow.sklearn
import pandas as pd
import plotly
from azure.storage.blob import BlobSasPermissions, BlobServiceClient, generate_blob_sas
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sqlalchemy import create_engine

from .utils import load_config

MYLOGGERNAME = "QueryInsights"

question_dir = []
children_dir = {}
Blob_Upload = False
container_name = ""


def upload_to_blob(child_run: str, artifact_name: str):
    """
    Logs the artifact as tag in child runs.

    This function takes child run and artifact as parameter. It uploads the artifact in the azure blob and creates a sas token for generating download link of the artifact.
    It logs the artifact name and download url as tags within the current child run.

    Parameters:
        - child_run (str): The input name for the child run being tracked.
        - artifact_name (str): The input name for the artifact to be uploaded.
    Returns:
        str: The download url for the artifact.
    """
    # configure folder heirarchy in the blob
    artifact_path_remote = f"{question_dir[0]}/{child_run}/{artifact_name}"
    blob_client = blob_service_client.get_blob_client(
        container=container_name, blob=artifact_path_remote
    )
    # read and upload data
    with open(artifact_name, "rb") as data:
        blob_client.upload_blob(data)
    # generate sas token for download url - current config 30 days
    sas_token = generate_blob_sas(
        blob_client.account_name,
        blob_client.container_name,
        blob_client.blob_name,
        account_key=blob_client.credential.account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.datetime.utcnow() + timedelta(days=30),
    )
    download_url = f"{blob_client.url}?{sas_token}"
    return download_url


def tag_formatter(tag_name: str):
    """
    Formats the artifact name name by replacing invalid characters for mlflow tag.

    Parameters:
        - tag_name (str): The input tag name to be formatted.
    Returns:
        str: The formatted tag name with invalid characters replaced by underscores.
    """
    return re.sub(r"[^a-zA-Z0-9_\-./ ]", "_", tag_name)


# Toggle Config
config_path = f"configs/mle_config/mlflow_config.json"
with open(config_path) as stream:
    try:
        mlflow_config = json.load(stream)
        print(f"Loaded Mlflow Config - {mlflow_config}")
    except Exception as exc:
        print(exc)

if mlflow_config["azure_tracking"]:
    mlflow.set_tracking_uri(mlflow_config["database_url"])
    blob_service_client = BlobServiceClient.from_connection_string(
        mlflow_config["connection_string"]
    )
    Blob_Upload = True
    container_name = mlflow_config["container_name"]


class MLflowManager:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.experiment_id = self._get_or_create_experiment_id()
        self.run_id = None
        self.question_id = None
        self.child_run_id = None
        self.kb_inf = None
        self.track_2_ins = None
        self.track_3_ins = None
        self.logger = logging.getLogger(MYLOGGERNAME)
        self.current_time = None
        self.timestamp_str = None

    def _get_or_create_experiment_id(self):
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment:
            return experiment.experiment_id
        else:
            return mlflow.create_experiment(name=self.experiment_name)

    def set_tracking_uri(path):
        mlflow.set_tracking_uri(path)

    def start_run(self, question_index=None, question=None):
        if not self.question_id:
            self.current_time = datetime.datetime.now()
            self.timestamp_str = self.current_time.strftime("%Y%m%d%H%M%S")
            self.question_id = f"Question_ID_{self.timestamp_str}"
        else:
            pass
        if not self.run_id:
            with mlflow.start_run(
                run_name=self.question_id, experiment_id=self.experiment_id
            ) as parent_run:
                self.logger.info("mlflow started")
                self.parentrunId = mlflow.active_run().info.run_id
                self.run_id = parent_run.info.run_id
                mlflow.set_tag("experiment_name", self.experiment_name)
                mlflow.set_tag("experiment_id", self.experiment_id)
                # mlflow.set_tag("run_name", run_name)
                mlflow.set_tag("index", question_index)
                mlflow.set_tag("QuestionID", self.question_id)
                mlflow.set_tag("question", question)
            # Adding Name of parent question
            question_dir.append(self.question_id)
            # return self.run_id
        # else:
        #     mlflow.set_experiment(self.experiment_name)
        #     if question_index:
        #         with mlflow.start_run(
        #             run_id=self.run_id, experiment_id=self.experiment_id
        #         ):
        #             mlflow.set_tag("index", question_index)
        #     elif question:
        #         with mlflow.start_run(
        #             run_id=self.run_id, experiment_id=self.experiment_id
        #         ):
        #             mlflow.set_tag("question", question)

    def log_param(self, child_run_id, parent_run_id, key, value):
        # if self.run_id is not None:
        with mlflow.start_run(run_id=child_run_id, experiment_id=self.experiment_id):
            mlflow.set_tag("mlflow.parentRunId", parent_run_id)
            mlflow.log_param(key, value)

    def log_metric(self, child_run_id, parent_run_id, key, value):
        # if self.run_id is not None:
        with mlflow.start_run(run_id=child_run_id, experiment_id=self.experiment_id):
            mlflow.set_tag("mlflow.parentRunId", parent_run_id)
            mlflow.log_metric(key, value)

    def end_run(self):
        mlflow.end_run()

    # def start_child_run(self, key=None, value=None, run_name="text_to_query"):
    #     if not self.child_run_id:
    #         with mlflow.start_run(
    #             run_name=run_name, experiment_id=self.experiment_id, nested=True
    #         ) as child_run:
    #             mlflow.set_experiment(self.experiment_name)
    #             self.child_run_id = child_run.info.run_id
    #             mlflow.set_tag("mlflow.parentRunId", self.run_id)

    #     else:
    #         if key:
    #             with mlflow.start_run(
    #                 run_id=self.child_run_id, experiment_id=self.experiment_id
    #             ):
    #                 mlflow.log_param(key, value)
    #         # return self.child_run_id  # To track this specific child run ID

    def start_child_run(self, parent_run_id, run_name):
        with mlflow.start_run(run_name=run_name, nested=True) as child_run:
            mlflow.set_experiment(self.experiment_name)
            self.child_run_id = child_run.info.run_id
            mlflow.set_tag("mlflow.parentRunId", parent_run_id)
            children_dir[
                self.child_run_id
            ] = run_name  # To make nested dir inside blob with this
            return self.child_run_id  # To track this specific child run ID

    def log_artifact(
        self, child_run_id, parent_run_id, artifact_name, artifact_content
    ):
        with mlflow.start_run(run_id=child_run_id, experiment_id=self.experiment_id):
            mlflow.set_tag("mlflow.parentRunId", parent_run_id)
            if isinstance(artifact_content, str):
                with open(artifact_name, "w") as file:
                    file.write(artifact_content)
                if Blob_Upload:
                    # Blob Operation
                    start_time = time.time()
                    print(f"Satrted uploading on blob - {artifact_name}")
                    artifact_url = upload_to_blob(
                        child_run=children_dir[child_run_id],
                        artifact_name=artifact_name,
                    )
                    mlflow.set_tag(
                        tag_formatter(f"{children_dir[child_run_id]}_{artifact_name}"),
                        artifact_url,
                    )
                    os.remove(artifact_name)
                    print(f"Uploaded on blob & cleaned from local - {artifact_name}")
                    print(f"Time Elapsed: {time.time() - start_time}")
                else:
                    mlflow.log_artifact(artifact_name)
            elif isinstance(artifact_content, pd.DataFrame):
                artifact_content.to_csv(artifact_name, index=False)
                if Blob_Upload:
                    # Blob Operation
                    start_time = time.time()
                    print(f"Satrted uploading on blob - {artifact_name}")
                    artifact_url = upload_to_blob(
                        child_run=children_dir[child_run_id],
                        artifact_name=artifact_name,
                    )
                    mlflow.set_tag(
                        tag_formatter(f"{children_dir[child_run_id]}_{artifact_name}"),
                        artifact_url,
                    )
                    os.remove(artifact_name)
                    print(f"Uploaded on blob & cleaned from local - {artifact_name}")
                    print(f"Time Elapsed: {time.time() - start_time}")
                else:
                    mlflow.log_artifact(artifact_name)
            elif isinstance(artifact_content, plotly.graph_objs._figure.Figure):
                with open(artifact_name, "wb") as file:
                    plotly.io.write_image(artifact_content, file)
                    if Blob_Upload:
                        # Blob Operation
                        start_time = time.time()
                        print(f"Satrted uploading on blob - {artifact_name}")
                        artifact_url = upload_to_blob(
                            child_run=children_dir[child_run_id],
                            artifact_name=artifact_name,
                        )
                        mlflow.set_tag(
                            tag_formatter(
                                f"{children_dir[child_run_id]}_{artifact_name}"
                            ),
                            artifact_url,
                        )
                        os.remove(artifact_name)
                        print(
                            f"Uploaded on blob & cleaned from local - {artifact_name}"
                        )
                        print(f"Time Elapsed: {time.time() - start_time}")
                    else:
                        mlflow.log_artifact(artifact_name)

    def log_artifacts(self, child_run_id, parent_run_id, artifacts):
        with mlflow.start_run(run_id=child_run_id, experiment_id=self.experiment_id):
            mlflow.set_tag("mlflow.parentRunId", parent_run_id)
            for artifact_name, artifact_content in artifacts.items():
                with open(artifact_name, "w") as file:
                    if isinstance(artifact_content, str):
                        file.write(artifact_content)
                    else:
                        json.dump(artifact_content, file, indent=4)
                if Blob_Upload:
                    # Blob Operation
                    start_time = time.time()
                    print(f"Satrted uploading on blob - {artifact_name}")
                    artifact_url = upload_to_blob(
                        child_run=children_dir[child_run_id],
                        artifact_name=artifact_name,
                    )
                    mlflow.set_tag(
                        tag_formatter(f"{children_dir[child_run_id]}_{artifact_name}"),
                        artifact_url,
                    )
                    os.remove(artifact_name)
                    print(f"Uploaded on blob & cleaned from local - {artifact_name}")
                    print(f"Time Elapsed: {time.time() - start_time}")
                else:
                    mlflow.log_artifact(artifact_name)

    def log_status(self, child_run_id, parent_run_id, status, error_message=None):
        with mlflow.start_run(run_id=child_run_id, experiment_id=self.experiment_id):
            mlflow.set_tag("mlflow.parentRunId", parent_run_id)
            with open("status.txt", "w") as file:
                file.write(status)
            artifact_name = "status.txt"
            if Blob_Upload:
                # Blob Operation
                start_time = time.time()
                print(f"Satrted uploading on blob - {artifact_name}")
                artifact_url = upload_to_blob(
                    child_run=children_dir[child_run_id], artifact_name=artifact_name
                )
                mlflow.set_tag(
                    tag_formatter(f"{children_dir[child_run_id]}_{artifact_name}"),
                    artifact_url,
                )
                os.remove(artifact_name)
                print(f"Uploaded on blob & cleaned from local - {artifact_name}")
                print(f"Time Elapsed: {time.time() - start_time}")
            else:
                mlflow.log_artifact(artifact_name)

            if error_message:
                with open("Error.txt", "w") as file:
                    file.write(error_message)
                artifact_name = "Error.txt"
                if Blob_Upload:
                    # Blob Operation
                    start_time = time.time()
                    print(f"Satrted uploading on blob - {artifact_name}")
                    artifact_url = upload_to_blob(
                        child_run=children_dir[child_run_id],
                        artifact_name=artifact_name,
                    )
                    mlflow.set_tag(
                        tag_formatter(f"{children_dir[child_run_id]}_{artifact_name}"),
                        artifact_url,
                    )
                    os.remove(artifact_name)
                    print(f"Uploaded on blob & cleaned from local - {artifact_name}")
                    print(f"Time Elapsed: {time.time() - start_time}")
                else:
                    mlflow.log_artifact(artifact_name)
