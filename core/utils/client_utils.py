from importlib import import_module

from core.database.database_client import DatabaseClient
from core.entity_extraction.entity_extraction_client import EntityExtractionClient
from core.model.model_client import Model
from core.retrievers.retrievers_client import Retrievers
from core.storage.storage_client import StorageClient
from core.utils.read_config import config
from core.vectordb.vectordb_client import VectorDB
from core.vectordb.pgvectordb_client import PGVectorDB
import logging


# Pass here only database config (app_database_config / domain_database_config)
def get_database_client(database_type: str):
    """
    Returns an instance of the appropriate database client based on the given database_type.

    Args:
        database_type (str): The type of the database client to retrieve. Supported values are "postgres" and "sqlite".

    Returns:
        DatabaseClient: An instance of the appropriate database client.

    Raises:
        ValueError: If an unsupported cloud service is specified.

    """
    logging.info("Loaded database configs inside utils")

    if database_type == "postgres":
        module_name = "core.clients.aws.database.postgres_client"
        class_name = "AzurePostgresClient"
    elif database_type == "sqlite":
        module_name = "core.clients.azure.database.sqlite_client"
        class_name = "SqliteClient"
    elif database_type == "hivedb":
        module_name = "core.clients.azure.database.hive_metastore_client"
        class_name = "HiveMetaStore"
    elif database_type == "mysql":
        module_name = "core.clients.aws.database.mysql_client"
        class_name = "MySqlClient"
    elif database_type == "athena":
        module_name = "core.clients.aws.database.athena_client"
        class_name = "AthenaClient"
    else:
        raise ValueError(f"Unsupported database type : {database_type}")

    module = import_module(module_name)
    database_service_client = getattr(module, class_name)

    if issubclass(database_service_client, DatabaseClient):
        return database_service_client()
    else:
        raise ValueError("Invalid database service client")


def get_entity_extraction_client(entity_extraction_config, data_dictionaries):
    """
    Returns an instance of the appropriate entity extraction client based on the given entity_extraction_type.

    Args:
        entity_extraction_config (dict): The configuration object containing entity extraction information.
        data dictionaries (dict): data dictionaries
    Returns:
        EntityExtractionClient: An instance of the appropriate entity_extraction client.

    Raises:
        ValueError: If an unsupported entity_extraction service is specified.

    """

    if entity_extraction_config.entity_extraction_type.lower() == "keybert":
        module_name = "core.clients.aws.entity_extraction.keybert_client"
        class_name = "KeyBert"
    else:
        raise ValueError(f"Unsupported EntityExtraction service")

    # Import the module dynamically based on the module name
    module = import_module(module_name)
    # Get the class object for the EntityExtraction service client dynamically based on the class name
    entity_extraction_service_client = getattr(module, class_name)

    if issubclass(entity_extraction_service_client, EntityExtractionClient):
        return entity_extraction_service_client(
            entity_extraction_config, data_dictionaries
        )
    else:
        raise ValueError("Invalid EntityExtraction service client")


def get_vectordb_client(vectordb_config, data_frame, module_name=None):
    """
    Returns an instance of the appropriate vectordb client based on the given vectordb_type.

    Args:
        vectordb_config (dict): The configuration object containing vectordb information.
        data_frame (pd.Dataframe): Dataframe having unique index for unique colum name and other information related to that column,
                                e.g. column description, table name etc.

    Returns:
        VectorDBClient: An instance of the appropriate vectordb client.

    Raises:
        ValueError: If an unsupported cloud service is specified.

    """
    if vectordb_config.vectordb_type.lower() == "milvusdb":
        module_name = "core.clients.aws.vectordb.milvusdb_client"
        class_name = "MilvusDB"
    elif vectordb_config.vectordb_type.lower() == "pgvector":
        module_name = "core.clients.aws.vectordb.pgvector_client"
        class_name = "PGVector"
    else:
        raise ValueError(f"Unsupported VectorDB service")

    # Import the module dynamically based on the module name
    module = import_module(module_name)

    # Get the class object for the vectorDB service client dynamically based on the class name
    vectordb_service_client = getattr(module, class_name)

    if issubclass(vectordb_service_client, VectorDB):
        return vectordb_service_client(vectordb_config, data_frame, module_name)
    elif issubclass(vectordb_service_client, PGVectorDB):
        return vectordb_service_client(vectordb_config, data_frame, module_name)
    else:
        raise ValueError("Invalid VectorDB service client")


def get_retriever_client(retriever_config, data_frame):
    """
    Returns an instance of the appropriate retriever client based on the given retrievers_type.

    Args:
        retriever_config (dict): The configuration object containing retriever information.
        data_frame (pd.Dataframe): Dataframe having unique index for unique colum name and other information related to that column,
                                e.g. column description, table name etc.

    Returns:
        RetrieverClient: An instance of the appropriate retriever client.

    Raises:
        ValueError: If an unsupported cloud service is specified.

    """

    if retriever_config.retriever_type.lower() == "bm25":
        module_name = "core.clients.aws.retrievers.bm25_client"
        class_name = "BM25"
    elif retriever_config.retriever_type.lower() == "tfidf":
        module_name = "core.clients.aws.retrievers.tfidf_client"
        class_name = "TFIDF"
    else:
        raise ValueError(f"Unsupported Retrievers service")

    # Import the module dynamically based on the module name
    module = import_module(module_name)
    # Get the class object for the Retrievers service client dynamically based on the class name
    retriever_service_client = getattr(module, class_name)
    if issubclass(retriever_service_client, Retrievers):
        return retriever_service_client(retriever_config, data_frame)
    else:
        raise ValueError("Invalid Retrievers service client")


def get_storage_client(cloud_config):
    """
    Factory function to create a storage service client based on the cloud provider.

    Parameters:
        config (Dynaconf): The configuration object containing cloud provider information.

    Returns:
        StorageService: An instance of the appropriate storage service client based on the cloud provider.

    Raises:
        ValueError: If an invalid cloud provider is provided in the configuration.
    """
    # Check if the cloud provider is Azure
    if cloud_config.cloud_provider.lower() == "azure":
        # Define the module name and class name for Azure Blob Storage client
        module_name = "core.clients.azure.storage.blob_storage_client"
        class_name = "AzureBlobStorageClient"
    elif cloud_config.cloud_provider.lower() == "s3":
        module_name = "core.clients.aws.storage.s3_client"
        class_name = "AWSStorageClient"
    else:
        # Define the module name and class name for Azure Blob Storage client
        module_name = "core.clients.azure.storage.blob_storage_client"
        class_name = "AzureBlobStorageClient"

        # # Raise an error for an unsupported cloud provider
        # raise ValueError(f"Unsupported cloud provider: {config.cloud_provider}")

    # Import the module dynamically based on the module name
    module = import_module(module_name)
    # Get the class object for the storage service client dynamically based on the class name
    storage_service_client = getattr(module, class_name)
    # Return an instance of the storage service client
    return storage_service_client()


def get_model_type(
    config,
    prompt_dict,
    question,
    additional_context,
    connection_param_dict,
    track,
    user_config,
    language,
    data_dictionary,
    business_overview,
    db_params=None,
    sample_input=None,
    code_result=None,
    table=None,
    history=None,
    error_message=None,
):
    """
    Returns an instance of the appropriate model client based on the given configuration.

    Args:
        config (dict): The configuration dictionary containing the model type.
        prompt_dict (dict): The dictionary containing prompts for the model.
        question (str): The question to be answered by the model.
        additional_context (str): Additional context for the model.
        connection_param_dict (dict): The dictionary containing connection parameters.
        user_config (dict): The user configuration dictionary.
        language (str): The language for the model.
        data_dictionary (dict): The dictionary containing data related to the model.
        business_overview (str): The business overview for the model.
        db_params (dict, optional): The dictionary containing database parameters. Defaults to None.
        sample_input (str, optional): The sample input for the model. Defaults to None.
        code_result (str, optional): The result of the code execution. Defaults to None.
        table (str, optional): The table name for the model. Defaults to None.

    Returns:
        Model: An instance of the appropriate model client.

    Raises:
        ValueError: If the model type is unsupported or the model client is invalid.
    """
    model_dict = {
        "text_to_query": user_config.connection_params.api_type.text_to_query,
        "query_to_chart_type": user_config.connection_params.api_type.query_to_chart_type,
        "query_to_chart_code": user_config.connection_params.api_type.query_to_chart_code,
        "table_to_insight_questions": user_config.connection_params.api_type.table_to_insight_questions,
        "insight_questions_to_code": user_config.connection_params.api_type.insight_questions_to_code,
        "summarize_tables": user_config.connection_params.api_type.summarize_tables,
    }

    if model_dict[track] == "openai":
        module_name = "core.clients.azure.model.openai_client"
        class_name = "GPTModelCall"
    elif model_dict[track] == "aws_mistral":
        module_name = "core.clients.aws.model.opensource_mistral_client"
        class_name = "OpenSourceMistralModelCall"
    elif model_dict[track] == "aws_llama":
        module_name = "core.clients.aws.model.opensource_llama_client"
        class_name = "OpenSourceLlamaModelCall"
    elif model_dict[track] == "aws_claude_2":
        module_name = "core.clients.aws.model.opensource_claude2_client"
        class_name = "OpenSourceClaude2ModelCall"
    elif model_dict[track] == "aws_claude_3":
        module_name = "core.clients.aws.model.opensource_claude3_client"
        class_name = "OpenSourceClaude3ModelCall"
    elif model_dict[track] == "aws_claude_3_5":
        module_name = "core.clients.aws.model.opensource_claude3_5_client"
        class_name = "OpenSourceClaude3_5ModelCall"

    else:
        raise ValueError(f"Unsupported model: {config.llm_model_type}")

    module = import_module(module_name)
    model_client = getattr(module, class_name)

    if issubclass(model_client, Model):
        return model_client(
            prompt_dict=prompt_dict,
            question=question,
            additional_context=additional_context,
            connection_param_dict=connection_param_dict,
            track=track,
            user_config=user_config,
            language=language,
            db_param_dict=db_params,
            dictionary=data_dictionary,
            business_overview=business_overview,
            sample_input=sample_input,
            suggestion=code_result,
            table=table,
            history=history,
            error_message=error_message,
        )
    else:
        raise ValueError("Invalid model client")
