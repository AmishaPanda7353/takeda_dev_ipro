from app.aws_secrets import SecretManager
from dynaconf import Dynaconf
from src.query_insights.utils.time_logging import timing_decorator

# log_stage_start, log_stage_end, set_current_track
opensource_config = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=[
        "configs/config.yaml",
        "configs/model.yaml",
        "configs/cloud/cloud.yaml",
        "configs/database/database.yaml",
        "configs/model/model_db_prompts.yaml",
        "configs/model/model_domain_prompts.yaml",
        "configs/model/model_params.yaml",
        "configs/model/opensource_prompts.yaml",
        "configs/user/user.yaml",
        "configs/data_files/data_path.yaml",
        "configs/debug_code.yaml",
        "configs/embeddings.yaml",
        "configs/retrievers.yaml",
        "configs/entity_extraction.yaml",
    ],
)

openai_config = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=[
        "configs/config.yaml",
        "configs/model.yaml",
        "configs/cloud/cloud.yaml",
        "configs/database/database.yaml",
        "configs/model/model_db_prompts.yaml",
        "configs/model/model_domain_prompts.yaml",
        "configs/model/model_params.yaml",
        "configs/model/openai_prompts.yaml",
        "configs/user/user.yaml",
        "configs/data_files/data_path.yaml",
        "configs/debug_code.yaml",
        "configs/embeddings.yaml",
        "configs/retrievers.yaml",
        "configs/entity_extraction.yaml",
    ],
)
if openai_config.llm_model_type == "openai":
    config = openai_config
else:
    config = opensource_config

try:
    secrets_config = Dynaconf(
        envvar_prefix="DYNACONF",
        settings_files=[
            "configs/cloud/.cloud_secrets.yaml",
            "configs/database/.db_secrets.yaml",
        ],
    )
except Exception:
    raise Exception(".cloud_secrets.yaml and .db_secrets.yaml file not found")

# Accessing the configuration dynamically based on the domain name variable
try:
    cloud_config = getattr(config.cloud_details, config.domain_name)
    db_config = getattr(config.database_details, config.domain_name)
    cloud_secrets = getattr(secrets_config.cloud_details, config.domain_name)

    app_database = getattr(config.database_details, "app_database")
    app_database_secrets = SecretManager(app_database)
    app_db_config = app_database_secrets.app_database_secret()
    # print(app_db_config, "***********")
    app_database_config = app_db_config

    mcd_db_config = getattr(config.database_details, config.domain_name)

    # _mysql_database
    mysql_database_config = mcd_db_config.reporting_db
    mcd_secrets = SecretManager(mysql_database_config)
    domain_db_config = mcd_secrets.mcd_reporting_database_secret()
    config.database_details[config.domain_name]["reporting_db"].update(domain_db_config)

    # athena_database
    athena_database_config = mcd_db_config.historical_db
    secrets = SecretManager(athena_database_config)
    domain_db_config = secrets.mcd_historical_database_secret()
    domain_db_config["domain_database"] = "athena"
    config.database_details[config.domain_name]["historical_db"].update(
        domain_db_config
    )

    database_config = config.database_details[config.domain_name]

    db_secrets = getattr(secrets_config.database_details, config.domain_name)
    model = config.llm_model_type
    deployment_url_config = getattr(config.user_config_domains, "deployment_url")

except ValueError:
    raise ValueError(f"No configuration found for domain '{config.domain_name}'")
except Exception as e:
    print(f"exception occured : {e}")


# Model config after restructuring
def process_guidelines(text):
    """
    Process the guidelines in the given text and return a list of updated lines.

    Args:
        text (str): The input text containing guidelines.

    Returns:
        list: A list of updated lines with guideline numbers.

    """
    updated_lines = ["Follow the below mentioned guidelines while generating response:"]
    # Process each line individually
    line = text.split("\n")
    line = [lin.replace("\r", "") for lin in line]
    k = 1
    for i, j in enumerate(line):
        parts = j.split(":", 1)
        if len(parts) == 2:
            num, text = parts
            if text.strip() == "":
                continue
            num = "Guideline " + str(k)
            k = k + 1
            updated_lines.append(num + ": " + text)

    return "\n".join(updated_lines)


def model_config_generator():
    """
    Generates a model configuration based on a list of parameters.

    Returns:
        model_config1 (Dynaconf): The generated model configuration.
    """
    if model == "openai":
        keys_to_check = Dynaconf(
            settings_files=["../configs/model/openai_prompts.yaml"],
            environments=False,
        )
    else:
        keys_to_check = Dynaconf(
            settings_files=["../configs/model/opensource_prompts.yaml"],
            environments=False,
        )
    list1 = list(
        keys_to_check.as_dict(internal=False).keys()
    )  # Getting all the track keys
    track_names = [
        "text_to_query",
        "query_to_chart_type",
        "query_to_chart_code",
        "table_to_insight_questions",
        "insight_questions_to_code",
        "summarize_tables",
        "summarize_insights",
        "questions_to_insights",
    ]
    model_config = Dynaconf()
    model_config._store.clear()  # removing the default values in dynaconf
    for a in list1:
        b = dict(getattr(config, a))  # getting the prompts using the track name
        if (config.domain_name == "mcd") and (a.lower() in track_names):
            c = getattr(config.user_config_domains.mcd.connection_params.api_type, a)
        else:
            c = config.llm_model_type
        param = getattr(
            config.param_dict, a
        )  # getting the param dict using the track name which are required.
        param_dict = dict(
            getattr(param, c)  # llm_model_type
        )  # Getting the param dict using the llm model type
        model_path = getattr(
            config.database_details, config.domain_name
        )  # getting the model path with the domain name
        db = getattr(
            model_path, "domain_database"
        )  # Taking the database name from the model path
        db_guidelines1 = getattr(
            config.db_config, a
        )  # taking the guidelines from db config using track name
        db_guidelines = dict(
            getattr(db_guidelines1, db)
        )  # taking the guidelines from db config using db name
        domain = getattr(config, "domain_name")  # getting the domain name
        domain_guidelines = getattr(
            config.domain_config, domain
        )  # getting the domain guidelines from domain config using domain name
        domain_track_guidelines = dict(getattr(domain_guidelines, a))
        b["prompts"] = dict(b["prompts"])
        b.update(param_dict)
        if b["prompts"]["guidelines"] is None or b["prompts"]["guidelines"] == "":
            b["prompts"]["guidelines"] = "NO_GUIDELINES"
        if db_guidelines["guidelines"] is None or db_guidelines["guidelines"] == "":
            db_guidelines["guidelines"] = "NO_GUIDELINES"
        if (
            domain_track_guidelines["guidelines"] is None
            or domain_track_guidelines["guidelines"] == ""
        ):
            domain_track_guidelines["guidelines"] = "NO_GUIDELINES"
        if "TEXT_TO_QUERY" in a:
            domain_track_guidelines["guidelines"] = "NO_GUIDELINES"
            b["prompts"]["guidelines"] = b["prompts"]["guidelines"] + "Athena_guide_sep"
        b["prompts"]["guidelines"] += (
            db_guidelines["guidelines"] + domain_track_guidelines["guidelines"]
        )
        b["prompts"]["guidelines"] = process_guidelines(
            b["prompts"]["guidelines"]
        )  # combining all the guidelines and changing the numbering
        b["prompts"]["guidelines"] = b["prompts"]["guidelines"].replace(
            "NO_GUIDELINES", ""
        )
        model_config[a] = b

    return model_config


@timing_decorator(track_app_start=False)
def initialize_config():
    """
    Initializing all the configaration files.
    """
    #  TODO: Converting dict object to be accessible using dot operator.
    # set_current_track("track1")
    # log_stage_start("Config_loading", 'Pre_track')
    user_config = getattr(config.user_config_domains, config.domain_name)
    data_config = getattr(config.data_config_domains, config.domain_name)
    model_config = model_config_generator()
    debug_config = config.debug_code
    # log_stage_end("Config_loading", "Pre_track")
    return user_config, data_config, model_config, debug_config
