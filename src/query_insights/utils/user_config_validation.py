from typing import Optional, List, Any

import yaml
from pydantic import BaseModel
from datetime import date


class ConnectionParams(BaseModel):
    api_type: Any
    api_base: str
    api_version: str


class UserInputs(BaseModel):
    question: Optional[str] = None
    additional_context: Optional[str] = None


class Similarity_Check(BaseModel):
    model: str
    threshold: int
    num_columns_per_chunk: int


class UserConfig(BaseModel):
    ui: bool
    skip_api_call: bool
    skip_list: Optional[List[str]]
    connection_params: ConnectionParams
    user_inputs: UserInputs
    table_top_rows: int
    why_question_threshold: Optional[int] = None
    time_delay: int
    today: date
    bot_response: str
    similarity_check: Similarity_Check


class DomainName(BaseModel):
    domain_name: str


def validate_user_config(config_file: str, domain_file: str):
    with open(domain_file, "r") as f:  # Reading the domain file to take domain name
        domain_file = yaml.safe_load(f)

    # # Load the configuration file
    with open(config_file, "r") as f:  # Reading the config file
        config_file = yaml.safe_load(f)

    domain_name = DomainName(**domain_file)
    domain_name = domain_name.domain_name  # Taking the domain name from the domain file

    user_config = config_file.get("user_config_domains", {})  # Taking all the user_config_domain from the config file

    if (domain_name not in user_config.keys()):  # Checking if the domain name is present or not.
        raise ValueError(f"Domain '{domain_name}' not found in cloud configuration.")

    user_config = user_config.get(domain_name, {})  # Taking user_config_domain based on the domain name which is present in domain_file
    config = UserConfig(**user_config)
    return


# domain_name = "configs/config.yaml"
# user_config = "configs/user/user.yaml"

# path_validation = validate_user_config(user_config, domain_name)
