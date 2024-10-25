from typing import Optional

import yaml
from pydantic import BaseModel


class DomainConfig(BaseModel):
    domain_database: str
    db_file_path: Optional[str] = None
    database_name: Optional[str] = None
    host: Optional[str] = None
    port: Optional[str] = None
    user: Optional[str] = None


class DomainName(BaseModel):
    domain_name: str


def validate_database_config(config_file: str, domain_file: str):
    with open(domain_file, "r") as f: # reading the domain name
        domain_file = yaml.safe_load(f)

    # # Load the configuration file
    with open(config_file, "r") as f: # reading the database config file
        config_file = yaml.safe_load(f)

    domain_name = DomainName(**domain_file)
    domain_name = domain_name.domain_name  # taking the domain name from domain_file

    config = config_file.get("database_details", {})  #gettting the database_details  from the config
    if (domain_name not in config.keys()):
        raise ValueError(f"Domain '{domain_name}' not found in cloud configuration.")
   
    domain_config = config.get(domain_name, {})  # getting the database_details only for the domain name.
    config = DomainConfig(**domain_config)

    if (config.domain_database == "sqlite"):
        if ((config.db_file_path is None)):
            raise ValueError("db_file_path should not contain None value it should please provide path")
        else:
            return
    else:
        if (config.db_file_path is not None):
            raise ValueError("db_file_path should not contain value it should be None")
        return


# domain_name = "configs/config.yaml"
# database_config = "configs/database/database.yaml"

# path_validation = validate_database_config(database_config, domain_name)
