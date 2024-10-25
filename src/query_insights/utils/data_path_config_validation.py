from typing import Optional, Dict, List

import yaml
from pydantic import BaseModel


class Data_Path(BaseModel):
    input_data_path: str
    exclude_table_names: Optional[List[str]] = None
    input_data_tables: Optional[List[str]] = None
    input_file_names: Optional[Dict[str, str]] = None
    data_dictionary_path: str
    business_overview_path: Optional[str] = None
    api_key_location: str
    output_path: str
    exp_name: str


class Validate_Data_Path(BaseModel):
    path: Data_Path


class DomainName(BaseModel):
    domain_name: str


def validate_data_path(config_file: str, domain_file: str):
    with open(domain_file, "r") as f:  # Reading the domain file to take domain name
        domain_file = yaml.safe_load(f)

    # # Load the configuration file
    with open(config_file, "r") as f:  # reading the config file
        config_file = yaml.safe_load(f)

    domain_name = DomainName(**domain_file)
    domain_name = domain_name.domain_name  # Taking only the domain name from domain_file

    data_path_config = config_file.get("data_config_domains", {})  # Getting all the detals of data_config_domains from config_file
    if (domain_name not in data_path_config.keys()):  # Checking if the domain name is present or not
        raise ValueError(f"Domain '{domain_name}' not found in cloud configuration.")

    data_path_config = data_path_config.get(domain_name, {})  # Taking only data_config_domains for the particular domain based on domain name.
    # config = Validate_Data_Path(**data_path_config)
    # if (len(config.path.input_data_tables.reporting) != len(config.path.input_file_names)):
    #     raise ValueError("We dont need to have inputfile names or inputdata names please verify!")  
    # if (len(config.path.input_data_tables.historical) != len(config.path.input_file_names)):
    #     raise ValueError("We dont need to have inputfile names or inputdata names please verify!")
    # print(config)
    return


# domain_name = "configs/config.yaml"
# data_path = "configs/data/data_path.yaml"

# path_validation = validate_data_path(data_path, domain_name)
