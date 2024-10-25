from typing import Optional

import yaml
from pydantic import BaseModel
from core.utils.read_config import cloud_secrets


class AccountAndStorage(BaseModel):
    account_name: Optional[str]
    storage_name: Optional[str]


class CloudConfig(BaseModel):
    cloud_provider: str
    domain_storage: AccountAndStorage
    reports_storage: AccountAndStorage


class DomainName(BaseModel):
    domain_name: str


def validate_cloud_config(config_file: str, domain_file: str):
    with open(domain_file, "r") as f:  #reading the domain file to get the domain name
        domain_file = yaml.safe_load(f)

    # # Load the configuration file
    with open(config_file, "r") as f:  # reading the config file
        config_file = yaml.safe_load(f)

    domain_name = DomainName(**domain_file)
    domain_name = domain_name.domain_name  #Taking the domain name from the domain_file

    config = config_file.get("cloud_details", {})  # Taking all the cloud_details from the cloud config

    if (domain_name not in config.keys()): # Checking if the domain is present in cloud config or not.
        raise ValueError(f"Domain '{domain_name}' not found in cloud configuration.")

    domain_config = config.get(domain_name, {})  # Taking the cloud_details only for the domain name which is from domain_file.
    config = CloudConfig(**domain_config)
    # print(config.cloud_details)
    if (config.cloud_provider == "azure"):
        if ((config.domain_storage.account_name is None)
            or (config.domain_storage.storage_name is None)
            or (config.reports_storage.account_name is None)
            or (config.reports_storage.storage_name is None)
            or (cloud_secrets.domain_storage.connection_key is None)
            or (cloud_secrets.reports_storage.connection_key is None)):
            raise ValueError("Please check for the connection keys and domain_storage or reports_storage should not contain none")
        else:
            return
    elif (config.cloud_provider == "s3"):
        if ((config.domain_storage.account_name is None)
            or (config.domain_storage.storage_name is not None)
            or (config.reports_storage.account_name is not None)
            or (config.reports_storage.storage_name is not None)):
            raise ValueError("Please check for the connection keys and domain_storage or reports_storage should not contain none")
        else:
            return
         
    else:
        if ((config.domain_storage.account_name is not None)
            or (config.domain_storage.storage_name is not None)
            or (config.reports_storage.account_name is not None)
            or (config.reports_storage.storage_name is not None)):
            raise ValueError("domain_storage or reports_storage should not contain any values for the account name and storage name")
        # print(config)
        return


# domain_name = "configs/config.yaml"
# cloud_config = "configs/cloud/cloud.yaml"

# path_validation = validate_cloud_config(cloud_config, domain_name)
