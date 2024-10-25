import base64
import json
import boto3
from botocore.exceptions import ClientError
import os
import yaml
from dynaconf import Dynaconf


# def get_config_data():
#     path = os.getcwd()
#     print(path)
#     f = open(path + "/configs/database/database.yaml")
#     data = yaml.safe_load(f)
#     return data


# CONFIG = get_config_data()


class SecretManager:
    def __init__(self, config):
        # self.data = get_config_data()
        self.region_name = config.region_name
        self.secret_name = config.secret_name
        if "database_name" in config.keys():
            self.database_name = config.database_name

    def get_secret(self, secret_name):
        session = boto3.session.Session()
        client = session.client(
            service_name="secretsmanager", region_name=self.region_name
        )

        # In this sample we only handle the specific exceptions for the 'GetSecretValue' API.
        # See https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        # We rethrow the exception by default.

        try:
            get_secret_value_response = client.get_secret_value(SecretId=secret_name)
        except ClientError as e:
            if e.response["Error"]["Code"] == "DecryptionFailureException":
                # Secrets Manager can't decrypt the protected secret text using the provided KMS key.
                # Deal with the exception here, and/or rethrow at your discretion.
                raise e
            elif e.response["Error"]["Code"] == "InternalServiceErrorException":
                # An error occurred on the server side.
                # Deal with the exception here, and/or rethrow at your discretion.
                raise e
            elif e.response["Error"]["Code"] == "InvalidParameterException":
                # You provided an invalid value for a parameter.
                # Deal with the exception here, and/or rethrow at your discretion.
                raise e
            elif e.response["Error"]["Code"] == "InvalidRequestException":
                # You provided a parameter value that is not valid for the current state of the resource.
                # Deal with the exception here, and/or rethrow at your discretion.
                raise e
            elif e.response["Error"]["Code"] == "ResourceNotFoundException":
                # We can't find the resource that you asked for.
                # Deal with the exception here, and/or rethrow at your discretion.
                raise e
        else:
            # Decrypts secret using the associated KMS CMK.
            # Depending on whether the secret is a string or binary, one of these fields will be populated.
            if "SecretString" in get_secret_value_response:
                secret = get_secret_value_response["SecretString"]
                return secret

            else:
                decoded_binary_secret = base64.b64decode(
                    get_secret_value_response["SecretBinary"]
                )

    # Gets DB details stores in secret manager
    def mcd_reporting_database_secret(self):
        #secret_name = self.config["secret_name"]
        db_secrets = json.loads(self.get_secret(self.secret_name))
        db_secrets["database_name"] = self.database_name
        settings = Dynaconf(settings_files=[], environments=True, load_dotenv=False)
        settings.set('MY_DICT', db_secrets)
        dot_dict = settings.MY_DICT
        dot_dict.domain_database = dot_dict.pop('engine')
        dot_dict.password_path = dot_dict.pop('password')
        return dot_dict
    
    def mcd_historical_database_secret(self):
        db_secrets = json.loads(self.get_secret(self.secret_name))
        settings = Dynaconf(settings_files=[], environments=True, load_dotenv=False)
        settings.set('MY_DICT', db_secrets)
        dot_dict = settings.MY_DICT
        return dot_dict

    # Gets DB details stores in secret manager
    def mcd_athena_database_secret(self):
        #secret_name = self.config["secret_name"]
        db_secrets = json.loads(self.get_secret(self.secret_name))
        db_secrets["database_name"] = self.database_name
        settings = Dynaconf(settings_files=[], environments=True, load_dotenv=False)
        settings.set('MY_DICT', db_secrets)
        dot_dict = settings.MY_DICT
        dot_dict.domain_database = dot_dict.pop('engine')
        dot_dict.password_path = dot_dict.pop('password')
        return dot_dict

    def app_database_secret(self):
        #secret_name = self.config["secret_name"]
        db_secrets = json.loads(self.get_secret(self.secret_name))
        db_secrets["database_name"] = self.database_name
        settings = Dynaconf(settings_files=[], environments=True, load_dotenv=False)
        settings.set('MY_DICT', db_secrets)
        dot_dict = settings.MY_DICT
        dot_dict.app_db_name = dot_dict["username"]
        return dot_dict

    # # Gets DJango API key from secret manager
    # def django_secret_key(self):
    #     secret_name = self.data['django_secret_key']['secret_name']
    #     key = self.data['django_secret_key']['key_name']
    #     api_secret_key = json.loads(self.get_secret(secret_name))[key]
    #     return api_secret_key

    # # Gets Email credentials from secret manager
    # def email_credentials(self):
    #     secret_name = self.data["email_credentials"]["secret_name"]
    #     email_secrets = json.loads(self.get_secret(secret_name))
    #     return email_secrets

    # def s3_secret(self):
    #     secret_name = self.data["s3_config"]["secret_name"]
    #     s3_secrets = json.loads(self.get_secret(secret_name))
    #     return s3_secrets


# obj = SecretManager(CONFIG)
# a = obj.mcd_database_secret()
# print(a.keys())
