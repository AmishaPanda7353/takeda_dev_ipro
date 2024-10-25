# To enable ssh & remote debugging on app service change the base image to the one below
# FROM mcr.microsoft.com/azure-functions/python:4-python3.7-appservice
FROM mcr.microsoft.com/azure-functions/python:4-python3.8

ENV AzureWebJobsScriptRoot=/home/site/wwwroot \
    AzureFunctionsJobHost__Logging__Console__IsEnabled=true


# COPY deploy/azure_function/requirements.txt requirements.txt
# RUN pip install -r /requirements.txt

# COPY deploy/ds/requirements_linux.txt requirements_linux.txt
# RUN pip install -r /requirements_linux.txt   

RUN apt-get update && apt-get install -y git
COPY deploy/requirements.txt requirements.txt
RUN pip install -r /requirements.txt  
RUN apt-get install -y wkhtmltopdf

COPY src/ /home/site/wwwroot
COPY ./azure_function /home/site/wwwroot

# CMD ['func','host start']
