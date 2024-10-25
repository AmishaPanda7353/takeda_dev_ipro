# Insights Pro - GenAI Capabilities (Tiger Agents)

Insights Pro is an easy-to-engage conversational agent for generating context-aware business insights from data. This is powered by Large Language Models (LLMs) which, based on their semantic understanding of a question, on-demand data analyses including those that require programming-based approaches. These range from exploratory analyses to complex visualizations and advanced ML models. Insights Pro interprets the questions or a tree of thought connecting multiple data tables and formulates complex table joins. It also deciphers the right metrics for a user persona and explains them through the proper visualization.

The main pre-requisites for InsightsPro include `Data dictionary`, `Business context`, `Questions`, `Language`, `Additional context`, and `Configurations`.

<p>&nbsp;</p>

## Installation

You can install the Insights Pro package by following these steps:

1. Setup miniconda by following the instructions given [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)

2. Create a new environment by running the below command in the terminal
```bash
conda create --name ta_ipro python=3.8
```

3. Activate the conda environment
```bash
conda activate ta_ipro
```

4. Clone this repository to your local machine:

```bash
git clone https://github.com/tigerrepository/Insights-Pro
```

2. Navigate to the root directory of the cloned repository
```bash
cd Insights-Pro
```

3. Install the package in editable mode using pip
```bash
pip install -e .
```
This will install the Insights Pro package and make it editable, allowing you to modify the source code if needed.

#### Usage:
Once the package is installed, you can import and use its modules and classes in your Python projects. For example:
```bash
import core
from src import query_insights
```

<p>&nbsp;</p>

## Steps to run

### 1. Update the config files
##### Note: 
The configuration changes listed below apply to Insights Pro as it is currently configured. If your setup information change, you have the flexible option to update the configuration details accordingly.

#### For Backend
1. You must select the specific domain for which you want to use. You should add the domain name in **config.yaml** to run backend.

#### To use data/ from local:
1. You must maintain the entire **data/** folder in the project's root directory as a prerequisite.

2. Navigate to `configs/cloud/cloud.yaml`, choose the `domain` you are using, set `cloud_provider` to `local`, and set remaining keys to empty.
```bash
cloud_provider: local 
domain_storage:
    account_name: 
    storage_name:   
reports_storage:
    account_name: 
    storage_name: 
```

3. Navigate to `configs/database/database.yaml`, choose the `domain` you are using, and keep the `db_file_path` as below.
```bash
db_file_path: "../<path>"
```

4. Navigate to `configs/data_files/data_path.yaml`, choose the `domain` you're using, and alter the paths as indicated below. For example:
```bash
input_data_path: ../data/supply_chain_management/db
data_dictionary_path: ../data/supply_chain_management/data_dictionary
api_key_location: ../../tmp/azure_api_key_gpt4
output_path: /data/supply_chain_management/output_folder
exp_name: insights_pro
```
#### For Frontend 

When the app is built at that time, it needs to update the config in the `.env.development` file ui\react\.env.development

- For Local you need to add a `.env` file
```bash
REACT_APP_API_URL="http://127.0.0.1:8000/" 
REACT_APP_REDIRECT_URL="http://localhost:3000"
REACT_APP_MS_TENANTID=""
REACT_APP_CLIENT_ID=""
```
- For Other than Local we have the `.env.development` file
  ```bash
  REACT_APP_API_URL="http://ipro-app-service.default.svc.cluster.local:80/" 
  REACT_APP_REDIRECT_URL="http://localhost:3000"
  REACT_APP_MS_TENANTID=""
  REACT_APP_CLIENT_ID=""
  ```
  Ask for the tenant ID and Client ID.

<p>&nbsp;</p>

#### To use data/ from Azure cloud:
1. You must maintain the entire **data/** folder in the azure blob storage as a prerequisite.

2. Navigate to `configs/cloud/cloud.yaml`, choose the `domain` you are using, set `cloud_provider` to `azure`, and then update the values of `account_name` and `storage name`.
```bash
cloud_provider: azure  
domain_storage:
    account_name: nlpdevreact
    storage_name: data  
reports_storage:
    account_name: nlpdevreact
    storage_name: backend
```

3. Navigate to `configs/database/database.yaml`, choose the `domain` you are using, and keep the `db_file_path` as below.
```bash
db_file_path: "<path>"
```

4. Navigate to `configs/data_files/data_path.yaml`, choose the `domain` you're using, and alter the paths as indicated below. For example:
```bash
input_data_path: /data/supply_chain_management/db
data_dictionary_path: /data/supply_chain_management/data_dictionary
api_key_location: ./../tmp/azure_api_key_gpt4
output_path: /data/supply_chain_management/output_folder_test
exp_name: insights_pro
```

#### Pre-requisite secret files

1. The configs folder should contain the two secret files `.cloud_secrets.yaml` and `.db_secrets.yaml` that are necessary. `.cloud_secrets.yaml` should be placed in `configs/cloud` and `.db_secrets.yaml` should be placed in `configs/database`.
2. None of the secret files are added to the git repository. To get those secret files, please reach out to `durga.koliparthi@tigeranalytics.com` or `gangadhar.shanka@tigeranalytics.com`

<p>&nbsp;</p>

###  2. To run the backend FastAPI server locally
1. Navigate to the `app` folder from the root folder, then launch the FastAPI server locally.
```bash
cd app

uvicorn ipro_app:app
```

2. After running the above command, open the link in the browser to access Swagger UI. For example:
```bash
http://127.0.0.1:8000/docs
```
#### To run the Frontend server locally
1. Install the node modules.
   ```bash
   npm install
   ```
2. To run the server
   ```bash
   npm start
   ```
      
<p>&nbsp;</p>

### 3. To run using Docker

#### To Run Backend:
1. Create the docker image from the dockerfile. Use below command to create the docker image.
docker build -t <image_name> -f <dockerfile_path> .
Example:
```bash
docker build -t insights_pro:latest -f deploy/docker/Dockerfile_ipro .
```
As a result, the image `insights_pro` will be created.

2. Create a docker container using the above image.
docker run -itd --rm --name <container_name> -p 80:80 <image_name>
Example:
```bash
docker run -itd --name insights_pro_container -p 80:80 insights_pro:latest
```
This will create a container called `insights_pro_container` from the image `insights_pro:latest`. Port number 80 is to access the container from local and inside container.

<p>&nbsp;</p>

#### To Run Frontend:

1.Create the docker image from the dockerfile. Use below command to create the docker image. 
docker build -t <image_name> -f <dockerfile_path> . Example:
```bash
 docker build -t demoacr4312.azurecr.io/ipro_fe:v1 -f deploy/docker/Dockerfile_fe .
```
As a result, the image `demoacr4312.azurecr.io/ipro_fe:v1` will be created.

2. Create the docker image from the dockerfile. Use below command to create the docker image.
docker run -itd --rm --name <container_name> -p 3000:80 <image_name>
Example:
```bash
 docker run -it --name ipro_fe_container -p 3000:80 demoacr4312.azurecr.io/ipro_fe:v1
```
This will create a container called `ipro_fe_container` from the image `demoacr4312.azurecr.io/ipro_fe:v1 `. Port number 80 is to access the service inside the container and Port number 3000 is to access the container from locally.

<p>&nbsp;</p>

### 5. To run using Kubernetes
Navigate to the root folder and run the below commands
1. create an yaml files and 
```bash
kubectl create -f <filename.yaml>
``` 
This will create the pods and deployements and required things that are present in yaml file

2. Run the command to get the external IP address to access the app
```bash
kubectl get service -o wide
```

<p>&nbsp;</p>

## To run test cases
1. Navigate to `tests/unit_tests` from the root folder in the terminal.
```bash
cd tests/unit_tests

pytest <filename.py>
```

<p>&nbsp;</p>



*For more information about Insights Pro, please refer the detailed KT videos and documentation below.* <br>
*1. [Insights Pro complete documentation](https://tigeranalytics-insights-pro.readthedocs-hosted.com/en/latest/index.html)* <br>
*2. [Demo setup](https://drive.google.com/file/d/1c5ox5POVQBkbqgsxxs7sfud_otb4aBgq/view)* <br>
*3. [Agnostic/core implementation](https://drive.google.com/file/d/1R_GtNBVowWowjLdfvNMZl61lbqBig5vw/view)* <br>
*4. [Core documentation](https://docs.google.com/document/d/1x7lpHmUl7i_ckSy3kIwXNPxK8u9_yFtm/edit#heading=h.gjdgxs)* <br>
*5. [API documentation](https://docs.google.com/document/d/1_6gABEvB2pTlzw79oDZgKSoskg_KwFnRvyB2Xzkuifs/edit#heading=h.z6ne0og04bp5)* <br>
