import logging
import pandas as pd
import re
import gzip
import boto3
from io import BytesIO, StringIO
from  sqlalchemy import create_engine, text

class S3:
    """
    Utility functions to perform read, write, delete, mkdir, listdir operations to S3 Bucket
    """
    def __init__(self,bucket):
        self.s3_resource = boto3.resource("s3")
        self.s3_cli = boto3.client("s3")
        self.bucket_name = bucket

    def read(self,file_name,sep=',',sheet_name=None):
        df=pd.DataFrame()
        obj = self.s3_resource.Object(self.bucket_name,file_name)
        if file_name.endswith('txt.gz'):
            print(file_name)
            with gzip.GzipFile(fileobj=obj.get()["Body"]) as gzipfile:
                content = gzipfile.read()
            df = pd.read_csv(BytesIO(content), sep=sep)
        elif file_name.endswith('.csv'):
            fileobj=obj.get()["Body"]
            content = fileobj.read()
            df = pd.read_csv(BytesIO(content))
        elif file_name.endswith('.tsv'):
            fileobj=obj.get()["Body"]
            content = fileobj.read()
            df = pd.read_csv(BytesIO(content),sep='\t')

        elif file_name.endswith('.xlsx'):
            fileobj=obj.get()["Body"]
            content = fileobj.read()
            if sheet_name is not None:
                df = pd.read_excel(BytesIO(content), sheet_name=sheet_name)
            else:
                df = pd.read_excel(BytesIO(content))

        elif file_name.endswith('.parquet'):
            fileobj=obj.get()["Body"]
            content = fileobj.read()
            df = pd.read_parquet(BytesIO(content))


        return df

    def write(self,df,destination_path):
        buffer = StringIO()
        if destination_path.endswith('.csv'):
            df.to_csv(buffer, index=False)
        elif destination_path.endswith('.xlsx'):
            df.to_excel(buffer, index=False)
        elif destination_path.endswith('.tsv'):
            df.to_csv(buffer,sep='\t', index=False)
        print(self.s3_cli.put_object(Bucket=self.bucket_name, Key=destination_path, Body=buffer.getvalue()))

    def listdirs3(self,directory):
        p3 = boto3.resource('s3')
        bucket = p3.Bucket(self.bucket_name)
        objs = bucket.objects.filter(Prefix=directory)
        folders = []
        for obj in objs:
            folders.append(obj.key)
        return folders[1:]

    def mkdir(self,destination_path):
        directory_name = destination_path
        a = self.s3_cli.put_object(Bucket=self.bucket_name, Key=(directory_name+'/'))
        if a['ResponseMetadata']['HTTPStatusCode']==200:
            print("{} folder is created successfully".format(directory_name))
        else:
            print("Error in {} created httpStatusCode = {}".format(directory_name,a['ResponseMetadata']['HTTPStatusCode']))
            
    def remove_dir(self,path):
        directory_name = path
        bucket1 = self.s3_resource.Bucket(self.bucket_name)
        a = bucket1.objects.filter(Prefix=directory_name).delete()
        if a[0]['ResponseMetadata']['HTTPStatusCode']==200:
            print("{} folder is deleted successfully".format(directory_name))

def clean_column_names(df):
    """
    Cleans the columns names in the dataframe.

    Parameters
    ----------
    df : dataframe
        The input dataframe .

    Returns
    -------
    df : dataframe
        dataframe with updated columns
    """
    cols = df.columns
    new_cols = []
    for col in cols:
        col = col.lower()
        col = col.strip()
        col = col.replace("%","_percentage")
        col = re.sub('[^a-zA-Z0-9\%]', '_', col)
        col = re.sub('_+', '_', col)
        new_cols.append(col)
    df.columns = new_cols
    return df

def connect_to_mysql_db(config):
    """
    Connection to mysql database
    Parameters
    ----------
    config : dictionary
        contains information to connect to db
    
    Returns
    --------
    conn : sqlalchemy connection
    """
    hostname = config.load_tables.db_params.host
    user_name =  config.load_tables.db_params.username
    password = config.load_tables.db_params.password
    port = config.load_tables.db_params.port
    database = config.load_tables.db_params.db_schema
    engine = create_engine(f'mysql+pymysql://{user_name}:{password}@{hostname}:{port}/{database}') 
    conn = engine.connect()
    return conn

def load_table_to_db(data, config, db_table, playbook=False):
    """
    Load table to MySql database. Creates the table if is not present
    Parameters
    ----------
    data : dataframe
        data to be loaded to db
    config: dictionary
    db_table: string
        name of the table
    """
    # get the db connection
    conn = connect_to_mysql_db(config)
    round_id = config.load_tables.market.round_id
    database = config.load_tables.db_params.db_schema

    datatype_mapping = {"object":"VARCHAR(255)","int64":"BIGINT","float64":"FLOAT(20)","bool":"BOOL",'datetime64[ns]': 'DATETIME'}
    
    # get all tables in db
    query = text("SHOW TABLES")
    all_tables_in_db = pd.read_sql_query(query, con=conn)
    all_tables_in_db = list(all_tables_in_db['Tables_in_' + database])

    # create the table if table is not present in db
    if db_table not in all_tables_in_db:
        create_table_query = f"CREATE TABLE {db_table} ("
        for col in data.columns:
            create_table_query = f"{create_table_query}{col} {datatype_mapping[str(data[col].dtype)]},"
        create_table_query = create_table_query[:-1]
        if playbook:
            create_table_query + ' PARTITION BY HASH(id) PARTITIONS 10;'
        create_table_query = f"{create_table_query})"
        conn.execute(text(create_table_query))
    else:
        if playbook:
            delete_data_query = f"DELETE FROM {db_table} WHERE round_id = {round_id} AND store_id IN {tuple(data['store_id'].unique())}"
        conn.execute(text(delete_data_query))
           
    data.to_sql(db_table, con=conn, if_exists='append', index=False)
    if playbook and db_table not in all_tables_in_db:
        create_index_query = "CREATE INDEX idx_playbook ON "+db_table+" (round_id, id, item_id, item_name);"
        conn.execute(text(create_index_query))
    conn.commit()
    conn.close()