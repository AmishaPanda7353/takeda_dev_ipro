import logging
import psycopg2
import psycopg2.extensions
from src.query_insights.utils.utils import DataConverter
from src.query_insights.facilitator import DataLoader
from core.utils.read_config import initialize_config, app_database_config
from sentence_transformers import SentenceTransformer
from psycopg2.extras import execute_batch, Json
import math

from core.vectordb.pgvectordb_client import PGVectorDB

MYLOGGERNAME = "QueryInsights"


class PGVector(PGVectorDB):
    def __init__(self, config, data_frame, module_name = None):
        super().__init__(config, data_frame)
        self.host = app_database_config.host
        self.database = app_database_config.database_name
        self.user = app_database_config.username
        self.password = app_database_config.password
        self.port = app_database_config.port
        self.model_name = config.embedding_model
        self.embeddings_dim = config.embedding_dim
        self.module_name = module_name

        if self.module_name: #Specifications for question classification 
            self.vector_db_table_name = "qc_vector_table"
            self.target_column = "category"
            self.top_k = 1
        else:           #Specification for table selection
            self.vector_db_table_name = self.vector_db_table_name
            self.target_column = "unique_id"
            self.top_k = self.config.top_k

        if self.model_name == "sentence_transformer":
            self.pre_built_model = self.config.pre_built_model
            self.model = SentenceTransformer(self.pre_built_model)
        self.df = data_frame
        self.conn = None
        self.connect()
        self.create_table()

    def connect(self):
        self.conn = psycopg2.connect(
            host=self.host,
            database=self.database,
            user=self.user,
            password=self.password,
            port=self.port
        )
        psycopg2.extensions.register_adapter(list, self.adapt_vector)

    def adapt_vector(self, embedding):
        return Json(embedding.tolist())

    def enable_vector_extension(self):
        query = "CREATE EXTENSION vector;"
        cur = self.conn.cursor()
        try:
            cur.execute(query)
            print("Vector Extension Enabled")
        except:
            self.conn.commit()
        cur.close()
        self.conn.commit()

    def create_table(self):
        self.enable_vector_extension()

        if not self.module_name: # this works for the table selection 
            table_create_command = f"""
            CREATE TABLE {self.vector_db_table_name} (
                unique_id bigserial primary key, 
                column_name text,
                column_description text,
                table_name text,
                id text,
                combined text,
                embedding vector({self.embeddings_dim})
            );
            """
        else:
            table_create_command = f"""
            CREATE TABLE {self.vector_db_table_name} (
            question_id int primary key,
            question_group_id int,
            question text,
            original text,
            category text,
            embedding vector({self.embeddings_dim})
            );
            """

        # with self.conn.cursor() as cur:
        cur = self.conn.cursor()
        try:
            cur.execute(table_create_command)
            print("Table created successfully")
        except Exception as e:
            self.conn.commit()
            print(f"Error while executing the query. {e}")
            print("Table Already Exists")
            cur.execute(f"DELETE FROM {self.vector_db_table_name}")
        cur.close()
        self.conn.commit()

    def convert_to_embeddings(self, text):
        return self.model.encode([text])[0]

    def insert_data(self):
        # Convert column_description to embeddings
        if not self.module_name:
            self.df['embeddings'] = self.df['column_description'].apply(self.convert_to_embeddings)

            self.df['embeddings'] = self.df['embeddings'].apply(self.adapt_vector)

            psycopg2.extensions.register_adapter(list, self.adapt_vector)

            with self.conn.cursor() as cur:
                insert_query = f"""
                    INSERT INTO {self.vector_db_table_name} (unique_id, column_name, column_description, table_name, id, combined, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """

                # Convert DataFrame rows to tuples
                data_tuples = [tuple(row) for row in self.df.itertuples(index=False, name=None)]

                # Execute the batch insert
                execute_batch(cur, insert_query, data_tuples)

                # Commit the transaction
                self.conn.commit()
                print("Data Inserted Successfully")
        else:
            self.df['embeddings'] = self.df['question'].apply(self.convert_to_embeddings)
            self.df['embeddings'] = self.df['embeddings'].apply(self.adapt_vector)

            psycopg2.extensions.register_adapter(list, self.adapt_vector)

            with self.conn.cursor() as cur:
                insert_query = f"""
                    INSERT INTO {self.vector_db_table_name} (question_id,question_group_id,question,original,category,embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """

                # Convert DataFrame rows to tuples
                data_tuples = [tuple(row) for row in self.df.itertuples(index=False, name=None)]

                # Execute the batch insert
                execute_batch(cur, insert_query, data_tuples)

                # Commit the transaction
                self.conn.commit()
                print("Data Inserted Successfully")

        self.create_index()

    def create_index(self):
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) as cnt FROM {self.vector_db_table_name};")
            num_records = cur.fetchone()[0]

            num_lists = num_records / 1000
            if num_lists < 10:
                num_lists = 10
            if num_records > 1000000:
                num_lists = math.sqrt(num_records)

            cur.execute(f'CREATE INDEX IF NOT EXISTS embedding_ivfflat_idx ON {self.vector_db_table_name} USING ivfflat (embedding vector_cosine_ops) WITH (lists = {num_lists});')
            self.conn.commit()

    def search(self, query):
        query_text_embedding = self.convert_to_embeddings(query)
        query_embedding = self.adapt_vector(query_text_embedding)
        
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT {self.target_column} FROM {self.vector_db_table_name} ORDER BY embedding <=> %s LIMIT {self.top_k}", (query_embedding,))
            query_results = cur.fetchall()
            category = query_results[0][0] #[x[0] for x in query_results]
            self.conn.commit()
        
        return category

    def close_connection(self):
        if self.conn is not None:
            self.conn.close()


# Example usage:

# Assuming `df` is your DataFrame containing 'column_description' to be converted to embeddings

# Instantiate the EmbeddingsDatabase class
# embeddings_db = PGVector(host, database, user, password, port, model)

# # Connect to the database
# embeddings_db.connect()

# # Create the table if it doesn't exist
# embeddings_db.create_table()

# # Insert embeddings into the database from your DataFrame df
# embeddings_db.insert_embeddings()

# # Create index for efficient searching
# embeddings_db.create_index()

# # Perform a search query
# query = "your query text"
# results = embeddings_db.search(query)
# print("Search results:", results)

# # Close the database connection
# embeddings_db.close_connection()
