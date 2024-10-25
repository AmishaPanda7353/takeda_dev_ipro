from databse_client import DatabaseClient


class DatabaseService:
    def __init__(self, database_service: DatabaseClient) -> None:
        self.database_service = database_service

    def send_notification(self, message, receiver):
        self.database_service.create_database_connection(config=None)
