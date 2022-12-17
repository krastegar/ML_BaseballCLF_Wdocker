import pandas as pd
import sqlalchemy


class sqlPandasSpark:
    """
    To make this class compatable with the docker container
    I need to use the same password and db_host as mariadb image in
    my docker compose file
    """

    def __init__(
        self,
        db_user="root",
        db_pass="ROOT_ACCESS_PASSWORD",  # pragma: allowlist secret
        db_host="mariadb",
        db_database="baseball",
        *args,
        **kwargs,
    ):
        self.db_user = db_user
        self.db_pass = db_pass  # pragma: allowlist secret
        self.db_host = db_host
        self.db_database = db_database

    def sql_to_pandas(self):
        # Connecting python to mariadb using mariadb connector
        # Grabbing my Master table from SQL to do calculations on
        connect_string = f"mariadb+mariadbconnector://{self.db_user}:{self.db_pass}@{self.db_host}/{self.db_database}"

        sql_engine = sqlalchemy.create_engine(connect_string)

        query = """
            SELECT * FROM FeatureTable
        """
        df = pd.read_sql_query(query, sql_engine)

        # -- want to remove metadata feature such as team_id or game_id
        # unfortunately do not know how to do this without hard coding
        droppedFeatures = ["Home_Team", "Away_Team", "game_id"]
        df = df.drop(droppedFeatures, axis=1)

        return df
