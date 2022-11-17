import sys

import pandas as pd
import sqlalchemy


class sqlPandasSpark:
    def __init__(
        self,
        db_user="root",
        db_pass="root",  # pragma: allowlist secret
        db_host="localhost",
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
            SELECT * FROM Master
        """
        df = pd.read_sql_query(query, sql_engine)
        _ = "dummy"
        return df


def main():

    sql_pandas_df = sqlPandasSpark().sql_to_pandas()
    print(sql_pandas_df)
    return


if __name__ == "__main__":
    sys.exit(main())
