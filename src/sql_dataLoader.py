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
        
        # making my response column : home_team_id, team_id, winner_home_or_away
        win_col = []
        for team_id, home_id, win in zip(df['team_id'],df['home_team_id'],df['winner_home_or_away']):
            if team_id == home_id and win =='H':
                i=1; win_col.append(i)
            else:
                i=0; win_col.append(i)
        df['HomeWins'] = win_col

        return df