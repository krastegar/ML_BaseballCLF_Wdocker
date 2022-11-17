import sys

import pyspark as spark
from pyspark import SparkConf, StorageLevel
from pyspark.ml.feature import CountVectorizer
from pyspark.sql import SQLContext
from pyspark.sql.functions import array


def set_conf():
    # Creating a configuration function for my Spark object
    # Need spark.jars setting for connection to mariadb

    conf = SparkConf().setAppName("App")
    conf = conf.setMaster("local[*]").set(
        "spark.jars",
        "/home/bioinfo/Desktop/mysql-connector-java-5.1.46/mysql-connector-java-5.1.46.jar",
    )

    return conf


def read_data(table_name):
    """
    Purpose: Function is to grab data from a user specified table from baseball database

    """
    # Creating pyspark object
    sc = spark.SparkContext.getOrCreate(conf=set_conf())
    sqlContext = SQLContext(sc)

    # Making connection to mariadb
    mysql_db_driver_class = "com.mysql.jdbc.Driver"
    host_name = "localhost"
    port_no = "3306"
    user_name = "root"
    password = "root"  # pragma: allowlist secret
    database_name = "baseball"

    # Making JDBC URL
    mysql_jdbc_url = (
        "jdbc:mysql://"
        + host_name
        + ":"
        + port_no
        + "/"
        + database_name
        + "?zeroDateTimeBehavior=convertToNull"  # to turn 0 timestamp into null
    )

    sql_data = (
        sqlContext.read.format("jdbc")
        .option("url", mysql_jdbc_url)
        .option("driver", mysql_db_driver_class)
        .option("dbtable", table_name)
        .option("user", user_name)
        .option("password", password)
        .load()
    )
    return sql_data


def main():

    # Creating pyspark object
    sc = spark.SparkContext.getOrCreate(conf=set_conf())
    sqlContext = SQLContext(sc)

    # Reading DataTable from jdbc
    batter_counts_rdd = read_data("batter_counts")
    game_rdd = read_data("game")

    # Created Pyspark table for queries
    batter_counts_rdd.createOrReplaceTempView("batter_counts")
    batter_counts_rdd.persist(StorageLevel.DISK_ONLY)
    game_rdd.createOrReplaceTempView("game")
    game_rdd.persist(StorageLevel.DISK_ONLY)

    # Creating a temporary table
    tempTableDf = sqlContext.sql(
        """
    SELECT bc.game_id, bc.batter, atBat, Hit,
    team_id, local_date
    FROM batter_counts bc
    JOIN game g
    ON bc.game_id = g.game_id;
        """
    )

    tempTableDf.createOrReplaceTempView("temp")
    tempTableDf.persist(StorageLevel.DISK_ONLY)

    # Creating Rolling averag
    rolling_avg_df = sqlContext.sql(
        """
    SELECT t.batter, CAST(t.local_date AS VARCHAR(30)),
    (sum(roll_avg.Hit)/NULLIF(sum(roll_avg.atBat),0)) as rolling_avg
    FROM temp as t
    JOIN temp as roll_avg
    ON t.batter = roll_avg.batter
    AND t.local_date > roll_avg.local_date
    AND roll_avg.local_date between  t.local_date - INTERVAL 100 DAY and t.local_date
    GROUP BY t.batter, t.local_date
    ORDER BY t.local_date DESC
        """
    )

    # Putting Rolling_AVG into a transformer
    # Have to make a column to feed into transformer
    rolling_avg_df = rolling_avg_df.withColumn(
        "arrayColumn", array("batter", "local_date", "rolling_avg")
    )

    # Count Vectorizer
    count_vectorizer = CountVectorizer(
        inputCol="arrayColumn", outputCol="vectorized_values"
    )
    count_vectorizer_fitted = count_vectorizer.fit(rolling_avg_df)
    rolling_avg_df = count_vectorizer_fitted.transform(rolling_avg_df)
    rolling_avg_df.show()

    return


if __name__ == "__main__":
    sys.exit(main())
