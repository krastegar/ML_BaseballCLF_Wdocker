import sys

import pyspark as spark
from pyspark import SparkConf
from pyspark.sql import SQLContext


def set_conf():
    # Creating a configuration function for my Spark object
    # Need spark.jars setting for connection to mariadb

    conf = SparkConf().setAppName("App")
    conf = conf.setMaster("local[*]").set(
        "spark.jars",
        "/home/bioinfo/Desktop/mysql-connector-java-5.1.46/mysql-connector-java-5.1.46.jar",
    )

    return conf


def main():

    # Making Spark Object
    sc = spark.SparkContext.getOrCreate(conf=set_conf())
    sqlContext = SQLContext(sc)

    """
    Different method to connect sql database

    spark = SparkSession \
        .builder \
        .appName("App") \
        .master('local[*]') \
        .config('spark.jars', '/home/bioinfo/Desktop/mysql-connector-java-5.1.46/mysql-connector-java-5.1.46.jar') \
        .enableHiveSupport() \
        .getOrCreate() \
    """

    # Making connection to MariaDB
    mysql_db_driver_class = "com.mysql.jdbc.Driver"
    host_name = "localhost"
    port_no = "3306"
    user_name = "root"
    password = "Hamid&Mahasty1"  # pragma: allowlist secret
    database_name = "baseball"

    # test_query = """
    # SELECT * FROM batter_counts
    # """

    # Making JDBC URL
    mysql_jdbc_url = "jdbc:mysql://" + host_name + ":" + port_no + "/" + database_name

    # Reading DataTable from jdbc
    jdbcDF = (
        sqlContext.read.format("jdbc")
        .option("url", mysql_jdbc_url)
        .option("driver", mysql_db_driver_class)
        .option("dbtable", "batter_counts")
        .option("user", user_name)
        .option("password", password)
        .load()
    )

    jdbcDF.show()
    return


if __name__ == "__main__":
    sys.exit(main())
