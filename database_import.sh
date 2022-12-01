#! /bin/bash

sleep 30

DATABASE_TO_COPY_INTO="baseball"
DATABASE_FILE="baseball.sql"
PASS="ROOT_ACCESS_PASSWORD"

mysql -u root -p${PASS} -h mariadb -e "CREATE DATABASE IF NOT EXISTS ${DATABASE_TO_COPY_INTO}"
mysql -u root -p${PASS} -h mariadb ${DATABASE_TO_COPY_INTO} < ${DATABASE_FILE}
echo "DATABASE IMPORT DONE....."
echo "Starting HW......."

mysql -u root -p${PASS} -h mariadb baseball < src/SQL_BAIN_OF_MY_EXISTENCE.sql > results.csv

echo "Done...."