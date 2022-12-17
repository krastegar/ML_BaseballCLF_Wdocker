#! /bin/bash

sleep 30

DATABASE_TO_COPY_INTO="baseball"
DATABASE_FILE="baseball.sql"
PASS="ROOT_ACCESS_PASSWORD"

mysql -u root -p${PASS} -h mariadb -e "CREATE DATABASE IF NOT EXISTS ${DATABASE_TO_COPY_INTO}"
mysql -u root -p${PASS} -h mariadb ${DATABASE_TO_COPY_INTO} < ${DATABASE_FILE}
echo "DATABASE IMPORT DONE....."
echo "Starting HW......."

echo "Making my feature tables in sql (ETA: 4min)"
mysql -u root -p${PASS} -h mariadb baseball < src/HW_5_SQL.sql

echo "RUNNING PYTHON SCRIPT FOR FEATURE ENGINEERING AND MODEL EVALUATION"
python3 src/HW_5.py


echo "Done...."