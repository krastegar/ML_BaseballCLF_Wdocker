FROM python:3.9.7
 
ENV APP_HOME /app
WORKDIR $APP_HOME
ENV PYTHONPATH /
 
# Get necessary system packages
RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
     build-essential \
     python3 \
     python3-pip \
     python3-dev \
     mariadb-client \
  && rm -rf /var/lib/apt/lists/*
 
# Get necessary python libraries
COPY requirements.txt .
RUN pip3 install --compile --no-cache-dir -r requirements.txt
 
# Copy over code
COPY src/SQL_BAIN_OF_MY_EXISTENCE.sql src/SQL_BAIN_OF_MY_EXISTENCE.sql
COPY database_import.sh .

# Run database import
# Need to update database_import.sh to run my SQL_BAIN script 
CMD /app/database_import.sh 
