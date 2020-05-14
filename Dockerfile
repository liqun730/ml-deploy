FROM tensorflow/tensorflow:2.0.0-py3

# Maintainer info
LABEL maintainer="yuliqun88@gmail.com"

# Make working directories
RUN  mkdir -p /home/project/ml-classifiler
WORKDIR /home/project/ml-classifiler

# Upgrade pip with no cache
RUN pip install --no-cache-dir -U pip

# Copy application requirements file to the created working directory
COPY requirements.txt .

# Install application dependencies from the requirements file
RUN pip install -r requirements.txt

# Copy every file in the source folder to the created working directory
COPY  . .

EXPOSE 8080

ENTRYPOINT uvicorn app.main:app --host 0.0.0.0 --port 8080
