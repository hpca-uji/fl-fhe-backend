# Use an official Python runtime as a base image
FROM python:3.10-slim
# Set the working directory in the container
WORKDIR /app
# Install system dependencies using apt
RUN apt-get update && apt-get install -y \
  libomp-dev \
  build-essential \
  libssl-dev \
  libffi-dev \
  gcc g++ \
  cmake \
  git \
  python3-dev \
  && rm -rf /var/lib/apt/lists/*
# Copy the current directory contents into the container at /app
COPY ../../. /app
# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --timeout=1800 -r requirements.txt
# Make port 9094 available for clients connection
EXPOSE 9094
# Define environment variable for Python
ENV PYTHONUNBUFFERED 1
# Run the application
ENTRYPOINT ["flower-supernode"]
CMD ["--insecure", "--superlink", "federated-server:9092", "--node-config", "partition-id=1 num-partitions=3"]

