FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies first (to leverage Docker cache)
COPY requirements.txt .

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the entire application folder, including subdirectories
COPY app/ ./*.parquet .

RUN chmod u+x ./*.sh

# Set the default command to run the script and clean up __pycache__
CMD ["bash", "-c", "./run.sh . && ./cleanup.sh"]
