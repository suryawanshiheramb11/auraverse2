# Use an official Python runtime as a parent image
# Slim version for smaller size
FROM python:3.9-slim

# Set the working directory to /app
WORKDIR /app

# Install system dependencies
# Install system dependencies
# libglib2.0-0 is still often needed for some core dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Make port 7860 available to the world outside this container
EXPOSE 7860

# Define environment variable
ENV PORT=7860

# Run server.py when the container launches
# Using host 0.0.0.0 is crucial for Docker
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
