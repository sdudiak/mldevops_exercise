# Base image with Python 3.12
FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye

# Set up Python environment
COPY ./requirements.txt /requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /requirements.txt && \
    python3 -m pipx ensurepath && \
    pipx install poetry

# Set up additional environment variables and devcontainer configurations
ENV PYTHONUNBUFFERED=1
