FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye

COPY ./requirements.txt /requirements.txt
# Install any additional dependencies here
RUN pip install --user -r /requirements.txt && \
    python3 -m pipx ensurepath && \
    pipx install poetry
