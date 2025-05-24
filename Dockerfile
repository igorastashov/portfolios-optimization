# Base Python image
FROM python:3.11-slim

WORKDIR /app

# Install Poetry
RUN pip install --upgrade pip
RUN pip install poetry==1.8.2

# Copy dependency definition files
COPY pyproject.toml poetry.lock* /app/

# Install project dependencies using Poetry
# --no-root: do not install the project itself as a package, only dependencies
# --no-dev: do not install development dependencies (standard for production images)
# virtualenvs.create false: install dependencies in the system Python environment within the container
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi --no-root --no-dev

# Copy the rest of the application code
COPY . /app

# Copy entrypoint and make it executable
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Environment variables (can be overridden in docker-compose.yml)
ENV PYTHONPATH=/app
# Default APP_MODULE, HOST, PORT are for Uvicorn; Celery worker will override the command.
ENV APP_MODULE="backend.app.main:app"
ENV HOST="0.0.0.0"
ENV PORT="8000"

ENTRYPOINT ["/app/entrypoint.sh"]
# Default command to be passed to entrypoint.sh for the backend service.
# The worker service in docker-compose.yml specifies its own command.
CMD ["uvicorn", "${APP_MODULE}", "--host", "${HOST}", "--port", "${PORT}"] 