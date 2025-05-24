from pydantic import validator, PostgresDsn, AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional, Dict, Any, Union
import json
import os

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Portfolio Optimization API"

    # JWT Settings
    JWT_SECRET_KEY: str
    JWT_REFRESH_SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7 # 7 days
    ALGORITHM: str = "HS256"

    # Database settings
    POSTGRES_SERVER: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_PORT: Optional[str] = "5432"
    SQLALCHEMY_DATABASE_URI: Optional[PostgresDsn] = None

    @validator("SQLALCHEMY_DATABASE_URI", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if isinstance(v, str): # Allows direct URI override from .env
            return v
        # Construct URI from components if not directly provided
        user = values.get("POSTGRES_USER")
        password = values.get("POSTGRES_PASSWORD")
        host = values.get("POSTGRES_SERVER")
        port = values.get("POSTGRES_PORT")
        db_name = values.get("POSTGRES_DB")
        if all([user, password, host, db_name]): # Ensure all parts are present
            return PostgresDsn.build(
                scheme="postgresql",
                username=user,
                password=password,
                host=host,
                port=port,
                path=f"/{db_name}",
            )
        return v # Return original value (None or existing URI) if components are missing

    # Redis settings (general, not specific to Celery backend if Celery uses RabbitMQ)
    REDIS_HOST: Optional[str] = "redis" # Docker Compose service name
    REDIS_PORT: Optional[str] = "6379"
    # CELERY_BROKER_URL and CELERY_RESULT_BACKEND will be loaded from environment variables
    # set in docker-compose.yml, which determine if Redis or RabbitMQ is used.
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    
    # CORS Origins
    BACKEND_CORS_ORIGINS: Union[str, List[AnyHttpUrl]]

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[AnyHttpUrl], str]:
        if isinstance(v, str) and not v.startswith("["):
            if v == "*": # Allow all origins
                return "*"
            # If it's a single string, not a JSON array, and not "*", it's an error.
            # It must be either "*" or a valid JSON string array of URLs.
            raise ValueError(f'''BACKEND_CORS_ORIGINS environment variable must be a JSON string array of URLs (e.g., '["http://localhost:3000"]') or "*" to allow all origins.''')
        elif isinstance(v, str) and v.startswith("["): # JSON string array
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON string for BACKEND_CORS_ORIGINS")
        elif isinstance(v, list): # Already a list (e.g., from direct instantiation in tests)
            return v
        raise ValueError("BACKEND_CORS_ORIGINS must be a list of URLs, a JSON string array, or \"*\"")

    # First Superuser (optional, for init_db.py)
    FIRST_SUPERUSER_EMAIL: Optional[str] = None
    FIRST_SUPERUSER_USERNAME: Optional[str] = None
    FIRST_SUPERUSER_PASSWORD: Optional[str] = None

    # ClearML Configuration
    CLEARML_API_ACCESS_KEY: Optional[str] = None
    CLEARML_API_SECRET_KEY: Optional[str] = None
    CLEARML_API_HOST: Optional[str] = None

    # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    LOGGING_LEVEL: str = "INFO"

    # Inference Service URLs
    CATBOOST_INFERENCE_SERVICE_URL: Optional[AnyHttpUrl] = None
    DRL_INFERENCE_SERVICE_URL: Optional[AnyHttpUrl] = None

    model_config = SettingsConfigDict(env_file=".env", extra='ignore')

    # __init__ is generally not needed if using validators correctly and Pydantic handles .env loading.
    # SQLALCHEMY_DATABASE_URI is handled by its validator.
    # CELERY_BROKER_URL and CELERY_RESULT_BACKEND are loaded directly from .env.
    # def __init__(self, **values: Any):
    #     super().__init__(**values)
        # The Pydantic model will automatically load from .env and environment variables.
        # Validators will run to assemble complex fields if needed.

settings = Settings() 