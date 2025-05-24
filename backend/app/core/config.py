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
    REFRESH_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7
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
        if isinstance(v, str):
            return v
        user = values.get("POSTGRES_USER")
        password = values.get("POSTGRES_PASSWORD")
        host = values.get("POSTGRES_SERVER")
        port = values.get("POSTGRES_PORT")
        db_name = values.get("POSTGRES_DB")
        if all([user, password, host, db_name]):
            return PostgresDsn.build(
                scheme="postgresql",
                username=user,
                password=password,
                host=host,
                port=port,
                path=f"/{db_name}",
            )
        return v

    REDIS_HOST: Optional[str] = "redis"
    REDIS_PORT: Optional[str] = "6379"
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    
    BACKEND_CORS_ORIGINS: Union[str, List[AnyHttpUrl]]

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[AnyHttpUrl], str]:
        if isinstance(v, str) and not v.startswith("["):
            if v == "*":
                return "*"
            raise ValueError(f'''BACKEND_CORS_ORIGINS environment variable must be a JSON string array of URLs (e.g., '["http://localhost:3000"]') or "*" to allow all origins.''')
        elif isinstance(v, str) and v.startswith("["):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON string for BACKEND_CORS_ORIGINS")
        elif isinstance(v, list):
            return v
        raise ValueError("BACKEND_CORS_ORIGINS must be a list of URLs, a JSON string array, or \"*\"")

    FIRST_SUPERUSER_EMAIL: Optional[str] = None
    FIRST_SUPERUSER_USERNAME: Optional[str] = None
    FIRST_SUPERUSER_PASSWORD: Optional[str] = None

    CLEARML_API_ACCESS_KEY: Optional[str] = None
    CLEARML_API_SECRET_KEY: Optional[str] = None
    CLEARML_API_HOST: Optional[str] = None

    LOGGING_LEVEL: str = "INFO"

    CATBOOST_INFERENCE_SERVICE_URL: Optional[AnyHttpUrl] = None
    DRL_INFERENCE_SERVICE_URL: Optional[AnyHttpUrl] = None

    model_config = SettingsConfigDict(env_file=".env", extra='ignore')

settings = Settings() 