"""Application configuration settings."""

from pydantic import BaseModel


class Settings(BaseModel):
    PROJECT_NAME: str = "Nexia"
    API_V1_PREFIX: str = "/api/v1"
    VERSION: str = "0.1.0"


settings = Settings()
