from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    postgres_host: str
    postgres_db: str
    postgres_user: str
    postgres_password: str
    redis_host: str
    redis_port: int
    kaggle_username: str
    kaggle_key: str
    mlflow_tracking_uri: str

    class Config:
        env_file = ".env"


settings = Settings()