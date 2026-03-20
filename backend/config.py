from decouple import config


class Settings:
    """Application settings loaded from environment variables."""

    # Database
    DATABASE_URL: str = config(
        "DATABASE_URL", default="sqlite:///./auth.db"
    )

    # JWT Settings
    SECRET_KEY: str = config(
        "SECRET_KEY",
        default="your-super-secret-key-change-this-in-production",
    )
    ALGORITHM: str = config("ALGORITHM", default="HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = config(
        "ACCESS_TOKEN_EXPIRE_MINUTES", default=30, cast=int
    )
    REFRESH_TOKEN_EXPIRE_DAYS: int = config(
        "REFRESH_TOKEN_EXPIRE_DAYS", default=7, cast=int
    )

    # Application Settings
    APP_NAME: str = config("APP_NAME", default="AuthKeyHub")
    DEBUG: bool = config("DEBUG", default=False, cast=bool)


settings = Settings()
