from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration management using Pydantic Settings."""

    livekit_url: str = Field(
        default="wss://your-project.livekit.cloud",
        description="LiveKit server WebSocket URL",
    )
    livekit_api_key: str = Field(
        default="devkey",
        description="LiveKit API key",
    )
    livekit_api_secret: str = Field(
        default="secret",
        description="LiveKit API secret",
    )
    deepgram_api_key: str = Field(
        default="your_deepgram_key",
        description="Deepgram API key for STT",
    )
    groq_api_key: str = Field(
        default="your_groq_key",
        description="Groq API key for LLM inference",
    )
    azure_speech_key: str = Field(
        default="your_azure_speech_key",
        description="Azure Speech key for TTS",
    )
    azure_speech_region: str = Field(
        default="centralindia",
        description="Azure Speech region",
    )
    log_level: str = Field(
        default="INFO",
        description="Application logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    """Retrieve cached application settings instance."""
    return Settings()


# Singleton instance for direct import convenience
settings = get_settings()
