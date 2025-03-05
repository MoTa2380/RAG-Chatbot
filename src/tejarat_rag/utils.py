import os
from dotenv import load_dotenv

class ConfigLoader:
    """Singleton class for loading and accessing environment variables."""

    _instance = None  # Singleton instance
    DEFAULT_ENV_FILE = ".env"

    def __new__(cls, env_file: str = DEFAULT_ENV_FILE):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._instance._load_env(env_file)
        return cls._instance

    def _load_env(self, env_file: str) -> None:
        """Load environment variables from a file."""
        load_dotenv(env_file)

    @staticmethod
    def get(key: str, default: str = None) -> str:
        """Get an environment variable with an optional default value."""
        return os.getenv(key, default)


