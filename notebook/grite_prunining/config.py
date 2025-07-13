

from pydantic import BaseModel, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
import pandas as pd

class Settings(BaseSettings):
    size_path: float = 5.0  # Valeur par défaut ajoutée
    
    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
