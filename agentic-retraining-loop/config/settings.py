"""
settings.py  —  pydantic_settings replaced with plain os.getenv
No external dependencies needed.
"""
from __future__ import annotations
import os
from dotenv import load_dotenv
load_dotenv()


class Settings:
    database_url:        str   = os.getenv("DATABASE_URL",        "postgresql://postgres:password@localhost:5432/retraining")
    mlflow_tracking_uri: str   = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    groq_api_key:        str   = os.getenv("GROQ_API_KEY",        "gsk_your_groq_key_here")
    groq_model:          str   = os.getenv("GROQ_MODEL",          "llama-3.3-70b-versatile")
    slack_bot_token:     str   = os.getenv("SLACK_BOT_TOKEN",     "xoxb-your-slack-token")
    slack_channel_id:    str   = os.getenv("SLACK_CHANNEL_ID",    "C0XXXXXXXXX")
    drift_threshold:     float = float(os.getenv("DRIFT_THRESHOLD",    "0.2"))
    min_accuracy_delta:  float = float(os.getenv("MIN_ACCURACY_DELTA", "0.01"))
    model_registry_path: str   = os.getenv("MODEL_REGISTRY_PATH", "data/models")
    log_level:           str   = os.getenv("LOG_LEVEL",           "INFO")


settings = Settings()
