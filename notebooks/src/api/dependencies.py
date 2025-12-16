"""
Dependencies for FastAPI application
"""

from functools import lru_cache
from typing import Dict, Any
import os

class APIConfig:
    """API Configuration"""
    def __init__(self):
        self.model_path = os.getenv("MODEL_PATH", "models/task5/best_model.pkl")
        self.scaler_path = os.getenv("SCALER_PATH", "models/task5/scaler.pkl")
        self.features_path = os.getenv("FEATURES_PATH", "data/processed/model_features_fixed.txt")
        self.metadata_path = os.getenv("METADATA_PATH", "reports/metrics_summary.json")
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "8000"))
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.api_key = os.getenv("API_KEY", "")
        self.allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")

@lru_cache()
def get_config():
    """Get cached configuration"""
    return APIConfig()