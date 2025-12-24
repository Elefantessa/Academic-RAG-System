"""
Academic RAG System Configuration Module

This module provides centralized configuration management using Pydantic Settings.
Settings can be overridden via environment variables with RAG_ prefix.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class AppSettings(BaseSettings):
    """Application settings with environment variable support"""

    # ===== Data Paths =====
    json_file: str = "/project_antwerp/pdf_pipline/data/processed/all_extracted.json"
    persist_dir: str = "/project_antwerp/pdf_pipline/data/db/unified_chroma_db"
    collection_name: str = "academic_courses"

    # ===== Model Configuration =====
    embed_model: str = "Salesforce/SFR-Embedding-Mistral"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ollama_model: str = "llama3.1:latest"
    ollama_base_url: str = "http://localhost:11434"

    # ===== Agent Configuration =====
    top_k_rerank: int = 5
    default_k: int = 12
    lecturer_k: int = 40
    device: str = "cuda:4"

    # ===== Web Server Configuration =====
    host: str = "127.0.0.1"
    port: int = 5003
    debug: bool = False

    # ===== Advanced Settings =====
    max_context_length: int = 4000
    temperature: float = 0.0
    enable_caching: bool = False
    log_level: str = "INFO"

    class Config:
        """Pydantic configuration"""
        env_prefix = "RAG_"
        case_sensitive = False
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = AppSettings()
