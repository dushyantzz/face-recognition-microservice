"""Configuration management for Face Recognition Service"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    app_name: str = Field(default="Face Recognition Service", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=True, env="DEBUG")
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    
    # Database
    database_url: str = Field(default="sqlite:///./frs_database.db", env="DATABASE_URL")
    
    # Model Paths
    detection_model_path: str = Field(default="models/detection/retinaface_resnet50.pth", env="DETECTION_MODEL_PATH")
    embedding_model_path: str = Field(default="models/embeddings/adaface_ir101_webface12m.ckpt", env="EMBEDDING_MODEL_PATH")
    onnx_detection_path: str = Field(default="models/onnx/retinaface.onnx", env="ONNX_DETECTION_PATH")
    onnx_embedding_path: str = Field(default="models/onnx/adaface.onnx", env="ONNX_EMBEDDING_PATH")
    
    # Model Settings
    use_onnx: bool = Field(default=True, env="USE_ONNX")
    detection_confidence_threshold: float = Field(default=0.8, env="DETECTION_CONFIDENCE_THRESHOLD")
    recognition_similarity_threshold: float = Field(default=0.6, env="RECOGNITION_SIMILARITY_THRESHOLD")
    min_face_size: int = Field(default=40, env="MIN_FACE_SIZE")
    top_k_matches: int = Field(default=5, env="TOP_K_MATCHES")
    
    # Processing Settings
    max_image_size: int = Field(default=1920, env="MAX_IMAGE_SIZE")
    face_size: int = Field(default=112, env="FACE_SIZE")
    num_workers: int = Field(default=4, env="NUM_WORKERS")
    batch_size: int = Field(default=8, env="BATCH_SIZE")
    
    # API Security
    api_key: Optional[str] = Field(default=None, env="API_KEY")
    enable_auth: bool = Field(default=False, env="ENABLE_AUTH")
    
    # Optimization
    use_cuda: bool = Field(default=False, env="USE_CUDA")
    onnx_optimization_level: int = Field(default=99, env="ONNX_OPTIMIZATION_LEVEL")
    faiss_use_gpu: bool = Field(default=False, env="FAISS_USE_GPU")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/frs.log", env="LOG_FILE")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()