"""Main FastAPI application for Face Recognition Service"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys
from pathlib import Path

from .config import settings
from .api.routes import router
from .api import routes as api_routes
from .detection import RetinaFaceDetector
from .embeddings import AdaFaceExtractor
from .matching import FaissIndexMatcher
from .database.database import engine
from .database import models as db_models

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(settings.log_file)
    ]
)
logger = logging.getLogger(__name__)

# Create database tables
db_models.Base.metadata.create_all(bind=engine)

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production-ready Face Recognition Service with RetinaFace detection, AdaFace embeddings, and Faiss search",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1", tags=["Face Recognition"])


@app.on_event("startup")
async def startup_event():
    """Initialize models and services on startup"""
    logger.info("Starting Face Recognition Service...")
    
    try:
        # Initialize detector
        logger.info("Loading face detector...")
        detector_path = settings.onnx_detection_path if settings.use_onnx else settings.detection_model_path
        api_routes.detector = RetinaFaceDetector(
            model_path=detector_path,
            use_onnx=settings.use_onnx,
            confidence_threshold=settings.detection_confidence_threshold,
            min_face_size=settings.min_face_size
        )
        logger.info("Face detector loaded successfully")
        
        # Initialize embedding extractor
        logger.info("Loading embedding extractor...")
        embedding_path = settings.onnx_embedding_path if settings.use_onnx else settings.embedding_model_path
        api_routes.embedding_extractor = AdaFaceExtractor(
            model_path=embedding_path,
            use_onnx=settings.use_onnx
        )
        logger.info("Embedding extractor loaded successfully")
        
        # Initialize matcher
        logger.info("Initializing face matcher...")
        api_routes.matcher = FaissIndexMatcher(
            embedding_dim=512,
            similarity_threshold=settings.recognition_similarity_threshold,
            top_k=settings.top_k_matches,
            index_type='flat',
            use_gpu=settings.faiss_use_gpu
        )
        logger.info("Face matcher initialized successfully")
        
        # Load existing embeddings from database if available
        from .database.database import SessionLocal
        db = SessionLocal()
        try:
            embeddings_data = db.query(db_models.FaceEmbedding).all()
            if embeddings_data:
                logger.info(f"Loading {len(embeddings_data)} embeddings from database...")
                for emb_data in embeddings_data:
                    import numpy as np
                    embedding = np.frombuffer(emb_data.embedding, dtype=np.float32)
                    api_routes.matcher.add_identity(
                        embedding=embedding,
                        identity_id=emb_data.identity_id,
                        identity_name=emb_data.identity.name,
                        metadata=emb_data.identity.metadata
                    )
                logger.info(f"Loaded {len(embeddings_data)} embeddings into matcher")
        finally:
            db.close()
        
        logger.info(f"{settings.app_name} v{settings.app_version} started successfully")
        logger.info(f"Running on {settings.host}:{settings.port}")
        logger.info(f"API docs available at http://{settings.host}:{settings.port}/docs")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Face Recognition Service...")
    
    # Save matcher state if needed
    try:
        if api_routes.matcher:
            index_path = "data/faiss_index"
            api_routes.matcher.save(index_path)
            logger.info(f"Saved matcher state to {index_path}")
    except Exception as e:
        logger.error(f"Error saving matcher state: {e}")
    
    logger.info("Shutdown complete")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )