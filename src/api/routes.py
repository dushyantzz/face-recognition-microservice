"""API routes for Face Recognition Service"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from fastapi.responses import JSONResponse
import base64
import numpy as np
import cv2
from typing import List
import time
import logging

from ..detection import RetinaFaceDetector
from ..embeddings import AdaFaceExtractor
from ..matching import FaissIndexMatcher
from ..database.database import get_db
from ..database import models as db_models
from .schemas import (
    DetectRequest, DetectResponse, Detection, BoundingBox, Landmarks,
    RecognizeRequest, RecognizeResponse, RecognitionResult, Match,
    AddIdentityRequest, AddIdentityResponse,
    ListIdentitiesResponse, Identity,
    DeleteIdentityRequest, DeleteIdentityResponse,
    HealthResponse, ErrorResponse
)
from sqlalchemy.orm import Session
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter()

# Global model instances (will be initialized at startup)
detector: RetinaFaceDetector = None
embedding_extractor: AdaFaceExtractor = None
matcher: FaissIndexMatcher = None
app_start_time = time.time()


def decode_image(image_str: str) -> np.ndarray:
    """Decode base64 image string to numpy array"""
    try:
        # Handle data URL format
        if image_str.startswith('data:image'):
            image_str = image_str.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_str)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        return image
    except Exception as e:
        logger.error(f"Error decoding image: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")


@router.post("/detect", response_model=DetectResponse)
async def detect_faces(request: DetectRequest):
    """Detect faces in an image"""
    start_time = time.time()
    
    try:
        # Decode image
        image = decode_image(request.image)
        
        # Update detector settings if provided
        if request.confidence_threshold:
            detector.confidence_threshold = request.confidence_threshold
        if request.min_face_size:
            detector.min_face_size = request.min_face_size
        
        # Detect faces
        detections = detector.detect(image)
        
        # Format response
        detection_results = []
        for det in detections:
            detection_results.append(Detection(
                bbox=BoundingBox(
                    x1=det.bbox[0],
                    y1=det.bbox[1],
                    x2=det.bbox[2],
                    y2=det.bbox[3]
                ),
                confidence=det.confidence,
                landmarks=Landmarks(points=det.landmarks.tolist()),
                quality_score=det.quality_score
            ))
        
        processing_time = (time.time() - start_time) * 1000
        
        return DetectResponse(
            detections=detection_results,
            num_faces=len(detections),
            processing_time_ms=processing_time
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recognize", response_model=RecognizeResponse)
async def recognize_faces(request: RecognizeRequest, db: Session = Depends(get_db)):
    """Recognize faces in an image"""
    start_time = time.time()
    
    try:
        # Decode image
        image = decode_image(request.image)
        
        # Update detector settings if provided
        if request.min_face_size:
            detector.min_face_size = request.min_face_size
        
        # Detect and align faces
        aligned_faces = detector.detect_and_align(image)
        
        if not aligned_faces:
            return RecognizeResponse(
                results=[],
                num_faces=0,
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        # Extract embeddings
        faces = [face for face, _ in aligned_faces]
        embeddings = embedding_extractor.extract_batch_embeddings(faces)
        
        # Match against gallery
        results = []
        for (aligned_face, detection), embedding in zip(aligned_faces, embeddings):
            if embedding is None:
                continue
            
            # Search for matches
            matches = matcher.search(
                embedding,
                top_k=request.top_k or 5
            )
            
            # Format matches
            match_results = [
                Match(
                    identity_id=m.identity_id,
                    identity_name=m.identity_name,
                    similarity=m.similarity,
                    rank=m.rank,
                    metadata=m.metadata
                )
                for m in matches
            ]
            
            results.append(RecognitionResult(
                bbox=BoundingBox(
                    x1=detection.bbox[0],
                    y1=detection.bbox[1],
                    x2=detection.bbox[2],
                    y2=detection.bbox[3]
                ),
                detection_confidence=detection.confidence,
                matches=match_results,
                quality_score=detection.quality_score
            ))
        
        processing_time = (time.time() - start_time) * 1000
        
        return RecognizeResponse(
            results=results,
            num_faces=len(results),
            processing_time_ms=processing_time
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in recognize endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/add_identity", response_model=AddIdentityResponse)
async def add_identity(request: AddIdentityRequest, db: Session = Depends(get_db)):
    """Add a new identity to the gallery"""
    start_time = time.time()
    
    try:
        # Create identity in database
        db_identity = db_models.Identity(
            name=request.name,
            metadata=request.metadata or {}
        )
        db.add(db_identity)
        db.commit()
        db.refresh(db_identity)
        
        # Process images and extract embeddings
        embeddings_added = 0
        for image_str in request.images:
            try:
                # Decode image
                image = decode_image(image_str)
                
                # Detect and align faces
                aligned_faces = detector.detect_and_align(image)
                
                if not aligned_faces:
                    logger.warning(f"No face detected in image for identity {request.name}")
                    continue
                
                # Use the face with highest quality
                aligned_faces.sort(key=lambda x: x[1].quality_score, reverse=True)
                best_face, best_detection = aligned_faces[0]
                
                # Extract embedding
                embedding = embedding_extractor.extract_embedding(best_face)
                
                if embedding is None:
                    logger.warning(f"Failed to extract embedding for identity {request.name}")
                    continue
                
                # Save to database
                db_embedding = db_models.FaceEmbedding(
                    identity_id=db_identity.id,
                    embedding=embedding.tobytes(),
                    image_path=f"identity_{db_identity.id}_img_{embeddings_added}.jpg",
                    quality_score=best_detection.quality_score
                )
                db.add(db_embedding)
                
                # Add to matcher index
                matcher.add_identity(
                    embedding=embedding,
                    identity_id=db_identity.id,
                    identity_name=db_identity.name,
                    metadata=request.metadata
                )
                
                embeddings_added += 1
                
            except Exception as e:
                logger.error(f"Error processing image for identity {request.name}: {e}")
                continue
        
        db.commit()
        
        if embeddings_added == 0:
            db.delete(db_identity)
            db.commit()
            raise HTTPException(
                status_code=400,
                detail="No valid face embeddings could be extracted from provided images"
            )
        
        processing_time = (time.time() - start_time) * 1000
        
        return AddIdentityResponse(
            identity_id=db_identity.id,
            identity_name=db_identity.name,
            num_embeddings_added=embeddings_added,
            processing_time_ms=processing_time
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in add_identity endpoint: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list_identities", response_model=ListIdentitiesResponse)
async def list_identities(db: Session = Depends(get_db)):
    """List all identities in the gallery"""
    try:
        identities = db.query(db_models.Identity).all()
        
        identity_list = []
        for identity in identities:
            num_images = db.query(db_models.FaceEmbedding).filter(
                db_models.FaceEmbedding.identity_id == identity.id
            ).count()
            
            identity_list.append(Identity(
                identity_id=identity.id,
                identity_name=identity.name,
                num_images=num_images,
                created_at=identity.created_at,
                updated_at=identity.updated_at,
                metadata=identity.metadata
            ))
        
        return ListIdentitiesResponse(
            identities=identity_list,
            total_count=len(identity_list)
        )
    
    except Exception as e:
        logger.error(f"Error in list_identities endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete_identity", response_model=DeleteIdentityResponse)
async def delete_identity(request: DeleteIdentityRequest, db: Session = Depends(get_db)):
    """Delete an identity from the gallery"""
    try:
        # Find identity
        identity = db.query(db_models.Identity).filter(
            db_models.Identity.id == request.identity_id
        ).first()
        
        if not identity:
            raise HTTPException(status_code=404, detail="Identity not found")
        
        # Delete from matcher
        matcher.remove_identity(request.identity_id)
        
        # Delete from database (cascade will handle embeddings)
        db.delete(identity)
        db.commit()
        
        return DeleteIdentityResponse(
            success=True,
            message=f"Identity {identity.name} deleted successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in delete_identity endpoint: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    models_loaded = (
        detector is not None and
        embedding_extractor is not None and
        matcher is not None
    )
    
    index_size = matcher.index.ntotal if matcher else 0
    uptime = time.time() - app_start_time
    
    return HealthResponse(
        status="healthy" if models_loaded else "unhealthy",
        version="1.0.0",
        models_loaded=models_loaded,
        index_size=index_size,
        uptime_seconds=uptime
    )