"""API request and response schemas"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class BoundingBox(BaseModel):
    """Bounding box coordinates"""
    x1: int
    y1: int
    x2: int
    y2: int


class Landmarks(BaseModel):
    """Facial landmarks (5 points)"""
    points: List[List[float]] = Field(description="List of [x, y] coordinates")


class Detection(BaseModel):
    """Face detection result"""
    bbox: BoundingBox
    confidence: float = Field(ge=0.0, le=1.0)
    landmarks: Landmarks
    quality_score: float = Field(ge=0.0, le=1.0)


class Match(BaseModel):
    """Face match result"""
    identity_id: int
    identity_name: str
    similarity: float = Field(ge=0.0, le=1.0)
    rank: int
    metadata: Optional[Dict[str, Any]] = None


class RecognitionResult(BaseModel):
    """Face recognition result combining detection and matching"""
    bbox: BoundingBox
    detection_confidence: float
    matches: List[Match]
    quality_score: float


class DetectRequest(BaseModel):
    """Request for face detection"""
    image: str = Field(description="Base64 encoded image or image URL")
    min_face_size: Optional[int] = Field(default=40, ge=20)
    confidence_threshold: Optional[float] = Field(default=0.8, ge=0.0, le=1.0)


class DetectResponse(BaseModel):
    """Response for face detection"""
    detections: List[Detection]
    num_faces: int
    processing_time_ms: float


class RecognizeRequest(BaseModel):
    """Request for face recognition"""
    image: str = Field(description="Base64 encoded image or image URL")
    top_k: Optional[int] = Field(default=5, ge=1, le=20)
    similarity_threshold: Optional[float] = Field(default=0.6, ge=0.0, le=1.0)
    min_face_size: Optional[int] = Field(default=40, ge=20)


class RecognizeResponse(BaseModel):
    """Response for face recognition"""
    results: List[RecognitionResult]
    num_faces: int
    processing_time_ms: float


class AddIdentityRequest(BaseModel):
    """Request to add a new identity"""
    name: str = Field(min_length=1, max_length=100)
    images: List[str] = Field(description="List of base64 encoded images", min_length=1)
    metadata: Optional[Dict[str, Any]] = None


class AddIdentityResponse(BaseModel):
    """Response for adding identity"""
    identity_id: int
    identity_name: str
    num_embeddings_added: int
    processing_time_ms: float


class Identity(BaseModel):
    """Identity information"""
    identity_id: int
    identity_name: str
    num_images: int
    created_at: datetime
    updated_at: datetime
    metadata: Optional[Dict[str, Any]] = None


class ListIdentitiesResponse(BaseModel):
    """Response for listing identities"""
    identities: List[Identity]
    total_count: int


class DeleteIdentityRequest(BaseModel):
    """Request to delete an identity"""
    identity_id: int


class DeleteIdentityResponse(BaseModel):
    """Response for deleting identity"""
    success: bool
    message: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    models_loaded: bool
    index_size: int
    uptime_seconds: float


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None