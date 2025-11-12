"""Image processing utilities for face detection and recognition"""
import cv2
import numpy as np
from typing import Tuple, List, Optional
from PIL import Image


def load_image(image_path: str) -> np.ndarray:
    """Load image from path and convert to RGB"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_image(image: np.ndarray, max_size: int = 1920) -> np.ndarray:
    """Resize image while maintaining aspect ratio"""
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image


def calculate_blur_score(image: np.ndarray) -> float:
    """Calculate blur score using Laplacian variance"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def align_face(image: np.ndarray, landmarks: np.ndarray, output_size: int = 112) -> np.ndarray:
    """Align face using 5-point landmarks"""
    reference = np.array([
        [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
        [41.5493, 92.3655], [70.7299, 92.2041]
    ], dtype=np.float32)
    
    if output_size != 112:
        scale = output_size / 112.0
        reference *= scale
    
    landmarks = np.array(landmarks, dtype=np.float32)
    tform = cv2.estimateAffinePartial2D(landmarks, reference)[0]
    aligned = cv2.warpAffine(image, tform, (output_size, output_size), 
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return aligned


def normalize_face(face: np.ndarray, mean: List[float] = None, std: List[float] = None) -> np.ndarray:
    """Normalize face image for model input"""
    if mean is None:
        mean = [0.5, 0.5, 0.5]
    if std is None:
        std = [0.5, 0.5, 0.5]
    
    face = face.astype(np.float32) / 255.0
    face = (face - mean) / std
    return face


def draw_bbox(image: np.ndarray, bbox: List[int], label: str = "", 
              confidence: float = None, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """Draw bounding box and label on image"""
    img = image.copy()
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    if label:
        text = label
        if confidence is not None:
            text += f" ({confidence:.2f})"
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - text_height - baseline - 5), (x1 + text_width, y1), color, -1)
        cv2.putText(img, text, (x1, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img


def crop_face(image: np.ndarray, bbox: List[int], margin: float = 0.2) -> np.ndarray:
    """Crop face from image with margin"""
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    width, height = x2 - x1, y2 - y1
    margin_w, margin_h = int(width * margin), int(height * margin)
    x1, y1 = max(0, x1 - margin_w), max(0, y1 - margin_h)
    x2, y2 = min(w, x2 + margin_w), min(h, y2 + margin_h)
    return image[y1:y2, x1:x2]


def compute_face_quality(face: np.ndarray) -> float:
    """Compute face quality score based on blur, brightness, contrast"""
    blur_score = min(calculate_blur_score(face) / 500.0, 1.0)
    gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY) if len(face.shape) == 3 else face
    brightness = np.mean(gray) / 255.0
    brightness_score = 1.0 - abs(brightness - 0.5) * 2
    contrast_score = min(np.std(gray) / 128.0, 1.0)
    resolution_score = min(min(face.shape[:2]) / 112.0, 1.0)
    return 0.3 * blur_score + 0.25 * brightness_score + 0.25 * contrast_score + 0.2 * resolution_score