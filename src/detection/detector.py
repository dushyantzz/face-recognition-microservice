"""Face detection module using RetinaFace"""
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
import onnxruntime as ort
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Face detection result"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    landmarks: np.ndarray  # 5 facial landmarks
    quality_score: float
    
    def to_dict(self) -> Dict:
        return {
            "bbox": list(self.bbox),
            "confidence": float(self.confidence),
            "landmarks": self.landmarks.tolist(),
            "quality_score": float(self.quality_score)
        }


class RetinaFaceDetector:
    """RetinaFace detector with ONNX optimization"""
    
    def __init__(
        self,
        model_path: str,
        use_onnx: bool = True,
        confidence_threshold: float = 0.8,
        nms_threshold: float = 0.4,
        min_face_size: int = 40
    ):
        self.model_path = model_path
        self.use_onnx = use_onnx
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.min_face_size = min_face_size
        
        # Input settings
        self.input_size = (640, 640)
        self.mean = np.array([104, 117, 123], dtype=np.float32)
        
        self._load_model()
        
    def _load_model(self):
        """Load detection model"""
        if self.use_onnx:
            logger.info(f"Loading ONNX model from {self.model_path}")
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            
            providers = ['CPUExecutionProvider']
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=providers
            )
            self.input_name = self.session.get_inputs()[0].name
            logger.info(f"ONNX model loaded successfully")
        else:
            # Fallback to PyTorch if needed
            raise NotImplementedError("PyTorch model loading not implemented yet")
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Preprocess image for detection"""
        # Get original dimensions
        h, w = image.shape[:2]
        
        # Calculate resize scale
        scale = min(self.input_size[0] / h, self.input_size[1] / w)
        
        # Resize image
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create padded image
        padded = np.zeros((self.input_size[0], self.input_size[1], 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        # Convert to float and subtract mean
        img = padded.astype(np.float32)
        img -= self.mean
        
        # Transpose to CHW format and add batch dimension
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        
        return img, scale
    
    def postprocess(
        self,
        outputs: List[np.ndarray],
        scale: float,
        orig_shape: Tuple[int, int]
    ) -> List[DetectionResult]:
        """Postprocess detection outputs"""
        # Parse outputs (boxes, scores, landmarks)
        boxes = outputs[0]  # [N, 4]
        scores = outputs[1]  # [N, 1]
        landmarks = outputs[2]  # [N, 10] (5 points x 2 coordinates)
        
        detections = []
        
        for i in range(boxes.shape[0]):
            score = scores[i, 0]
            
            if score < self.confidence_threshold:
                continue
            
            # Get box coordinates
            box = boxes[i] / scale
            x1, y1, x2, y2 = box.astype(int)
            
            # Clip to image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(orig_shape[1], x2)
            y2 = min(orig_shape[0], y2)
            
            # Filter small faces
            if (x2 - x1) < self.min_face_size or (y2 - y1) < self.min_face_size:
                continue
            
            # Get landmarks
            lm = landmarks[i].reshape(5, 2) / scale
            
            # Calculate quality score
            quality = self._calculate_quality(
                box=(x1, y1, x2, y2),
                landmarks=lm,
                confidence=score
            )
            
            detection = DetectionResult(
                bbox=(x1, y1, x2, y2),
                confidence=float(score),
                landmarks=lm,
                quality_score=quality
            )
            detections.append(detection)
        
        # Apply NMS
        detections = self._nms(detections)
        
        return detections
    
    def _calculate_quality(
        self,
        box: Tuple[int, int, int, int],
        landmarks: np.ndarray,
        confidence: float
    ) -> float:
        """Calculate face quality score based on multiple factors"""
        x1, y1, x2, y2 = box
        
        # Size factor (larger faces generally better quality)
        face_area = (x2 - x1) * (y2 - y1)
        size_score = min(1.0, face_area / (112 * 112))  # Normalize to standard face size
        
        # Landmark spread (measures if face is frontal)
        eye_distance = np.linalg.norm(landmarks[0] - landmarks[1])
        face_width = x2 - x1
        spread_score = min(1.0, eye_distance / (face_width * 0.3))  # Expected eye distance is ~30% of width
        
        # Confidence score
        conf_score = confidence
        
        # Combined quality score
        quality = (size_score * 0.3 + spread_score * 0.3 + conf_score * 0.4)
        
        return quality
    
    def _nms(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """Non-maximum suppression"""
        if len(detections) == 0:
            return []
        
        boxes = np.array([d.bbox for d in detections])
        scores = np.array([d.confidence for d in detections])
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= self.nms_threshold)[0]
            order = order[inds + 1]
        
        return [detections[i] for i in keep]
    
    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """Detect faces in image"""
        if image is None or image.size == 0:
            logger.warning("Empty image provided")
            return []
        
        orig_shape = image.shape[:2]
        
        # Preprocess
        input_tensor, scale = self.preprocess(image)
        
        # Run inference
        if self.use_onnx:
            outputs = self.session.run(None, {self.input_name: input_tensor})
        else:
            raise NotImplementedError("PyTorch inference not implemented")
        
        # Postprocess
        detections = self.postprocess(outputs, scale, orig_shape)
        
        logger.info(f"Detected {len(detections)} faces")
        return detections
    
    def detect_and_align(self, image: np.ndarray, target_size: int = 112) -> List[Tuple[np.ndarray, DetectionResult]]:
        """Detect faces and return aligned face crops"""
        detections = self.detect(image)
        
        aligned_faces = []
        for detection in detections:
            aligned_face = self._align_face(image, detection, target_size)
            if aligned_face is not None:
                aligned_faces.append((aligned_face, detection))
        
        return aligned_faces
    
    def _align_face(self, image: np.ndarray, detection: DetectionResult, target_size: int = 112) -> Optional[np.ndarray]:
        """Align face using 5-point landmarks"""
        landmarks = detection.landmarks
        
        # Standard 5-point landmarks for 112x112 face
        src = np.array([
            [30.2946, 51.6963],  # Left eye
            [65.5318, 51.5014],  # Right eye
            [48.0252, 71.7366],  # Nose
            [33.5493, 92.3655],  # Left mouth
            [62.7299, 92.2041]   # Right mouth
        ], dtype=np.float32)
        
        # Scale to target size
        src = src * (target_size / 112.0)
        
        # Calculate similarity transform
        tform = cv2.estimateAffinePartial2D(landmarks, src)[0]
        
        if tform is None:
            logger.warning("Failed to estimate transform for face alignment")
            return None
        
        # Apply transform
        aligned = cv2.warpAffine(
            image,
            tform,
            (target_size, target_size),
            flags=cv2.INTER_LINEAR
        )
        
        return aligned