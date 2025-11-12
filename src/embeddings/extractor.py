"""Face embedding extraction module using AdaFace"""
import numpy as np
import cv2
from typing import Optional, List
import onnxruntime as ort
import logging

logger = logging.getLogger(__name__)


class AdaFaceExtractor:
    """AdaFace embedding extractor with ONNX optimization"""
    
    def __init__(
        self,
        model_path: str,
        use_onnx: bool = True,
        embedding_size: int = 512,
        input_size: int = 112
    ):
        self.model_path = model_path
        self.use_onnx = use_onnx
        self.embedding_size = embedding_size
        self.input_size = input_size
        
        # Normalization parameters (ImageNet)
        self.mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        
        self._load_model()
    
    def _load_model(self):
        """Load embedding model"""
        if self.use_onnx:
            logger.info(f"Loading ONNX embedding model from {self.model_path}")
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
            logger.info(f"ONNX embedding model loaded successfully")
        else:
            # Fallback to PyTorch if needed
            raise NotImplementedError("PyTorch model loading not implemented yet")
    
    def preprocess(self, face: np.ndarray) -> np.ndarray:
        """Preprocess face image for embedding extraction"""
        # Ensure correct size
        if face.shape[:2] != (self.input_size, self.input_size):
            face = cv2.resize(face, (self.input_size, self.input_size))
        
        # Convert BGR to RGB
        if len(face.shape) == 3 and face.shape[2] == 3:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        face = face.astype(np.float32) / 255.0
        
        # Apply mean and std normalization
        face = (face - self.mean) / self.std
        
        # Transpose to CHW format
        face = face.transpose(2, 0, 1)
        
        # Add batch dimension
        face = np.expand_dims(face, axis=0)
        
        return face
    
    def extract_embedding(self, face: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding from face image"""
        if face is None or face.size == 0:
            logger.warning("Empty face image provided")
            return None
        
        try:
            # Preprocess
            input_tensor = self.preprocess(face)
            
            # Run inference
            if self.use_onnx:
                outputs = self.session.run(None, {self.input_name: input_tensor})
                embedding = outputs[0][0]  # Remove batch dimension
            else:
                raise NotImplementedError("PyTorch inference not implemented")
            
            # L2 normalization
            embedding = self._normalize_embedding(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            return None
    
    def extract_batch_embeddings(self, faces: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """Extract embeddings from multiple faces"""
        if not faces:
            return []
        
        embeddings = []
        
        try:
            # Preprocess all faces
            input_tensors = []
            for face in faces:
                if face is not None and face.size > 0:
                    input_tensors.append(self.preprocess(face))
                else:
                    input_tensors.append(None)
            
            # Process valid faces
            batch_input = np.concatenate([t for t in input_tensors if t is not None], axis=0)
            
            if batch_input.shape[0] > 0:
                # Run batch inference
                if self.use_onnx:
                    outputs = self.session.run(None, {self.input_name: batch_input})
                    batch_embeddings = outputs[0]
                else:
                    raise NotImplementedError("PyTorch inference not implemented")
                
                # Normalize embeddings
                batch_embeddings = np.array([self._normalize_embedding(emb) for emb in batch_embeddings])
                
                # Map back to original order
                emb_idx = 0
                for tensor in input_tensors:
                    if tensor is not None:
                        embeddings.append(batch_embeddings[emb_idx])
                        emb_idx += 1
                    else:
                        embeddings.append(None)
            else:
                embeddings = [None] * len(faces)
                
        except Exception as e:
            logger.error(f"Error extracting batch embeddings: {e}")
            embeddings = [None] * len(faces)
        
        return embeddings
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """L2 normalize embedding"""
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
    
    @staticmethod
    def compute_similarity(emb1: np.ndarray, emb2: np.ndarray, metric: str = 'cosine') -> float:
        """Compute similarity between two embeddings"""
        if emb1 is None or emb2 is None:
            return 0.0
        
        if metric == 'cosine':
            # Cosine similarity (embeddings should be L2 normalized)
            similarity = np.dot(emb1, emb2)
            return float(similarity)
        
        elif metric == 'euclidean':
            # Euclidean distance (convert to similarity)
            distance = np.linalg.norm(emb1 - emb2)
            # Convert distance to similarity (closer to 0 means more similar)
            similarity = 1.0 / (1.0 + distance)
            return float(similarity)
        
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    
    @staticmethod
    def compute_batch_similarity(
        query_embedding: np.ndarray,
        gallery_embeddings: np.ndarray,
        metric: str = 'cosine'
    ) -> np.ndarray:
        """Compute similarity between query and multiple gallery embeddings"""
        if query_embedding is None or gallery_embeddings is None:
            return np.array([])
        
        if metric == 'cosine':
            # Matrix multiplication for cosine similarity
            similarities = np.dot(gallery_embeddings, query_embedding)
            return similarities
        
        elif metric == 'euclidean':
            # Euclidean distances
            distances = np.linalg.norm(gallery_embeddings - query_embedding, axis=1)
            # Convert to similarities
            similarities = 1.0 / (1.0 + distances)
            return similarities
        
        else:
            raise ValueError(f"Unsupported metric: {metric}")


class EmbeddingNormalizer:
    """Utility class for embedding normalization and preprocessing"""
    
    @staticmethod
    def whiten_embeddings(embeddings: np.ndarray) -> np.ndarray:
        """Apply ZCA whitening to embeddings"""
        # Center the data
        mean = np.mean(embeddings, axis=0)
        centered = embeddings - mean
        
        # Compute covariance matrix
        cov = np.cov(centered.T)
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Whitening transformation
        epsilon = 1e-5
        whitening_matrix = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues + epsilon)) @ eigenvectors.T
        
        # Apply whitening
        whitened = centered @ whitening_matrix
        
        return whitened
    
    @staticmethod
    def pca_reduction(embeddings: np.ndarray, n_components: int = 128) -> np.ndarray:
        """Apply PCA for dimensionality reduction"""
        # Center the data
        mean = np.mean(embeddings, axis=0)
        centered = embeddings - mean
        
        # Compute covariance matrix
        cov = np.cov(centered.T)
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort by eigenvalues (descending)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select top components
        pca_matrix = eigenvectors[:, :n_components]
        
        # Apply PCA
        reduced = centered @ pca_matrix
        
        return reduced