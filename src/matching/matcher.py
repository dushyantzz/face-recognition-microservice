"""Face matching pipeline with Faiss index"""
import numpy as np
from typing import List, Tuple, Optional, Dict
import faiss
import logging
from dataclasses import dataclass
import pickle
import os

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Face match result"""
    identity_id: int
    identity_name: str
    similarity: float
    rank: int
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return {
            "identity_id": self.identity_id,
            "identity_name": self.identity_name,
            "similarity": float(self.similarity),
            "rank": self.rank,
            "metadata": self.metadata or {}
        }


class FaissIndexMatcher:
    """Face matching using Faiss index for fast similarity search"""
    
    def __init__(
        self,
        embedding_dim: int = 512,
        similarity_threshold: float = 0.6,
        top_k: int = 5,
        index_type: str = 'flat',  # 'flat', 'ivf', 'hnsw'
        use_gpu: bool = False
    ):
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.index_type = index_type
        self.use_gpu = use_gpu
        
        # Initialize index
        self.index = None
        self.identity_map = {}  # Maps index position to identity info
        self.identity_embeddings = []  # Store embeddings for updates
        
        self._create_index()
    
    def _create_index(self):
        """Create Faiss index based on type"""
        if self.index_type == 'flat':
            # Flat L2 index (exact search)
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product for cosine similarity
        
        elif self.index_type == 'ivf':
            # IVF index for faster approximate search
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            n_clusters = 100  # Number of Voronoi cells
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, n_clusters, faiss.METRIC_INNER_PRODUCT)
        
        elif self.index_type == 'hnsw':
            # HNSW index for very fast approximate search
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32, faiss.METRIC_INNER_PRODUCT)
        
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        # Move to GPU if requested
        if self.use_gpu and faiss.get_num_gpus() > 0:
            logger.info("Moving Faiss index to GPU")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        logger.info(f"Created {self.index_type} index with dim={self.embedding_dim}")
    
    def add_identity(
        self,
        embedding: np.ndarray,
        identity_id: int,
        identity_name: str,
        metadata: Optional[Dict] = None
    ):
        """Add a single identity to the index"""
        # Ensure embedding is normalized and correct shape
        if len(embedding.shape) == 1:
            embedding = embedding.reshape(1, -1)
        
        # Normalize for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)
        
        # Get current index position
        idx_position = self.index.ntotal
        
        # Add to index
        if self.index_type == 'ivf' and not self.index.is_trained:
            # Train IVF index if not trained
            logger.info("Training IVF index...")
            # Need at least n_clusters samples for training
            if idx_position >= 100:
                training_data = np.vstack(self.identity_embeddings)
                self.index.train(training_data)
        
        self.index.add(embedding.astype(np.float32))
        
        # Store identity mapping
        self.identity_map[idx_position] = {
            "identity_id": identity_id,
            "identity_name": identity_name,
            "metadata": metadata or {}
        }
        
        # Store embedding for potential reindexing
        self.identity_embeddings.append(embedding[0])
        
        logger.info(f"Added identity '{identity_name}' (id={identity_id}) at position {idx_position}")
    
    def add_identities_batch(
        self,
        embeddings: np.ndarray,
        identity_ids: List[int],
        identity_names: List[str],
        metadata_list: Optional[List[Dict]] = None
    ):
        """Add multiple identities to the index"""
        if metadata_list is None:
            metadata_list = [None] * len(identity_ids)
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        # Get starting position
        start_idx = self.index.ntotal
        
        # Train IVF if needed
        if self.index_type == 'ivf' and not self.index.is_trained:
            logger.info("Training IVF index...")
            if len(embeddings) >= 100:
                self.index.train(embeddings.astype(np.float32))
        
        # Add to index
        self.index.add(embeddings.astype(np.float32))
        
        # Store identity mappings
        for i, (identity_id, identity_name, metadata) in enumerate(zip(identity_ids, identity_names, metadata_list)):
            idx_position = start_idx + i
            self.identity_map[idx_position] = {
                "identity_id": identity_id,
                "identity_name": identity_name,
                "metadata": metadata or {}
            }
            self.identity_embeddings.append(embeddings[i])
        
        logger.info(f"Added {len(identity_ids)} identities to index")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: Optional[int] = None
    ) -> List[MatchResult]:
        """Search for matching identities"""
        if self.index.ntotal == 0:
            logger.warning("Index is empty, no matches possible")
            return []
        
        if top_k is None:
            top_k = self.top_k
        
        # Ensure query is normalized and correct shape
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search index
        k = min(top_k, self.index.ntotal)
        similarities, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        # Parse results
        matches = []
        for rank, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
            # Filter by threshold
            if sim < self.similarity_threshold:
                continue
            
            # Get identity info
            if idx == -1:  # Faiss returns -1 for not found
                continue
            
            identity_info = self.identity_map.get(int(idx))
            if identity_info is None:
                logger.warning(f"Identity info not found for index {idx}")
                continue
            
            match = MatchResult(
                identity_id=identity_info["identity_id"],
                identity_name=identity_info["identity_name"],
                similarity=float(sim),
                rank=rank + 1,
                metadata=identity_info["metadata"]
            )
            matches.append(match)
        
        return matches
    
    def search_batch(
        self,
        query_embeddings: np.ndarray,
        top_k: Optional[int] = None
    ) -> List[List[MatchResult]]:
        """Search for multiple queries"""
        results = []
        for query in query_embeddings:
            matches = self.search(query, top_k)
            results.append(matches)
        return results
    
    def remove_identity(self, identity_id: int):
        """Remove an identity from the index"""
        # Find all positions with this identity_id
        positions_to_remove = []
        for pos, info in self.identity_map.items():
            if info["identity_id"] == identity_id:
                positions_to_remove.append(pos)
        
        if not positions_to_remove:
            logger.warning(f"Identity {identity_id} not found in index")
            return
        
        # Faiss doesn't support direct removal, so we need to rebuild
        logger.info(f"Rebuilding index to remove identity {identity_id}")
        
        # Keep embeddings and info for non-removed identities
        new_embeddings = []
        new_map = {}
        new_identity_embeddings = []
        
        new_idx = 0
        for pos in sorted(self.identity_map.keys()):
            if pos not in positions_to_remove:
                new_embeddings.append(self.identity_embeddings[pos])
                new_map[new_idx] = self.identity_map[pos]
                new_identity_embeddings.append(self.identity_embeddings[pos])
                new_idx += 1
        
        # Recreate index
        self._create_index()
        self.identity_map = new_map
        self.identity_embeddings = new_identity_embeddings
        
        # Re-add embeddings
        if new_embeddings:
            embeddings_array = np.vstack(new_embeddings)
            if self.index_type == 'ivf' and len(new_embeddings) >= 100:
                self.index.train(embeddings_array.astype(np.float32))
            self.index.add(embeddings_array.astype(np.float32))
        
        logger.info(f"Removed identity {identity_id} from index")
    
    def get_statistics(self) -> Dict:
        """Get index statistics"""
        unique_identities = len(set(info["identity_id"] for info in self.identity_map.values()))
        
        return {
            "total_embeddings": self.index.ntotal,
            "unique_identities": unique_identities,
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "similarity_threshold": self.similarity_threshold,
            "top_k": self.top_k
        }
    
    def save(self, path: str):
        """Save index and mappings to disk"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        # Save Faiss index
        index_path = f"{path}.index"
        if self.use_gpu:
            # Move to CPU before saving
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, index_path)
        else:
            faiss.write_index(self.index, index_path)
        
        # Save mappings and embeddings
        metadata_path = f"{path}.meta"
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'identity_map': self.identity_map,
                'identity_embeddings': self.identity_embeddings,
                'embedding_dim': self.embedding_dim,
                'similarity_threshold': self.similarity_threshold,
                'top_k': self.top_k,
                'index_type': self.index_type
            }, f)
        
        logger.info(f"Saved index to {path}")
    
    def load(self, path: str):
        """Load index and mappings from disk"""
        # Load Faiss index
        index_path = f"{path}.index"
        self.index = faiss.read_index(index_path)
        
        # Move to GPU if needed
        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        # Load mappings and embeddings
        metadata_path = f"{path}.meta"
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.identity_map = data['identity_map']
            self.identity_embeddings = data['identity_embeddings']
            self.embedding_dim = data['embedding_dim']
            self.similarity_threshold = data['similarity_threshold']
            self.top_k = data['top_k']
            self.index_type = data['index_type']
        
        logger.info(f"Loaded index from {path} with {self.index.ntotal} embeddings")