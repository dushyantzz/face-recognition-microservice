"""Evaluation script for face recognition metrics"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from pathlib import Path
import json
import time
from typing import List, Dict, Tuple
import argparse
from tqdm import tqdm
from collections import defaultdict

from src.detection import RetinaFaceDetector
from src.embeddings import AdaFaceExtractor
from src.matching import FaissIndexMatcher
from src.config import settings


class RecognitionEvaluator:
    """Evaluator for face recognition performance"""
    
    def __init__(
        self,
        detector: RetinaFaceDetector,
        extractor: AdaFaceExtractor,
        matcher: FaissIndexMatcher
    ):
        self.detector = detector
        self.extractor = extractor
        self.matcher = matcher
    
    def build_gallery(self, gallery_dir: str) -> int:
        """Build gallery from directory structure"""
        gallery_dir = Path(gallery_dir)
        identity_id = 0
        num_added = 0
        
        print("Building gallery...")
        
        for identity_folder in tqdm(sorted(gallery_dir.iterdir())):
            if not identity_folder.is_dir():
                continue
            
            identity_name = identity_folder.name
            identity_id += 1
            
            # Process all images for this identity
            embeddings = []
            for img_path in identity_folder.glob("*.jpg") or identity_folder.glob("*.png"):
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                # Detect and align
                aligned_faces = self.detector.detect_and_align(image)
                if not aligned_faces:
                    continue
                
                # Use best quality face
                aligned_faces.sort(key=lambda x: x[1].quality_score, reverse=True)
                face, _ = aligned_faces[0]
                
                # Extract embedding
                embedding = self.extractor.extract_embedding(face)
                if embedding is not None:
                    embeddings.append(embedding)
            
            # Add to gallery
            if embeddings:
                # Use average embedding or multiple
                for emb in embeddings:
                    self.matcher.add_identity(
                        embedding=emb,
                        identity_id=identity_id,
                        identity_name=identity_name,
                        metadata={"folder": identity_name}
                    )
                    num_added += 1
        
        print(f"Gallery built: {identity_id} identities, {num_added} embeddings")
        return identity_id
    
    def evaluate_probe_set(self, probe_dir: str) -> Dict:
        """Evaluate on probe set"""
        probe_dir = Path(probe_dir)
        
        # Collect results
        results = {
            "top1_correct": 0,
            "top5_correct": 0,
            "total_probes": 0,
            "per_identity": defaultdict(lambda: {"correct": 0, "total": 0}),
            "inference_times": [],
            "similarities": []
        }
        
        print("\nEvaluating probe set...")
        
        for identity_folder in tqdm(sorted(probe_dir.iterdir())):
            if not identity_folder.is_dir():
                continue
            
            identity_name = identity_folder.name
            
            for img_path in identity_folder.glob("*.jpg") or identity_folder.glob("*.png"):
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                # Detect and align
                start_time = time.time()
                aligned_faces = self.detector.detect_and_align(image)
                
                if not aligned_faces:
                    continue
                
                # Use best quality face
                aligned_faces.sort(key=lambda x: x[1].quality_score, reverse=True)
                face, _ = aligned_faces[0]
                
                # Extract embedding
                embedding = self.extractor.extract_embedding(face)
                if embedding is None:
                    continue
                
                # Search for matches
                matches = self.matcher.search(embedding, top_k=5)
                inference_time = (time.time() - start_time) * 1000
                
                results["inference_times"].append(inference_time)
                results["total_probes"] += 1
                results["per_identity"][identity_name]["total"] += 1
                
                if matches:
                    # Top-1 accuracy
                    if matches[0].identity_name == identity_name:
                        results["top1_correct"] += 1
                        results["per_identity"][identity_name]["correct"] += 1
                    
                    # Top-5 accuracy
                    top5_names = [m.identity_name for m in matches[:5]]
                    if identity_name in top5_names:
                        results["top5_correct"] += 1
                    
                    # Store similarity
                    if matches[0].identity_name == identity_name:
                        results["similarities"].append(matches[0].similarity)
        
        return results
    
    def compute_metrics(self, results: Dict) -> Dict:
        """Compute final metrics"""
        total = results["total_probes"]
        
        if total == 0:
            return {"error": "No probes processed"}
        
        metrics = {
            "identification_rate_top1": results["top1_correct"] / total,
            "identification_rate_top5": results["top5_correct"] / total,
            "total_probes": total,
            "correct_top1": results["top1_correct"],
            "correct_top5": results["top5_correct"],
            "average_inference_time_ms": np.mean(results["inference_times"]),
            "std_inference_time_ms": np.std(results["inference_times"]),
            "fps": 1000.0 / np.mean(results["inference_times"]),
            "average_similarity": np.mean(results["similarities"]) if results["similarities"] else 0,
            "per_identity_accuracy": {}
        }
        
        # Per-identity metrics
        for identity, stats in results["per_identity"].items():
            if stats["total"] > 0:
                metrics["per_identity_accuracy"][identity] = stats["correct"] / stats["total"]
        
        return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate face recognition")
    parser.add_argument("--gallery", type=str, required=True, help="Path to gallery directory")
    parser.add_argument("--probe", type=str, required=True, help="Path to probe directory")
    parser.add_argument("--detector-model", type=str, default="models/onnx/retinaface.onnx")
    parser.add_argument("--embedding-model", type=str, default="models/onnx/adaface.onnx")
    parser.add_argument("--output", type=str, default="results/recognition_metrics.json")
    parser.add_argument("--similarity-threshold", type=float, default=0.6)
    
    args = parser.parse_args()
    
    # Initialize models
    print("Loading models...")
    detector = RetinaFaceDetector(
        model_path=args.detector_model,
        use_onnx=True,
        confidence_threshold=settings.detection_confidence_threshold,
        min_face_size=settings.min_face_size
    )
    
    extractor = AdaFaceExtractor(
        model_path=args.embedding_model,
        use_onnx=True
    )
    
    matcher = FaissIndexMatcher(
        embedding_dim=512,
        similarity_threshold=args.similarity_threshold,
        top_k=5,
        index_type='flat'
    )
    
    # Initialize evaluator
    evaluator = RecognitionEvaluator(detector, extractor, matcher)
    
    # Build gallery
    num_identities = evaluator.build_gallery(args.gallery)
    
    # Evaluate on probe set
    results = evaluator.evaluate_probe_set(args.probe)
    
    # Compute metrics
    metrics = evaluator.compute_metrics(results)
    
    # Print results
    print("\n" + "="*60)
    print("RECOGNITION EVALUATION RESULTS")
    print("="*60)
    print(f"Gallery Size: {num_identities} identities")
    print(f"Total Probes: {metrics['total_probes']}")
    print(f"Top-1 Identification Rate: {metrics['identification_rate_top1']:.4f} ({metrics['correct_top1']}/{metrics['total_probes']})")
    print(f"Top-5 Identification Rate: {metrics['identification_rate_top5']:.4f} ({metrics['correct_top5']}/{metrics['total_probes']})")
    print(f"Average Inference Time: {metrics['average_inference_time_ms']:.2f} ± {metrics['std_inference_time_ms']:.2f} ms")
    print(f"FPS: {metrics['fps']:.2f}")
    print(f"Average Similarity (correct matches): {metrics['average_similarity']:.4f}")
    print("="*60)
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()