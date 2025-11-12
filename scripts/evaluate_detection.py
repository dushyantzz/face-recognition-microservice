"""Evaluation script for face detection metrics"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from pathlib import Path
import json
import time
from typing import List, Tuple, Dict
import argparse
from tqdm import tqdm

from src.detection import RetinaFaceDetector
from src.config import settings


class DetectionEvaluator:
    """Evaluator for face detection performance"""
    
    def __init__(self, detector: RetinaFaceDetector, iou_threshold: float = 0.5):
        self.detector = detector
        self.iou_threshold = iou_threshold
        
    def compute_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Compute IoU between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def evaluate_image(self, image_path: str, ground_truth_boxes: List[Tuple]) -> Dict:
        """Evaluate detection on a single image"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Failed to load image"}
        
        # Detect faces
        start_time = time.time()
        detections = self.detector.detect(image)
        inference_time = (time.time() - start_time) * 1000
        
        # Extract predicted boxes
        pred_boxes = [det.bbox for det in detections]
        pred_scores = [det.confidence for det in detections]
        
        # Match predictions to ground truth
        matched_gt = set()
        matched_pred = set()
        
        for i, pred_box in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt_box in enumerate(ground_truth_boxes):
                if j in matched_gt:
                    continue
                
                iou = self.compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= self.iou_threshold:
                matched_gt.add(best_gt_idx)
                matched_pred.add(i)
        
        # Compute metrics
        true_positives = len(matched_pred)
        false_positives = len(pred_boxes) - true_positives
        false_negatives = len(ground_truth_boxes) - len(matched_gt)
        
        return {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "num_predictions": len(pred_boxes),
            "num_ground_truth": len(ground_truth_boxes),
            "inference_time_ms": inference_time
        }
    
    def evaluate_dataset(self, dataset_path: str, annotations_file: str) -> Dict:
        """Evaluate on entire dataset"""
        # Load annotations
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_inference_time = 0
        num_images = 0
        
        print(f"Evaluating on {len(annotations)} images...")
        
        for image_name, gt_boxes in tqdm(annotations.items()):
            image_path = os.path.join(dataset_path, image_name)
            
            if not os.path.exists(image_path):
                print(f"Warning: Image {image_name} not found")
                continue
            
            result = self.evaluate_image(image_path, gt_boxes)
            
            if "error" in result:
                print(f"Error processing {image_name}: {result['error']}")
                continue
            
            total_tp += result["true_positives"]
            total_fp += result["false_positives"]
            total_fn += result["false_negatives"]
            total_inference_time += result["inference_time_ms"]
            num_images += 1
        
        # Compute overall metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        avg_inference_time = total_inference_time / num_images if num_images > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "average_precision": self._compute_ap(annotations, dataset_path),
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "num_images": num_images,
            "average_inference_time_ms": avg_inference_time,
            "fps": 1000.0 / avg_inference_time if avg_inference_time > 0 else 0
        }
    
    def _compute_ap(self, annotations: Dict, dataset_path: str) -> float:
        """Compute Average Precision (AP)"""
        all_detections = []
        all_ground_truths = []
        
        for image_name, gt_boxes in annotations.items():
            image_path = os.path.join(dataset_path, image_name)
            
            if not os.path.exists(image_path):
                continue
            
            image = cv2.imread(image_path)
            if image is None:
                continue
            
            detections = self.detector.detect(image)
            
            for det in detections:
                all_detections.append({
                    "image": image_name,
                    "box": det.bbox,
                    "score": det.confidence
                })
            
            for gt_box in gt_boxes:
                all_ground_truths.append({
                    "image": image_name,
                    "box": gt_box
                })
        
        # Sort detections by confidence
        all_detections.sort(key=lambda x: x["score"], reverse=True)
        
        # Compute precision-recall curve
        tp = 0
        fp = 0
        precisions = []
        recalls = []
        
        matched_gt = set()
        
        for det in all_detections:
            # Find matching ground truth
            best_iou = 0
            best_gt = None
            
            for i, gt in enumerate(all_ground_truths):
                if gt["image"] != det["image"]:
                    continue
                
                gt_key = f"{gt['image']}_{i}"
                if gt_key in matched_gt:
                    continue
                
                iou = self.compute_iou(det["box"], gt["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt_key
            
            if best_iou >= self.iou_threshold and best_gt:
                tp += 1
                matched_gt.add(best_gt)
            else:
                fp += 1
            
            precision = tp / (tp + fp)
            recall = tp / len(all_ground_truths) if len(all_ground_truths) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Compute AP using 11-point interpolation
        ap = 0
        for threshold in np.arange(0, 1.1, 0.1):
            precisions_above_threshold = [p for p, r in zip(precisions, recalls) if r >= threshold]
            if precisions_above_threshold:
                ap += max(precisions_above_threshold)
        ap /= 11
        
        return ap


def main():
    parser = argparse.ArgumentParser(description="Evaluate face detection")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--annotations", type=str, required=True, help="Path to annotations JSON file")
    parser.add_argument("--model", type=str, default="models/onnx/retinaface.onnx", help="Path to detection model")
    parser.add_argument("--output", type=str, default="results/detection_metrics.json", help="Output file for metrics")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for matching")
    
    args = parser.parse_args()
    
    # Initialize detector
    print("Loading detector...")
    detector = RetinaFaceDetector(
        model_path=args.model,
        use_onnx=True,
        confidence_threshold=settings.detection_confidence_threshold,
        min_face_size=settings.min_face_size
    )
    
    # Initialize evaluator
    evaluator = DetectionEvaluator(detector, iou_threshold=args.iou_threshold)
    
    # Run evaluation
    print("\nRunning evaluation...")
    metrics = evaluator.evaluate_dataset(args.dataset, args.annotations)
    
    # Print results
    print("\n" + "="*60)
    print("DETECTION EVALUATION RESULTS")
    print("="*60)
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Average Precision (AP): {metrics['average_precision']:.4f}")
    print(f"True Positives: {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"False Negatives: {metrics['false_negatives']}")
    print(f"Num Images: {metrics['num_images']}")
    print(f"Average Inference Time: {metrics['average_inference_time_ms']:.2f} ms")
    print(f"FPS: {metrics['fps']:.2f}")
    print("="*60)
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()