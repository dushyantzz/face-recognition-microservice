"""Convert PyTorch models to ONNX format for optimized CPU inference"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
import argparse
from pathlib import Path
import time


class ONNXConverter:
    """Convert and optimize PyTorch models to ONNX"""
    
    @staticmethod
    def convert_detection_model(
        pytorch_model_path: str,
        onnx_output_path: str,
        input_size: tuple = (640, 640),
        opset_version: int = 12
    ):
        """Convert RetinaFace detection model to ONNX"""
        print(f"Converting detection model: {pytorch_model_path}")
        
        try:
            # Load PyTorch model
            from retinaface import RetinaFace
            model = RetinaFace.load_model(pytorch_model_path)
            model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
            
            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                onnx_output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['boxes', 'scores', 'landmarks'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'boxes': {0: 'batch_size'},
                    'scores': {0: 'batch_size'},
                    'landmarks': {0: 'batch_size'}
                }
            )
            
            print(f"Detection model converted successfully: {onnx_output_path}")
            return True
            
        except Exception as e:
            print(f"Error converting detection model: {e}")
            return False
    
    @staticmethod
    def convert_embedding_model(
        pytorch_model_path: str,
        onnx_output_path: str,
        input_size: int = 112,
        opset_version: int = 12
    ):
        """Convert AdaFace embedding model to ONNX"""
        print(f"Converting embedding model: {pytorch_model_path}")
        
        try:
            # Load PyTorch model (example - adjust based on actual model)
            import torch.nn as nn
            
            # This is a placeholder - replace with actual model loading
            # For AdaFace, you would load the checkpoint properly
            model = torch.load(pytorch_model_path, map_location='cpu')
            if isinstance(model, dict) and 'state_dict' in model:
                # Load state dict into model architecture
                from adaface import build_model
                net = build_model('ir_101')
                net.load_state_dict(model['state_dict'])
                model = net
            
            model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(1, 3, input_size, input_size)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                onnx_output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['embedding'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'embedding': {0: 'batch_size'}
                }
            )
            
            print(f"Embedding model converted successfully: {onnx_output_path}")
            return True
            
        except Exception as e:
            print(f"Error converting embedding model: {e}")
            print("Note: You may need to install the model's original package")
            return False
    
    @staticmethod
    def optimize_onnx_model(onnx_path: str, optimized_path: str = None):
        """Optimize ONNX model for inference"""
        if optimized_path is None:
            optimized_path = onnx_path.replace('.onnx', '_optimized.onnx')
        
        print(f"Optimizing ONNX model: {onnx_path}")
        
        try:
            # Load model
            model = onnx.load(onnx_path)
            
            # Check model
            onnx.checker.check_model(model)
            
            # Optimize
            from onnxruntime.transformers import optimizer
            from onnxruntime.transformers.fusion_options import FusionOptions
            
            optimization_options = FusionOptions('bert')
            optimization_options.enable_gelu = True
            optimization_options.enable_layer_norm = True
            optimization_options.enable_attention = True
            optimization_options.enable_skip_layer_norm = True
            optimization_options.enable_embed_layer_norm = True
            optimization_options.enable_bias_skip_layer_norm = True
            optimization_options.enable_bias_gelu = True
            optimization_options.enable_gelu_approximation = False
            
            optimized_model = optimizer.optimize_model(
                onnx_path,
                model_type='bert',
                num_heads=0,
                hidden_size=0,
                optimization_options=optimization_options
            )
            
            optimized_model.save_model_to_file(optimized_path)
            print(f"Optimized model saved: {optimized_path}")
            return True
            
        except Exception as e:
            print(f"Error optimizing model: {e}")
            # Fallback: just copy the model
            import shutil
            shutil.copy(onnx_path, optimized_path)
            print("Using unoptimized model")
            return False
    
    @staticmethod
    def benchmark_onnx_model(onnx_path: str, input_shape: tuple, num_iterations: int = 100):
        """Benchmark ONNX model inference speed"""
        print(f"\nBenchmarking model: {onnx_path}")
        print(f"Input shape: {input_shape}")
        
        try:
            # Create session
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4
            
            session = ort.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=['CPUExecutionProvider']
            )
            
            input_name = session.get_inputs()[0].name
            
            # Create dummy input
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
            
            # Warmup
            for _ in range(10):
                session.run(None, {input_name: dummy_input})
            
            # Benchmark
            times = []
            for _ in range(num_iterations):
                start = time.time()
                session.run(None, {input_name: dummy_input})
                times.append((time.time() - start) * 1000)
            
            times = np.array(times)
            
            print(f"\nBenchmark Results ({num_iterations} iterations):")
            print(f"  Average: {times.mean():.2f} ms")
            print(f"  Std Dev: {times.std():.2f} ms")
            print(f"  Min: {times.min():.2f} ms")
            print(f"  Max: {times.max():.2f} ms")
            print(f"  FPS: {1000.0 / times.mean():.2f}")
            
            return {
                "average_ms": float(times.mean()),
                "std_ms": float(times.std()),
                "min_ms": float(times.min()),
                "max_ms": float(times.max()),
                "fps": float(1000.0 / times.mean())
            }
            
        except Exception as e:
            print(f"Error benchmarking model: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch models to ONNX")
    parser.add_argument("--model-type", type=str, choices=['detection', 'embedding', 'both'], default='both')
    parser.add_argument("--detection-model", type=str, default="models/detection/retinaface_resnet50.pth")
    parser.add_argument("--embedding-model", type=str, default="models/embeddings/adaface_ir101_webface12m.ckpt")
    parser.add_argument("--output-dir", type=str, default="models/onnx")
    parser.add_argument("--optimize", action="store_true", help="Optimize ONNX models")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark converted models")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    converter = ONNXConverter()
    
    # Convert detection model
    if args.model_type in ['detection', 'both']:
        detection_onnx = os.path.join(args.output_dir, "retinaface.onnx")
        
        if os.path.exists(args.detection_model):
            success = converter.convert_detection_model(
                args.detection_model,
                detection_onnx
            )
            
            if success and args.optimize:
                converter.optimize_onnx_model(detection_onnx)
            
            if success and args.benchmark:
                converter.benchmark_onnx_model(
                    detection_onnx,
                    input_shape=(1, 3, 640, 640),
                    num_iterations=100
                )
        else:
            print(f"Detection model not found: {args.detection_model}")
            print("Skipping detection model conversion")
    
    # Convert embedding model
    if args.model_type in ['embedding', 'both']:
        embedding_onnx = os.path.join(args.output_dir, "adaface.onnx")
        
        if os.path.exists(args.embedding_model):
            success = converter.convert_embedding_model(
                args.embedding_model,
                embedding_onnx
            )
            
            if success and args.optimize:
                converter.optimize_onnx_model(embedding_onnx)
            
            if success and args.benchmark:
                converter.benchmark_onnx_model(
                    embedding_onnx,
                    input_shape=(1, 3, 112, 112),
                    num_iterations=100
                )
        else:
            print(f"Embedding model not found: {args.embedding_model}")
            print("Skipping embedding model conversion")
    
    print("\nConversion complete!")


if __name__ == "__main__":
    main()