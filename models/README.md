# Models Directory

## Structure

### detection/
Face detection models:
- RetinaFace with ResNet50 backbone
- Download pretrained weights

### embeddings/
Face embedding models:
- AdaFace IR101 trained on WebFace12M
- ArcFace alternatives

### onnx/
Optimized ONNX models for production deployment

## Model Downloads

### RetinaFace
```bash
# Download from official repo
wget https://github.com/biubug6/Pytorch_Retinaface/releases/download/resnet50/Resnet50_Final.pth
mv Resnet50_Final.pth models/detection/retinaface_resnet50.pth
```

### AdaFace
```bash
# Download from official repo
wget https://github.com/mk-minchul/AdaFace/releases/download/v1.0/adaface_ir101_webface12m.ckpt
mv adaface_ir101_webface12m.ckpt models/embeddings/
```