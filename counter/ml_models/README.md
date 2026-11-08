# Detector models

The project runs out of the box with OpenCV's built-in HOG pedestrian
detector, which needs no files in this folder.

For better accuracy, drop a YOLOv8 ONNX export here and switch backends:

```bash
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', opset=12)"
mv yolov8n.onnx counter/ml_models/

export COUNTER_DETECTOR=onnx
```

The loader falls back to HOG if the file is missing or unreadable, so a bad
export never takes the app down. `.onnx` files are gitignored.
