from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# COCO dataset class labels. The model returns numbers (1, 2, 3...)
# and we map them to human-readable names using this list.
# Index 0 is empty because COCO labels are 1-indexed.
COCO_LABELS = [
    '', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', '',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack',
    'umbrella', '', '', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '',
    'toilet', '', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Load model once at startup, not on every request.
# hub.load() downloads the model weights the first time (few hundred MB),
# then caches them locally. Subsequent startups are instant.
print("Loading model... this may take a moment on first run")
model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1")
print("Model loaded successfully")


@app.get("/")
def root():
    return {"message": "Object Detection API is running"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Read the raw bytes sent from React
    contents = await file.read()
    
    # Convert bytes → PIL Image → RGB
    # We force RGB because some images (PNG with transparency) are RGBA,
    # and the model only understands 3-channel images.
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Convert PIL Image → NumPy array
    # Shape becomes (height, width, 3) — 3 for R, G, B channels
    image_np = np.array(image)
    
    # The model expects a batch of images, not a single image.
    # np.expand_dims adds a batch dimension: (height, width, 3) → (1, height, width, 3)
    # Think of it as wrapping your single image in a list of one.
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]  # same as expand_dims
    
    # Run inference — this is the actual detection
    results = model(input_tensor)
    
    # The model returns TensorFlow tensors. .numpy() converts them to NumPy arrays
    # [0] gets the first (and only) item in the batch
    boxes = results["detection_boxes"][0].numpy()      # shape: (100, 4)
    scores = results["detection_scores"][0].numpy()    # shape: (100,)
    classes = results["detection_classes"][0].numpy()  # shape: (100,)
    
    # Get original image dimensions — needed to convert normalized coords to pixels
    img_height, img_width = image_np.shape[:2]
    
    detections = []
    
    # The model always returns 100 detections, sorted by confidence.
    # We filter to only keep ones above our threshold.
    CONFIDENCE_THRESHOLD = 0.5
    
    for i in range(len(scores)):
        if scores[i] < CONFIDENCE_THRESHOLD:
            break  # Since sorted by score, once we hit low confidence we can stop
        
        class_id = int(classes[i])
        label = COCO_LABELS[class_id] if class_id < len(COCO_LABELS) else "unknown"
        
        # boxes are normalized [0,1]. Multiply by image dimensions to get pixel coords.
        # Format from model: [ymin, xmin, ymax, xmax]
        ymin, xmin, ymax, xmax = boxes[i]
        detections.append({
            "label": label,
            "confidence": float(scores[i]),
            "box": {
                "xmin": float(xmin * img_width),
                "ymin": float(ymin * img_height),
                "xmax": float(xmax * img_width),
                "ymax": float(ymax * img_height),
            }
        })
    
    return {"detections": detections}