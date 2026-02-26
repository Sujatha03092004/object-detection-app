from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# COCO class names — YOLOv8 uses the same 80 COCO classes
COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Load ONNX model once at startup
# ONNX Runtime is much lighter than TensorFlow — ~100MB RAM vs ~800MB
print("Loading YOLOv8 ONNX model...")
session = ort.InferenceSession("yolov8n.onnx")
input_name = session.get_inputs()[0].name  # gets the name of the input tensor
print("Model loaded successfully")

CONFIDENCE_THRESHOLD = 0.5
INPUT_SIZE = 640  # YOLOv8 expects 640x640 input


def preprocess(image: Image.Image) -> tuple[np.ndarray, int, int]:
    """
    Resize image to 640x640 with letterboxing (padding to preserve aspect ratio),
    normalize pixel values to [0,1], and reorder to BCHW format.
    
    Returns the processed tensor plus original dimensions for scaling boxes back.
    """
    orig_w, orig_h = image.size

    # Letterbox: scale image to fit 640x640 while keeping aspect ratio
    # Then pad the remaining space with grey (114, 114, 114)
    scale = min(INPUT_SIZE / orig_w, INPUT_SIZE / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    resized = image.resize((new_w, new_h), Image.BILINEAR)

    # Create grey canvas and paste resized image centered
    canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (114, 114, 114))
    pad_x = (INPUT_SIZE - new_w) // 2
    pad_y = (INPUT_SIZE - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))

    # Convert to numpy, normalize to [0,1], change from HWC to CHW, add batch dim
    img_np = np.array(canvas, dtype=np.float32) / 255.0
    img_np = img_np.transpose(2, 0, 1)   # HWC → CHW
    img_np = np.expand_dims(img_np, 0)   # CHW → BCHW

    return img_np, orig_w, orig_h, pad_x, pad_y, scale


def postprocess(outputs, orig_w, orig_h, pad_x, pad_y, scale):
    """
    YOLOv8 output shape: (1, 84, 8400)
    84 = 4 box coords + 80 class scores
    8400 = number of candidate detections
    
    We transpose to (8400, 84), extract boxes and scores,
    filter by confidence, and scale boxes back to original image size.
    """
    predictions = outputs[0][0].T  # shape: (8400, 84)

    boxes_out = []

    for pred in predictions:
        # First 4 values are box: cx, cy, w, h (center format, normalized to 640)
        cx, cy, w, h = pred[:4]
        # Remaining 80 values are class scores
        class_scores = pred[4:]
        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        # Convert from center format to corner format
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # Remove letterbox padding and scale back to original image coordinates
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        # Clamp to image bounds
        x1 = max(0, min(x1, orig_w))
        y1 = max(0, min(y1, orig_h))
        x2 = max(0, min(x2, orig_w))
        y2 = max(0, min(y2, orig_h))

        label = COCO_LABELS[class_id] if class_id < len(COCO_LABELS) else "unknown"

        boxes_out.append({
            "label": label,
            "confidence": confidence,
            "box": {
                "xmin": float(x1),
                "ymin": float(y1),
                "xmax": float(x2),
                "ymax": float(y2),
            },
            # Store for NMS
            "_coords": [x1, y1, x2, y2],
            "_score": confidence,
        })

    # Non-Maximum Suppression — remove duplicate boxes for the same object
    # OpenCV's NMS expects boxes as [x, y, w, h]
    if not boxes_out:
        return []

    cv_boxes = [[b["box"]["xmin"], b["box"]["ymin"],
                 b["box"]["xmax"] - b["box"]["xmin"],
                 b["box"]["ymax"] - b["box"]["ymin"]] for b in boxes_out]
    scores = [b["_score"] for b in boxes_out]

    indices = cv2.dnn.NMSBoxes(cv_boxes, scores, CONFIDENCE_THRESHOLD, 0.45)
    result = []
    for i in indices:
        b = boxes_out[i]
        result.append({
            "label": b["label"],
            "confidence": b["confidence"],
            "box": b["box"],
        })

    return result


@app.get("/")
def root():
    return {"message": "YOLOv8 Object Detection API is running"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Preprocess
    input_tensor, orig_w, orig_h, pad_x, pad_y, scale = preprocess(image)

    # Run inference
    outputs = session.run(None, {input_name: input_tensor})

    # Postprocess
    detections = postprocess(outputs, orig_w, orig_h, pad_x, pad_y, scale)

    return {"detections": detections}