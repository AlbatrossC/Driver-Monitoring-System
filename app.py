from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import logging
from functools import lru_cache
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global model variables
chaitanya_model = None
soham_model = None
cnn_model = None

def load_models():
    """Load all models at startup with error handling"""
    global chaitanya_model, soham_model, cnn_model
    
    try:
        logger.info("Starting model loading process...")
        
        start_time = time.time()
        chaitanya_model = YOLO('models/chaitanya/best.pt')
        logger.info(f"chaitanya YOLO model loaded in {time.time() - start_time:.2f}s")
        
        start_time = time.time()
        soham_model = YOLO('models/soham/best.pt')
        logger.info(f"Soham YOLO model loaded in {time.time() - start_time:.2f}s")
        
        start_time = time.time()
        cnn_model = load_model('models/mishra/driver_behavior_cnn.h5')
        logger.info(f"CNN model loaded in {time.time() - start_time:.2f}s")
        
        logger.info("All models loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

# Load models at startup
load_models()

# Class mappings
chaitanya_CLASSES = ['Cigarette', 'Drinking', 'Eating', 'Phone', 'Seatbelt']
SOHAM_CLASSES = ['Distracted', 'Drinking', 'Drowsy', 'Eating', 'PhoneUse', 'SafeDriving', 'Seatbelt', 'Smoking']
CNN_CLASSES = ['Smoking/Drinking/Yawning', 'safe_driving', 'talking_phone', 'texting_phone', 'turning']

# Unified YOLO class mapping
UNIFIED_MAPPING = {
    'Cigarette': 'Smoking',
    'Smoking': 'Smoking',
    'Drinking': 'Drinking',
    'Eating': 'Eating',
    'Phone': 'Phone Usage',
    'PhoneUse': 'Phone Usage',
    'Seatbelt': 'Seatbelt',
    'Distracted': 'Distracted',
    'Drowsy': 'Drowsy',
    'SafeDriving': 'Safe Driving'
}

# Safety instructions
SAFETY_INSTRUCTIONS = {
    'Seatbelt': '✓ Thank you for wearing your seatbelt. Stay safe!',
    'Drinking': '⚠ Avoid drinking while driving.',
    'Smoking': '⚠ Smoking while driving is unsafe and distracting.',
    'Phone Usage': '⚠ Do not use your phone while driving. Pull over if necessary.',
    'Drowsy': '⚠ You appear drowsy. Please take a break and rest.',
    'Eating': '⚠ Eating while driving can be distracting. Please focus on the road.',
    'Distracted': '⚠ Please focus on the road and avoid distractions.'
}

def generate_safety_instructions(detected_classes):
    """Generate safety instructions including checking for missing seatbelt"""
    instructions = {'positive': [], 'negative': []}
    
    # Check if seatbelt is detected
    has_seatbelt = 'Seatbelt' in detected_classes
    
    # If no seatbelt detected, add warning
    if not has_seatbelt:
        instructions['negative'].append('⚠ Please fasten your seatbelt for your safety.')
    
    # Add instructions for detected behaviors
    for unified_class in detected_classes:
        if unified_class in SAFETY_INSTRUCTIONS:
            instruction = SAFETY_INSTRUCTIONS[unified_class]
            if instruction.startswith('✓'):
                instructions['positive'].append(instruction)
            else:
                instructions['negative'].append(instruction)
    
    return instructions

@lru_cache(maxsize=128)
def unify_class_name(original_class):
    """Map original class names to unified names with caching"""
    return UNIFIED_MAPPING.get(original_class, original_class)

def iou(box1, box2):
    """Calculate Intersection over Union between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def non_max_suppression_per_class(boxes, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to remove overlapping boxes of the same class
    """
    if len(boxes) == 0:
        return []
    
    initial_count = len(boxes)
    
    # Sort boxes by confidence (descending)
    boxes = sorted(boxes, key=lambda x: x['conf'], reverse=True)
    
    keep = []
    
    while len(boxes) > 0:
        # Keep the box with highest confidence
        best_box = boxes[0]
        keep.append(best_box)
        boxes = boxes[1:]
        
        # Remove all boxes that overlap significantly with the best box
        filtered_boxes = []
        for box in boxes:
            if iou(best_box['bbox'], box['bbox']) < iou_threshold:
                filtered_boxes.append(box)
        boxes = filtered_boxes
    
    if initial_count > len(keep):
        logger.debug(f"NMS reduced {initial_count} boxes to {len(keep)} boxes")
    
    return keep

def confidence_based_voting(chaitanya_results, soham_results):
    """
    Apply confidence-based voting with proper NMS:
    - Combine all detections from both models per class
    - Apply Non-Maximum Suppression to remove overlapping duplicates
    - Keep only the highest confidence detection for overlapping boxes
    """
    IOU_THRESHOLD = 0.5
    
    # Parse chaitanya detections
    chaitanya_detections = {}
    for box in chaitanya_results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy()
        
        original_class = chaitanya_CLASSES[cls_id]
        unified_class = unify_class_name(original_class)
        
        if unified_class not in chaitanya_detections:
            chaitanya_detections[unified_class] = []
        chaitanya_detections[unified_class].append({
            'bbox': xyxy,
            'conf': conf,
            'source': 'chaitanya',
            'original_class': original_class
        })
    
    # Parse soham detections
    soham_detections = {}
    for box in soham_results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy()
        
        original_class = SOHAM_CLASSES[cls_id]
        unified_class = unify_class_name(original_class)
        
        # Skip SafeDriving at presentation layer
        if unified_class == 'Safe Driving':
            continue
        
        if unified_class not in soham_detections:
            soham_detections[unified_class] = []
        soham_detections[unified_class].append({
            'bbox': xyxy,
            'conf': conf,
            'source': 'soham',
            'original_class': original_class
        })
    
    # Combine and apply NMS per class
    final_detections = {}
    all_classes = set(chaitanya_detections.keys()) | set(soham_detections.keys())
    
    logger.debug(f"Processing {len(all_classes)} unique classes")
    
    for unified_class in all_classes:
        chaitanya_boxes = chaitanya_detections.get(unified_class, [])
        soham_boxes = soham_detections.get(unified_class, [])
        
        # Combine all boxes for this class
        all_boxes = chaitanya_boxes + soham_boxes
        
        logger.debug(f"{unified_class}: chaitanya={len(chaitanya_boxes)}, Soham={len(soham_boxes)}, Total={len(all_boxes)}")
        
        # Apply NMS to remove overlapping duplicates
        filtered_boxes = non_max_suppression_per_class(all_boxes, IOU_THRESHOLD)
        
        logger.debug(f"{unified_class}: After NMS={len(filtered_boxes)}")
        
        if filtered_boxes:
            final_detections[unified_class] = filtered_boxes
    
    return final_detections, chaitanya_detections, soham_detections

def draw_detections(image, detections):
    """Draw bounding boxes and labels on image"""
    img = image.copy()
    
    for unified_class, boxes in detections.items():
        for detection in boxes:
            bbox = detection['bbox']
            conf = detection['conf']
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label with confidence
            label = f"{unified_class}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Background rectangle for text
            cv2.rectangle(img, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            
            # Text
            cv2.putText(img, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return img

def predict_cnn(image_path):
    """Run CNN prediction with optimized preprocessing"""
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    
    predictions = cnn_model.predict(img, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])
    predicted_class = CNN_CLASSES[predicted_class_idx]
    
    return {
        'class': predicted_class,
        'confidence': confidence
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_images():
    """Process uploaded images with all models"""
    if 'images' not in request.files:
        logger.warning("No images in request")
        return jsonify({'error': 'No images uploaded'}), 400
    
    files = request.files.getlist('images')
    show_cnn = request.form.get('show_cnn', 'false') == 'true'
    
    logger.info(f"Processing {len(files)} images (show_cnn={show_cnn})")
    
    results = []
    
    for idx, file in enumerate(files):
        if file.filename == '':
            continue
        
        start_time = time.time()
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"Processing image {idx + 1}/{len(files)}: {filename}")
        
        # Load image
        image = cv2.imread(filepath)
        
        # Run YOLO models
        chaitanya_results = chaitanya_model(image, verbose=False)
        soham_results = soham_model(image, verbose=False)
        
        # Apply confidence-based voting
        final_detections, chaitanya_raw, soham_raw = confidence_based_voting(
            chaitanya_results, soham_results
        )
        
        # Draw detections
        annotated_image = draw_detections(image, final_detections)
        
        # Save annotated image
        annotated_filename = f"annotated_{filename}"
        annotated_filepath = os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename)
        cv2.imwrite(annotated_filepath, annotated_image)
        
        # Get detected classes and count
        detected_classes = list(final_detections.keys())
        detection_counts = {cls: len(boxes) for cls, boxes in final_detections.items()}
        
        # Generate safety instructions
        instructions = generate_safety_instructions(detected_classes)
        
        # CNN prediction
        cnn_result = predict_cnn(filepath)
        
        # Prepare detailed report data
        chaitanya_details = []
        for cls, boxes in chaitanya_raw.items():
            for box in boxes:
                chaitanya_details.append({
                    'class': cls,
                    'confidence': float(box['conf']),
                    'bbox': box['bbox'].tolist()
                })
        
        soham_details = []
        for cls, boxes in soham_raw.items():
            for box in boxes:
                soham_details.append({
                    'class': cls,
                    'confidence': float(box['conf']),
                    'bbox': box['bbox'].tolist()
                })
        
        result_data = {
            'original_image': f'/static/uploads/{filename}',
            'annotated_image': f'/static/uploads/{annotated_filename}',
            'detected_classes': detected_classes,
            'detection_counts': detection_counts,
            'instructions': instructions,
            'detailed_report': {
                'chaitanya_model': chaitanya_details,
                'soham_model': soham_details,
                'cnn_model': cnn_result
            }
        }
        
        # Only include CNN result in main view if toggle is enabled
        if show_cnn:
            result_data['cnn_result'] = cnn_result
        
        results.append(result_data)
        
        processing_time = time.time() - start_time
        logger.info(f"Image {idx + 1} processed in {processing_time:.2f}s - Detected: {detected_classes}")
    
    logger.info(f"All {len(files)} images processed successfully")
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)