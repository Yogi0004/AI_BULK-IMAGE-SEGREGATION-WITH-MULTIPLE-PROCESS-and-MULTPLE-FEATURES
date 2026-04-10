"""
Prediction Module
Handles image classification
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Configuration
BASE_DIR = r'C:\Users\Homes247\Desktop\Bulk_image'
MODEL_PATH = os.path.join(BASE_DIR, 'classifier.h5')

# Classes (alphabetical - matches training order)
CLASSES = ['floorplan', 'gallery', 'masterplan']

# Load model
print("Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded")

def predict_image(img_path):
    """
    Predict class for single image
    
    Args:
        img_path: Path to image file
    
    Returns:
        (predicted_class, confidence)
    """
    try:
        # Load image
        img = Image.open(img_path)
        
        # Convert to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize
        img = img.resize((224, 224))
        
        # To array and normalize
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, 0)
        
        # Predict
        preds = model.predict(arr, verbose=0)[0]
        
        # Get result
        idx = np.argmax(preds)
        label = CLASSES[idx]
        confidence = float(preds[idx])
        
        return label, confidence
        
    except Exception as e:
        print(f"Error: {img_path} - {e}")
        return 'error', 0.0