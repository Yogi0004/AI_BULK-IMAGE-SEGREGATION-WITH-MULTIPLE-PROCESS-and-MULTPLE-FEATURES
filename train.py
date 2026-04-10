"""
Step 2: Train Image Classification Model
CPU-optimized MobileNetV2 for architectural image classification
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
from PIL import Image

# Configuration
BASE_DIR = r'C:\Users\Homes247\Desktop\Bulk_image'
TRAIN_DIR = os.path.join(BASE_DIR, 'dataset', 'train')
VAL_DIR = os.path.join(BASE_DIR, 'dataset', 'val')
MODEL_PATH = os.path.join(BASE_DIR, 'classifier.h5')

IMG_SIZE = 224
BATCH = 20
EPOCHS = 30


def check_dataset():
    """Verify dataset exists"""
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ Training data not found: {TRAIN_DIR}")
        print("📋 Run: python 1_preprocess.py first")
        return False
    
    train_count = sum([len(os.listdir(os.path.join(TRAIN_DIR, d))) 
                      for d in os.listdir(TRAIN_DIR)])
    val_count = sum([len(os.listdir(os.path.join(VAL_DIR, d))) 
                    for d in os.listdir(VAL_DIR)])
    
    print(f"✅ Dataset found:")
    print(f"   Training: {train_count} images")
    print(f"   Validation: {val_count} images")
    
    if train_count == 0:
        print("❌ No training images!")
        return False
    
    return True

def train():
    """Train the model"""
    print("\n" + "="*60)
    print("🚀 TRAINING MODEL")
    print("="*60)
    
    # Check dataset
    if not check_dataset():
        return
    
    print("\n📦 Loading data...")
    
    train_gen = ImageDataGenerator(rescale=1./255)
    val_gen = ImageDataGenerator(rescale=1./255)
    
    train_data = train_gen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH,
        class_mode='categorical'
    )
    
    val_data = val_gen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH,
        class_mode='categorical'
    )
    
    print(f"✅ Data loaded")
    print(f"   Classes: {list(train_data.class_indices.keys())}")
    
    # Build model
    print("\n🏗️  Building model...")
    
    base = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base.trainable = False
    
    x = base.output
    x = GlobalAveragePooling2D()(x)
    out = Dense(3, activation='softmax')(x)
    
    model = Model(inputs=base.input, outputs=out)
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Model created")
    print(f"   Parameters: {model.count_params():,}")
    
    # Train
    print(f"\n🎓 Training for {EPOCHS} epochs...\n")
    
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        verbose=1
    )
    
    # Save
    model.save(MODEL_PATH)
    
    # Results
    final_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"\n📈 Results:")
    print(f"   Training Accuracy: {final_acc*100:.2f}%")
    print(f"   Validation Accuracy: {final_val_acc*100:.2f}%")
    print(f"\n💾 Model saved: {MODEL_PATH}")
    print("\n📋 Next: python 3_app.py")
    print("="*60)

if __name__ == "__main__":
    train()