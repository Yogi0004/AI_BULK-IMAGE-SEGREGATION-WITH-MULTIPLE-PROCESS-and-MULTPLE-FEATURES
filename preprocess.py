"""
Step 1: Data Preprocessing and Validation
Organizes images into train/val splits for model training
"""

import os
import shutil
from pathlib import Path
from PIL import Image
import random

# Your dataset path
BASE_DIR = r'C:\Users\Homes247\Desktop\Bulk_image'
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
RAW_DIR = os.path.join(BASE_DIR, 'raw_images')

TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VAL_DIR = os.path.join(DATASET_DIR, 'val')

# Configuration
CATEGORIES = ['floorplan', 'masterplan', 'gallery']
TRAIN_SPLIT = 0.85  # 85% training, 15% validation
MIN_IMAGES = 10

class Preprocessor:
    def __init__(self):
        self.stats = {
            'total': 0,
            'valid': 0,
            'invalid': 0,
            'categories': {}
        }
    
    def validate_image(self, img_path):
        """Validate if image is usable"""
        try:
            with Image.open(img_path) as img:
                img.verify()
            
            with Image.open(img_path) as img:
                w, h = img.size
                if w < 32 or h < 32:
                    return False
            return True
        except:
            return False
    
    def process_category(self, category):
        """Process images for one category"""
        raw_cat = os.path.join(RAW_DIR, category)
        
        if not os.path.exists(raw_cat):
            print(f"⚠️  {category} folder not found at {raw_cat}")
            return
        
        print(f"\n📂 Processing {category}...")
        
        # Get image files
        valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        files = [f for f in os.listdir(raw_cat) 
                if Path(f).suffix.lower() in valid_ext]
        
        print(f"   Found {len(files)} files")
        
        # Validate
        valid_imgs = []
        for fname in files:
            fpath = os.path.join(raw_cat, fname)
            self.stats['total'] += 1
            
            if self.validate_image(fpath):
                valid_imgs.append(fpath)
                self.stats['valid'] += 1
            else:
                self.stats['invalid'] += 1
                print(f"   ⚠️  Invalid: {fname}")
        
        if len(valid_imgs) < MIN_IMAGES:
            print(f"   ❌ Need at least {MIN_IMAGES} images, found {len(valid_imgs)}")
            return
        
        print(f"   ✅ Valid: {len(valid_imgs)}")
        
        # Split train/val
        random.shuffle(valid_imgs)
        split_idx = int(len(valid_imgs) * TRAIN_SPLIT)
        
        train_imgs = valid_imgs[:split_idx]
        val_imgs = valid_imgs[split_idx:]
        
        # Create directories
        train_cat = os.path.join(TRAIN_DIR, category)
        val_cat = os.path.join(VAL_DIR, category)
        os.makedirs(train_cat, exist_ok=True)
        os.makedirs(val_cat, exist_ok=True)
        
        # Copy training images
        for idx, img_path in enumerate(train_imgs):
            ext = Path(img_path).suffix
            dst = os.path.join(train_cat, f"{category}_{idx:04d}{ext}")
            shutil.copy2(img_path, dst)
        
        # Copy validation images
        for idx, img_path in enumerate(val_imgs):
            ext = Path(img_path).suffix
            dst = os.path.join(val_cat, f"{category}_val_{idx:04d}{ext}")
            shutil.copy2(img_path, dst)
        
        # Stats
        self.stats['categories'][category] = {
            'total': len(valid_imgs),
            'train': len(train_imgs),
            'val': len(val_imgs)
        }
        
        print(f"   📊 Train: {len(train_imgs)}, Val: {len(val_imgs)}")
    
    def run(self):
        """Execute preprocessing"""
        print("="*60)
        print("🚀 DATA PREPROCESSING")
        print("="*60)
        
        # Check raw_images exists
        if not os.path.exists(RAW_DIR):
            print(f"\n❌ Raw images folder not found: {RAW_DIR}")
            print("\n📋 Please create:")
            print("raw_images/")
            print("├── floorplan/")
            print("├── masterplan/")
            print("└── gallery/")
            print("\nAdd your images and run again.")
            
            # Create structure
            os.makedirs(RAW_DIR, exist_ok=True)
            for cat in CATEGORIES:
                os.makedirs(os.path.join(RAW_DIR, cat), exist_ok=True)
            print(f"\n✅ Created folder structure")
            return False
        
        # Clean existing dataset
        if os.path.exists(DATASET_DIR):
            shutil.rmtree(DATASET_DIR)
        
        # Process
        for category in CATEGORIES:
            self.process_category(category)
        
        # Summary
        print("\n" + "="*60)
        print("📊 SUMMARY")
        print("="*60)
        print(f"\nTotal: {self.stats['total']}")
        print(f"✅ Valid: {self.stats['valid']}")
        print(f"❌ Invalid: {self.stats['invalid']}")
        
        if self.stats['categories']:
            print("\n📚 Distribution:")
            for cat, counts in self.stats['categories'].items():
                print(f"\n{cat.upper()}:")
                print(f"  Total: {counts['total']}")
                print(f"  Train: {counts['train']}")
                print(f"  Val: {counts['val']}")
            
            print("\n✅ Preprocessing complete!")
            print(f"📂 Dataset: {DATASET_DIR}")
            print("\n📋 Next: python 2_train.py")
        else:
            print("\n⚠️  No valid images found!")
        
        print("="*60)
        return True


def check_existing_dataset():
    """Check if dataset already exists"""
    if os.path.exists(TRAIN_DIR) and os.path.exists(VAL_DIR):
        train_count = 0
        val_count = 0
        
        for cat in CATEGORIES:
            train_cat = os.path.join(TRAIN_DIR, cat)
            val_cat = os.path.join(VAL_DIR, cat)
            
            if os.path.exists(train_cat):
                train_count += len(os.listdir(train_cat))
            if os.path.exists(val_cat):
                val_count += len(os.listdir(val_cat))
        
        if train_count > 0 or val_count > 0:
            print(f"✅ Existing dataset found:")
            print(f"   Train: {train_count} images")
            print(f"   Val: {val_count} images")
            
            response = input("\nUse existing dataset? (y/n): ")
            if response.lower() == 'y':
                print("\n✅ Using existing dataset")
                print("📋 Next: python 2_train.py")
                return True
    
    return False


def main():
    print("\n📸 Data Preprocessor\n")
    
    # Check existing
    if check_existing_dataset():
        return
    
    # Check raw images
    if os.path.exists(RAW_DIR):
        total = 0
        for cat in CATEGORIES:
            cat_dir = os.path.join(RAW_DIR, cat)
            if os.path.exists(cat_dir):
                count = len([f for f in os.listdir(cat_dir) 
                           if Path(f).suffix.lower() in {'.jpg', '.jpeg', '.png'}])
                print(f"{cat}: {count} images")
                total += count
        
        if total == 0:
            print("\n⚠️  No images found. Add images to raw_images/ folders.\n")
            return
        
        print(f"\nTotal: {total} images\n")
        
        response = input("▶️  Start preprocessing? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    
    # Run
    random.seed(42)
    preprocessor = Preprocessor()
    preprocessor.run()


if __name__ == "__main__":
    main()