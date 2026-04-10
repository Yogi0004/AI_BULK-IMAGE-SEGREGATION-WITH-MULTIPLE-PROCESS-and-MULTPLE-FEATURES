"""
Streamlit Web Application
Bulk Image Classification with Auto-Segregation
"""

import streamlit as st
import os
import shutil
from predict import predict_image
import pandas as pd
from datetime import datetime
import time

# Configuration
BASE_DIR = r'C:\Users\Homes247\Desktop\Bulk_image'
OUTPUT = os.path.join(BASE_DIR, 'output')
FOLDERS = ['floorplan', 'masterplan', 'gallery', 'rejected']
THRESHOLD = 0.6

# Create output folders
for f in FOLDERS:
    os.makedirs(os.path.join(OUTPUT, f), exist_ok=True)

# Page config
st.set_page_config(
    page_title="Bulk Image Classifier",
    page_icon="🏗️",
    layout="wide"
)

class BulkClassifier:
    def __init__(self):
        self.results = []
    
    def process(self, uploaded_files):
        """Process uploaded images"""
        if not uploaded_files:
            return None, None, None
        
        self.results = []
        total = len(uploaded_files)
        
        print("\n" + "="*60)
        print(f"🔄 Processing {total} images")
        print("="*60)
        
        start_time = time.time()
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process each file
        for idx, uploaded_file in enumerate(uploaded_files):
            try:
                # Update progress
                progress = (idx + 1) / total
                progress_bar.progress(progress)
                status_text.text(f"Processing {idx+1}/{total}: {uploaded_file.name}")
                
                # Save temp file
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Predict
                label, conf = predict_image(temp_path)
                
                # Determine folder
                if conf < THRESHOLD:
                    save_dir = 'rejected'
                else:
                    save_dir = label
                
                # Save image
                filename = uploaded_file.name
                dst = os.path.join(OUTPUT, save_dir, filename)
                
                # Handle duplicates
                base = os.path.splitext(filename)[0]
                ext = os.path.splitext(filename)[1]
                counter = 1
                
                while os.path.exists(dst):
                    dst = os.path.join(OUTPUT, save_dir, f"{base}_{counter}{ext}")
                    counter += 1
                
                shutil.copy(temp_path, dst)
                
                # Clean up temp
                os.remove(temp_path)
                
                # Store result
                self.results.append({
                    'Filename': filename,
                    'Category': save_dir,
                    'Confidence': f"{conf*100:.2f}%",
                    'Saved To': dst
                })
                
                # Log
                if (idx+1) % 10 == 0 or (idx+1) == total:
                    print(f"   ✅ {idx+1}/{total} - {filename} → {save_dir}")
                
            except Exception as e:
                print(f"   ❌ Error: {uploaded_file.name} - {e}")
                self.results.append({
                    'Filename': uploaded_file.name,
                    'Category': 'error',
                    'Confidence': '0%',
                    'Error': str(e)
                })
                
                # Clean up temp if exists
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        elapsed = time.time() - start_time
        
        # Clear progress
        progress_bar.empty()
        status_text.empty()
        
        print(f"\n✅ Completed in {elapsed:.1f}s")
        print("="*60 + "\n")
        
        # Generate outputs
        summary = self.generate_summary(total, elapsed)
        df = pd.DataFrame(self.results)
        stats = self.generate_stats()
        
        return summary, df, stats
    
    def generate_summary(self, total, elapsed):
        """Generate summary"""
        counts = {
            'floorplan': 0,
            'masterplan': 0,
            'gallery': 0,
            'rejected': 0,
            'error': 0
        }
        
        for r in self.results:
            cat = r['Category']
            counts[cat] = counts.get(cat, 0) + 1
        
        return {
            'total': total,
            'elapsed': elapsed,
            'counts': counts
        }
    
    def generate_stats(self):
        """Quick stats"""
        counts = {}
        for r in self.results:
            cat = r['Category']
            counts[cat] = counts.get(cat, 0) + 1
        
        return counts

# Initialize session state
if 'classifier' not in st.session_state:
    st.session_state.classifier = BulkClassifier()

# Header
st.title("🏗️ Bulk Architectural Image Classifier")
st.markdown("**Upload many images → AI classifies → Auto-saves to folders**")
st.markdown("Categories: **Floorplan** | **Masterplan** | **Gallery**")

st.divider()

# Settings sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.write(f"**Threshold:** {THRESHOLD*100}%")
    st.write(f"**Output:** `{OUTPUT}`")
    
    st.divider()
    
    st.header("📝 Instructions")
    st.markdown("""
    1. Upload images below
    2. Click "🚀 Classify & Save"
    3. Wait for processing
    4. Check results and output folders!
    
    **Supported Formats:**  
    JPG, PNG, BMP, TIFF
    
    **Processing Speed:**  
    ~3-5 images/sec on CPU
    """)

# Main content
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📤 Upload Images")
    
    uploaded_files = st.file_uploader(
        "Choose images",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        accept_multiple_files=True,
        help="Select multiple images to classify"
    )
    
    if uploaded_files:
        st.info(f"📊 {len(uploaded_files)} images uploaded")
    
    classify_btn = st.button(
        "🚀 Classify & Save",
        type="primary",
        use_container_width=True
    )

with col2:
    st.header("📊 Results")
    results_container = st.container()

# Process images when button clicked
if classify_btn and uploaded_files:
    with st.spinner("Processing images..."):
        summary, df, stats = st.session_state.classifier.process(uploaded_files)
    
    if summary:
        # Display summary
        with results_container:
            st.success("✅ Classification Complete!")
            
            # Metrics
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("Total Images", summary['total'])
                st.metric("Processing Time", f"{summary['elapsed']:.1f}s")
            
            with col_b:
                st.metric("Floorplan", summary['counts']['floorplan'])
                st.metric("Masterplan", summary['counts']['masterplan'])
            
            with col_c:
                st.metric("Gallery", summary['counts']['gallery'])
                st.metric("Rejected", summary['counts']['rejected'])
            
            # Speed
            speed = summary['total'] / summary['elapsed'] if summary['elapsed'] > 0 else 0
            st.write(f"⚡ **Speed:** {speed:.1f} images/sec")
            
            st.divider()
            
            # Output folders
            st.subheader("📁 Output Folders")
            st.code(f"""
{OUTPUT}\\
├── floorplan\\    ({summary['counts']['floorplan']} images)
├── masterplan\\   ({summary['counts']['masterplan']} images)
├── gallery\\      ({summary['counts']['gallery']} images)
└── rejected\\     ({summary['counts']['rejected']} images)
            """)
        
        # Quick stats in sidebar
        with st.sidebar:
            st.divider()
            st.header("📊 Quick Stats")
            for cat in ['floorplan', 'masterplan', 'gallery', 'rejected']:
                count = stats.get(cat, 0)
                st.write(f"• **{cat.title()}:** {count}")
        
        # Detailed results table
        st.divider()
        st.header("📋 Detailed Results")
        st.dataframe(df, use_container_width=True, height=400)
        
        # Download results
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"classification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

elif classify_btn and not uploaded_files:
    st.error("❌ Please upload images first!")

elif not classify_btn:
    with results_container:
        st.info("📤 Upload images and click 'Classify & Save' to begin")

# How it works section
st.divider()

with st.expander("ℹ️ How It Works"):
    st.markdown(f"""
    ## Classification Process
    
    **For each image:**
    
    1. ✅ Load & preprocess (resize to 224×224)
    2. 🤖 AI prediction (MobileNetV2)
    3. ✓ Confidence check (threshold: {THRESHOLD*100}%)
    4. 💾 Auto-save to correct folder
    
    ---
    
    ## Output Structure
    
    ```
    {OUTPUT}\\
    ├── floorplan\\      # Floor plans
    ├── masterplan\\     # Master plans
    ├── gallery\\        # Gallery images
    └── rejected\\       # Low confidence
    ```
    
    ---
    
    ## Example
    
    **Upload 90 images:**
    
    - ✅ 30 → floorplan/
    - ✅ 30 → masterplan/
    - ✅ 28 → gallery/
    - ⚠️ 2 → rejected/
    
    All done automatically!
    
    ---
    
    **Confidence Threshold:** {THRESHOLD*100}%
    
    Images with confidence below this threshold are saved to the "rejected" folder for manual review.
    """)

# Footer
st.divider()
st.markdown("**Built with TensorFlow & Streamlit** | **CPU Optimized** | **Production Ready**")