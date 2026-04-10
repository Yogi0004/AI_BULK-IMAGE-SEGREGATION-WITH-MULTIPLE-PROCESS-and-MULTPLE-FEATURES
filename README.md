🏠 Homes247 Premium Image Processing System

An advanced AI-powered Streamlit application for automating real estate image processing workflows including classification, enhancement, text removal, resizing, watermarking, and quality analysis.

🚀 Features
🔍 AI-Based Classification
Automatically classifies images into:
Floor Plan
Master Plan
Gallery
Rejected
🧹 OCR Text & Logo Removal
Uses EasyOCR + OpenCV
Removes:
Watermarks
Logos
Edge text
Legends (plans)
🌟 Super Resolution (AI Upscaling)
Powered by Real-ESRGAN
Supports:
×2, ×4 upscale
Tile-based processing for large images
📐 Smart Resize Engine
Auto resize based on category:
Floor Plan → 1500×1500
Master Plan → 1640×860
Gallery → 820×430
🏷️ Watermark Integration
Center-based watermark
Adjustable opacity & size
Auto compression after watermark
📊 Quality Detection (Blur-Based)
Uses Laplacian Variance
Normalized to 500×500
Classifies:
Good Quality
Bad Quality
⚡ Batch Processing
Process multiple images at once
Full pipeline automation
📦 Export & Reports
CSV + JSON reports
ZIP download with categorized images
Session logs & analytics
☁️ Cloud Upload (R2 Integration)
Upload only Good Quality images

Structured storage:

category/filename
🔄 Processing Pipeline
Text Removal → Super Resolution → Resize → Watermark → Save

✔ Fully automated
✔ Error-handled
✔ Scalable

🧠 Technologies Used
Frontend: Streamlit
AI/ML:
TensorFlow (Image Classification)
PyTorch (Real-ESRGAN, MobileNet)
Computer Vision:
OpenCV
EasyOCR
Image Processing:
PIL (Pillow)
Cloud:
Cloudflare R2 (S3-compatible)
📂 Project Structure
├── app.py
├── weights/
├── api_output/
│   ├── floorplan/
│   ├── masterplan/
│   ├── gallery/
│   └── rejected/
├── session_reports/
├── upload_history.json
├── api_processing_statistics.json
└── watermark_logo.png
⚙️ Installation
1️⃣ Clone the Repository
git clone <your-repo-link>
cd project-folder
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run Application
streamlit run app.py
📌 Configuration
Key Settings:
Output Format: WEBP / JPEG / PNG
Confidence Threshold
Blur Threshold (default: 500)
SR Model Selection
Batch Processing Toggle
📊 Quality Logic
Uses Laplacian Variance
Example:
Sharp Image → ~2800 ✅
Blurry Image → ~200 ❌
📁 Output Naming Format
CATEGORY-YYYYMMDD-HHMMSS.webp

Example:

GALLERY-20260410-153045.webp
📈 Reports Included
Processing Summary
Category-wise Distribution
Quality Metrics
Confidence Scores
Text Removal %
🔐 Error Handling
Safe fallback for:
OCR failure
SR failure
Image read issues
Automatic retries for:
File naming conflicts
Model loading
🌐 API & Server Integration
Sends results to external server
Supports:
API Key authentication
JSON payload
☁️ Cloud Upload Logic
Only uploads Good Quality images
Skips invalid files
Retry-safe mechanism
🎯 Use Cases
Real Estate Portals
Property Listing Automation
Bulk Image Optimization
Image Cleaning & Standardization
👨‍💻 Developed By

Midddi Yogananda Reddy
📧 Email: yogireddymiddi2004@gmail.com
