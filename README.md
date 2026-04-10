# 🏠 Homes247 Premium Image Processing

## 📌 Overview

Homes247 Premium Image Processing is an AI-powered Streamlit application designed to automate and optimize real estate image workflows. It intelligently processes images through a structured pipeline ensuring high-quality, standardized, and production-ready outputs.

---

## 🚀 Key Features

### 🤖 AI-Based Image Classification

* Automatically classifies images into:

  * Floor Plan
  * Master Plan
  * Gallery
  * Rejected
* Uses TensorFlow deep learning model

### 🧹 OCR-Based Text & Logo Removal

* Detects unwanted text, watermarks, and logos
* Uses EasyOCR + OpenCV
* Smart edge detection and masking

### 🌟 Super Resolution (AI Upscaling)

* Powered by Real-ESRGAN
* Enhances image clarity and resolution
* Supports multiple upscale modes (×2, ×4)

### 📐 Smart Resizing

* Auto-resizes based on category:

  * Floor Plan → 1500×1500
  * Master Plan → 1640×860
  * Gallery → 820×430

### 🏷️ Watermark Integration

* Adds branding logo automatically
* Adjustable opacity and size

### 📊 Quality Detection (Blur Analysis)

* Uses Laplacian Variance
* Size-normalized (500×500)
* Classifies images into:

  * Good Quality
  * Bad Quality

### ⚡ Batch Processing

* Process multiple images simultaneously
* Generates reports and statistics

### ☁️ Cloud Upload (R2 Storage)

* Automatically uploads good-quality images
* Structured storage by category

---

## 🔄 Processing Pipeline

The system follows a strict and optimized workflow:

```
1. Image Upload
2. AI Classification
3. Quality Check (Blur Detection)
4. Text & Logo Removal (OCR + Inpainting)
5. Super Resolution (Optional AI Upscale)
6. Smart Resize (Category Based)
7. Watermark Application
8. Final Output Save
9. Cloud Upload (Optional)
10. Report Generation (CSV + JSON + ZIP)
```

---

## 📁 Output Structure

```
api_output/
 ├── floorplan/
 │   ├── good_quality/
 │   └── bad_quality/
 ├── masterplan/
 ├── gallery/
 ├── rejected/
```

---

## 📊 Reports Generated

* CSV Report
* JSON Report
* Session Summary
* ZIP Download Package

Includes:

* Category distribution
* Quality stats
* Confidence scores
* Processing logs

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **AI Models:** TensorFlow, PyTorch (Real-ESRGAN)
* **Image Processing:** OpenCV, PIL
* **OCR:** EasyOCR
* **Cloud:** Cloudflare R2 (S3 Compatible)

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📌 Configuration

* Update model path for classifier.h5
* Configure watermark_logo.png
* Set output format (WEBP / JPEG / PNG)
* Adjust quality threshold (recommended: 500)

---

## 📈 Performance Highlights

* Fully automated pipeline
* High-speed batch processing
* AI-driven enhancement
* Production-ready outputs

---

## 👨‍💻 Developed By

**Midddi Yogananda Reddy**
📧 Email: [yogireddymiddi2004@gmail.com](mailto:yogireddymiddi2004@gmail.com)

---

## ⭐ Final Note

This system is built for real estate platforms to automate image preprocessing at scale, ensuring consistent quality, branding, and optimization with minimal manual effort.

---

✨ *Smart. Automated. Scalable.*
