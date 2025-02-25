from flask import Flask, request, jsonify
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image, ExifTags
import os
from flask_cors import CORS  # Allow cross-origin requests

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Folder Paths
UPLOAD_FOLDER = 'uploads'
REAL_SAMPLE_FOLDER = 'real_samples'

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REAL_SAMPLE_FOLDER, exist_ok=True)

# Define reference samples for Al Barid Bank French & Arabic receipts
SAMPLE_CATEGORIES = {
    "french": "real_sample_french.png",
    "arabic": "real_sample_arabic.png"
}

# Full paths for real sample images
real_sample_paths = {key: os.path.join(REAL_SAMPLE_FOLDER, filename) for key, filename in SAMPLE_CATEGORIES.items()}

# SSIM Thresholds
SSIM_THRESHOLD_REAL = 0.7800  # Above this = Real
SSIM_THRESHOLD_FAKE = 0.65  # Below this = Fake
SSIM_NOT_RECEIPT = 0.40  # Below this = Not a receipt

def detect_photoshop_edit(image_path):
    """ Checks if an image contains Photoshop or similar editing software metadata. """
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if exif_data:
            for tag, value in exif_data.items():
                tag_name = ExifTags.TAGS.get(tag, tag)
                if tag_name == "Software" and any(app in str(value) for app in ["Photoshop", "GIMP", "Pixelmator", "Snapseed"]):
                    return True  # Photoshop detected
    except Exception:
        pass
    return False  # No editing software detected

def compare_with_references(uploaded_img):
    """ Compares uploaded image with French & Arabic reference images to select best match. """
    best_match = None
    best_score = 0

    for lang, ref_path in real_sample_paths.items():
        if os.path.exists(ref_path):
            reference_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
            if uploaded_img
