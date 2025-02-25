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
            uploaded_gray = cv2.cvtColor(uploaded_img, cv2.COLOR_BGR2GRAY)

            # Resize the uploaded image to match the reference size
            reference_resized = cv2.resize(reference_img, (uploaded_gray.shape[1], uploaded_gray.shape[0]))

            # Compute SSIM
            score, _ = ssim(reference_resized, uploaded_gray, full=True)
            
            if score > best_score:
                best_score = score
                best_match = lang

    return best_match, best_score

@app.route('/upload', methods=['POST'])
def upload_file():
    """ Handles image uploads and fraud detection. """
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save uploaded image
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Read image
    uploaded_img = cv2.imread(file_path)
    if uploaded_img is None:
        return jsonify({"error": "Invalid image file"}), 400

    # Check for Photoshop edits
    photoshop_detected = detect_photoshop_edit(file_path)

    # Compare with reference samples
    best_match, ssim_score = compare_with_references(uploaded_img)

    # Classification based on SSIM score
    if ssim_score >= SSIM_THRESHOLD_REAL:
        classification = "Real"
    elif ssim_score < SSIM_NOT_RECEIPT:
        classification = "Not a Receipt"
    elif ssim_score < SSIM_THRESHOLD_FAKE:
        classification = "Fake"
    else:
        classification = "Probably Fake"

    # Construct response
    response = {
        "file": file.filename,
        "classification": classification,
        "similarity_score": round(ssim_score, 4),
        "reference_matched": best_match,
        "photoshop_detected": photoshop_detected
    }

    return jsonify(response), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
