from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image, ExifTags
import os

# Initialize Flask App
app = Flask(__name__)

# Corrected Folder Paths
UPLOAD_FOLDER = 'uploads'
REAL_SAMPLE_FOLDER = r"C:\Users\lenovo\Desktop\background.programes\2025\real_samples"  # Corrected path

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
                    return True  # Photoshop or similar app detected
    except Exception:
        pass  # Ignore errors when reading metadata

    return False  # No editing app detected

def compare_with_references(uploaded_img):
    """ Compares the uploaded image with both French and Arabic reference images and selects the best match. """
    best_match = None
    best_score = 0

    for lang, ref_path in real_sample_paths.items():
        if os.path.exists(ref_path):
            reference_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
            if uploaded_img.shape != reference_img.shape:
                reference_img = cv2.resize(reference_img, (uploaded_img.shape[1], uploaded_img.shape[0]))
            score, _ = ssim(uploaded_img, reference_img, full=True)
            if score > best_score:
                best_match = lang
                best_score = score

    return best_match, best_score

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Fraud Detection</title>
    </head>
    <body>
        <h2>Upload Screenshot for Fraud Detection</h2>
        <form id="fraudDetectionForm" enctype="multipart/form-data">
            <input type="file" id="fileInput" name="file" required>
            <button type="submit">Verify</button>
        </form>
        <div id="result"></div>

        <script>
            document.getElementById("fraudDetectionForm").onsubmit = async function(event) {
                event.preventDefault();
                let formData = new FormData();
                let fileInput = document.getElementById("fileInput").files[0];

                if (!fileInput) {
                    document.getElementById("result").innerHTML = "<p>Please select a file.</p>";
                    return;
                }

                formData.append("file", fileInput);
                let response = await fetch("/upload", { method: "POST", body: formData });
                let result = await response.json();
                
                document.getElementById("result").innerHTML = `
                    <p>Classification: <b style="color:${result.color};">${result.classification}</b></p>
                    <p>SSIM Score: ${result.ssim_score}</p>
                    ${result.fake_reason ? `<p>Reason: ${result.fake_reason.join("<br>")}</p>` : ""}
                `;
            };
        </script>
    </body>
    </html>
    '''

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        # Load uploaded image
        uploaded_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if uploaded_img is None:
            return jsonify({
                'classification': "Not a Payment Screenshot ❌",
                'color': 'black',
                'fake_reason': ["Uploaded image is not a valid receipt."]
            })

        # Compare with reference images
        best_match, best_ssim = compare_with_references(uploaded_img)

        if not best_match:
            return jsonify({
                'classification': "Not a Payment Screenshot ❌",
                'color': 'black',
                'fake_reason': ["Uploaded image does not match any known receipt format."]
            })

        # Detect Photoshop or similar editing
        is_edited = detect_photoshop_edit(file_path)
        classification = "Real ✅"
        color = "green"
        fake_reason = []
        show_probability = False
        probability = 0

        # Classification Logic
        if is_edited:
            classification = "Fake 🔴"
            color = "red"
            fake_reason.append("Metadata shows the image was edited in Photoshop or similar software.")
        elif best_ssim >= SSIM_THRESHOLD_REAL:
            classification = "Real ✅"
            color = "green"
        elif best_ssim >= SSIM_THRESHOLD_FAKE:
            classification = "Probably Fake ⚠️"
            color = "orange"
            show_probability = True
            probability = round((1 - best_ssim) * 100, 2)  # Convert SSIM into fake probability
            fake_reason.append(f"Image appears modified. Probability of being fake: {probability}%")
        else:
            classification = "Fake 🔴"
            color = "red"
            fake_reason.append(f"Very low similarity score: {best_ssim:.4f}. Image does not match reference.")

        return jsonify({
            'classification': classification,
            'color': color,
            'fake_reason': fake_reason if classification != "Real ✅" else [],
            'show_probability': show_probability,
            'probability': probability if show_probability else None
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)  # Run on all networks
