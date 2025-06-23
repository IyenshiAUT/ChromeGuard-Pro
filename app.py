import json
import base64
import io
import os
from flask import Flask, request, jsonify, send_file, render_template, Response
from PIL import Image
import numpy as np
import cv2

# Import the core logic from other files
from watermarker import Watermarker
from utils import Crypto, Config

app = Flask(__name__, template_folder='templates')
watermarker = Watermarker()

# --- Flask API Endpoints ---
@app.route('/')
def home():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/embed', methods=['POST'])
def embed_endpoint():
    """API endpoint to embed a watermark."""
    if 'image' not in request.files or 'text' not in request.form:
        return jsonify({'error': "Missing image or text"}), 400
    try:
        # Get the original filename from the form data
        original_filename = request.form.get('filename', 'image.png')

        watermarked_bytes, keys = watermarker.embed(
            request.files['image'].read(), 
            request.form['text']
        )
        watermarked_base64 = base64.b64encode(watermarked_bytes).decode('utf-8')
        
        # Return the original filename along with the other data
        return jsonify({
            'image': watermarked_base64, 
            'keys': keys,
            'original_filename': original_filename 
        })
    except Exception as e:
        app.logger.error(f"Embedding failed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/extract', methods=['POST'])
def extract_endpoint():
    """API endpoint to extract a watermark."""
    if 'image' not in request.files or 'keys' not in request.form:
        return jsonify({'error': "Missing image or keys"}), 400
    try:
        extracted_text = watermarker.extract(
            request.files['image'].read(), 
            json.loads(request.form['keys'])
        )
        return jsonify({'text': extracted_text})
    except Exception as e:
        app.logger.error(f"Extraction failed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500
        
@app.route('/attack', methods=['POST'])
def attack_endpoint():
    if 'image' not in request.files: return jsonify({'error': "Missing image for attack"}), 400
    try:
        img_bytes = request.files['image'].read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        cv_img = np.array(pil_img)[:, :, ::-1].copy() # RGB to BGR
        
        attack_type = request.form.get('attack_type')
        params = json.loads(request.form.get('params', '{}'))
        attacked_img = None

        if attack_type == 'jpeg':
            quality = int(params.get('quality', 50))
            _, buffer = cv2.imencode('.jpg', cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            attacked_img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        elif attack_type == 'gauss_noise':
            std_dev = int(params.get('std_dev', 10))
            noise = np.random.normal(0, std_dev, cv_img.shape)
            attacked_img = np.clip(cv_img + noise, 0, 255).astype(np.uint8)
        elif attack_type == 'salt_pepper_noise':
            amount = float(params.get('amount', 0.05))
            attacked_img = cv_img.copy()
            num_salt = np.ceil(amount * cv_img.size * 0.5); coords = [np.random.randint(0, i - 1, int(num_salt)) for i in cv_img.shape]; attacked_img[coords[0], coords[1], :] = 255
            num_pepper = np.ceil(amount* cv_img.size * 0.5); coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in cv_img.shape]; attacked_img[coords[0], coords[1], :] = 0
        elif attack_type == 'blur':
            ksize = int(params.get('ksize', 5))
            attacked_img = cv2.blur(cv_img, (ksize, ksize))
        elif attack_type == 'resize':
            factor = float(params.get('factor', 0.5))
            h, w, _ = cv_img.shape
            small = cv2.resize(cv_img, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_AREA)
            attacked_img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        elif attack_type == 'rotate':
            angle = float(params.get('angle', 5))
            h, w, _ = cv_img.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            attacked_img = cv2.warpAffine(cv_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        if attacked_img is None: return jsonify({'error': 'Unknown attack type'}), 400
        
        _, buffer = cv2.imencode('.png', attacked_img)
        return send_file(io.BytesIO(buffer.tobytes()), mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Attack failed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)