"""
ChromaGuard Pro - Advanced Image Watermarking Flask Application

This module provides a web-based interface for embedding and extracting invisible watermarks
from images using DWT-SVD (Discrete Wavelet Transform - Singular Value Decomposition) technique.
The application also includes attack simulation capabilities to test watermark robustness.
"""

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

# Initialize Flask application with template folder
app = Flask(__name__, template_folder='templates')

# Create a global watermarker instance for processing
watermarker = Watermarker()

# --- Flask API Endpoints ---

@app.route('/')
def home():
    """
    Serves the main HTML page for the ChromaGuard Pro interface.
    
    Returns:
        str: Rendered HTML template for the web interface
    """
    return render_template('index.html')

@app.route('/embed', methods=['POST'])
def embed_endpoint():
    """
    API endpoint to embed a watermark into an image.
    
    This endpoint accepts an image file and text, then embeds the text as an invisible
    watermark using DWT-SVD technique. The watermark is encrypted using AES encryption
    for security.
    
    Expected form data:
        - image: Image file to watermark
        - text: Secret text to embed
        - filename: Original filename (optional)
    
    Returns:
        JSON response containing:
            - image: Base64 encoded watermarked image
            - keys: Decryption keys needed for extraction
            - original_filename: Original filename for download
    
    Raises:
        400: If required form data is missing
        500: If embedding process fails
    """
    # Validate required form data
    if 'image' not in request.files or 'text' not in request.form:
        return jsonify({'error': "Missing image or text"}), 400
    
    try:
        # Get the original filename from the form data for download purposes
        original_filename = request.form.get('filename', 'image.png')

        # Embed watermark using the watermarker
        watermarked_bytes, keys = watermarker.embed(
            request.files['image'].read(), 
            request.form['text']
        )
        
        # Convert watermarked image to base64 for JSON response
        watermarked_base64 = base64.b64encode(watermarked_bytes).decode('utf-8')
        
        # Return the original filename along with the other data
        return jsonify({
            'image': watermarked_base64, 
            'keys': keys,
            'original_filename': original_filename 
        })
    except Exception as e:
        # Log error and return error response
        app.logger.error(f"Embedding failed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/extract', methods=['POST'])
def extract_endpoint():
    """
    API endpoint to extract a watermark from an image.
    
    This endpoint accepts a watermarked image and the corresponding decryption keys,
    then extracts and decrypts the embedded text.
    
    Expected form data:
        - image: Watermarked image file
        - keys: JSON string containing decryption keys
    
    Returns:
        JSON response containing:
            - text: Extracted and decrypted secret text
    
    Raises:
        400: If required form data is missing
        500: If extraction process fails
    """
    # Validate required form data
    if 'image' not in request.files or 'keys' not in request.form:
        return jsonify({'error': "Missing image or keys"}), 400
    
    try:
        # Extract watermark using the watermarker
        extracted_text = watermarker.extract(
            request.files['image'].read(), 
            json.loads(request.form['keys'])
        )
        return jsonify({'text': extracted_text})
    except Exception as e:
        # Log error and return error response
        app.logger.error(f"Extraction failed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500
        
@app.route('/attack', methods=['POST'])
def attack_endpoint():
    """
    API endpoint to simulate various attacks on watermarked images.
    
    This endpoint applies different types of image processing attacks to test
    the robustness of the watermarking technique. Supported attacks include:
    - JPEG compression
    - Gaussian noise
    - Salt & pepper noise
    - Blur
    - Resize
    - Rotation
    
    Expected form data:
        - image: Image file to attack
        - attack_type: Type of attack to apply
        - params: JSON string containing attack parameters
    
    Returns:
        PNG image file: Attacked image
    
    Raises:
        400: If required form data is missing or attack type is unknown
        500: If attack process fails
    """
    # Validate required form data
    if 'image' not in request.files: 
        return jsonify({'error': "Missing image for attack"}), 400
    
    try:
        # Read and convert image to OpenCV format
        img_bytes = request.files['image'].read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        cv_img = np.array(pil_img)[:, :, ::-1].copy()  # RGB to BGR conversion
        
        # Get attack parameters
        attack_type = request.form.get('attack_type')
        params = json.loads(request.form.get('params', '{}'))
        attacked_img = None

        # Apply different types of attacks based on attack_type
        if attack_type == 'jpeg':
            # JPEG compression attack
            quality = int(params.get('quality', 50))
            _, buffer = cv2.imencode('.jpg', cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            attacked_img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            
        elif attack_type == 'gauss_noise':
            # Gaussian noise attack
            std_dev = int(params.get('std_dev', 10))
            noise = np.random.normal(0, std_dev, cv_img.shape)
            attacked_img = np.clip(cv_img + noise, 0, 255).astype(np.uint8)
            
        elif attack_type == 'salt_pepper_noise':
            # Salt & pepper noise attack
            amount = float(params.get('amount', 0.05))
            attacked_img = cv_img.copy()
            
            # Add salt noise (white pixels)
            num_salt = np.ceil(amount * cv_img.size * 0.5)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in cv_img.shape]
            attacked_img[coords[0], coords[1], :] = 255
            
            # Add pepper noise (black pixels)
            num_pepper = np.ceil(amount * cv_img.size * 0.5)
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in cv_img.shape]
            attacked_img[coords[0], coords[1], :] = 0
            
        elif attack_type == 'blur':
            # Blur attack using average filter
            ksize = int(params.get('ksize', 5))
            attacked_img = cv2.blur(cv_img, (ksize, ksize))
            
        elif attack_type == 'resize':
            # Resize attack (downscale then upscale)
            factor = float(params.get('factor', 0.5))
            h, w, _ = cv_img.shape
            small = cv2.resize(cv_img, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_AREA)
            attacked_img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
            
        elif attack_type == 'rotate':
            # Rotation attack
            angle = float(params.get('angle', 5))
            h, w, _ = cv_img.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            attacked_img = cv2.warpAffine(cv_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        # Validate that an attack was applied
        if attacked_img is None: 
            return jsonify({'error': 'Unknown attack type'}), 400
        
        # Encode and return the attacked image
        _, buffer = cv2.imencode('.png', attacked_img)
        return send_file(io.BytesIO(buffer.tobytes()), mimetype='image/png')
        
    except Exception as e:
        # Log error and return error response
        app.logger.error(f"Attack failed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

# Main entry point for the Flask application
if __name__ == '__main__':
    # Run the Flask app in debug mode on port 5000
    app.run(debug=True, port=5000)