"""
ChromaGuard Pro - DWT-SVD Watermarking Implementation

This module implements the core watermarking functionality using DWT-SVD (Discrete Wavelet Transform - 
Singular Value Decomposition) technique combined with chaotic scrambling for enhanced security.
The watermarking process embeds encrypted text into the blue channel of images.

Author: Image Processing Project
"""

import cv2
import numpy as np
import pywt
import os
import base64
from utils import Crypto, Config

class Watermarker:
    """
    Main watermarking class that handles embedding and extraction of invisible watermarks
    using DWT-SVD technique with chaotic scrambling.
    """
    
    def _text_to_binary_image(self, text, size):
        """
        Converts text to a binary image representation using OpenCV text rendering.
        
        This method creates a black image and renders the text in white, creating
        a binary pattern that can be embedded as a watermark.
        
        Args:
            text (str): Text to convert to binary image
            size (tuple): Size of the output image (height, width)
            
        Returns:
            numpy.ndarray: Binary image with text rendered in white on black background
        """
        # Create a black image of specified size
        img = np.zeros(size, dtype=np.uint8)
        
        # Configure font parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7 if size[0] > 20 else 0.4  # Adjust scale based on image size
        thickness = 1
        
        # Get text dimensions for centering
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Calculate center position for text
        x = max(0, (size[1] - text_w) // 2)
        y = max(text_h, (size[0] + text_h) // 2)
        
        # Render text in white on the black background
        cv2.putText(img, text, (x, y), font, font_scale, 255, thickness, cv2.LINE_AA)
        return img
        
    def _bytes_to_cv2_image(self, image_bytes, mode=cv2.IMREAD_COLOR):
        """
        Converts image bytes to OpenCV image format.
        
        Args:
            image_bytes (bytes): Raw image data
            mode: OpenCV image reading mode (default: cv2.IMREAD_COLOR)
            
        Returns:
            numpy.ndarray: OpenCV image array
            
        Raises:
            ValueError: If image cannot be decoded
        """
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode image using OpenCV
        img = cv2.imdecode(nparr, mode)
        if img is None: 
            raise ValueError("Could not decode image.")
        return img
        
    def chaotic_scramble(self, image, key, unscramble=False):
        """
        Applies chaotic scrambling to an image using logistic map.
        
        This method uses the logistic map (x_{n+1} = 3.99 * x_n * (1 - x_n)) to generate
        a chaotic sequence that determines the pixel permutation order. This adds an
        additional layer of security to the watermarking process.
        
        Args:
            image (numpy.ndarray): Input image to scramble/unscramble
            key (float): Initial value for the logistic map (chaotic key)
            unscramble (bool): If True, unscrambles the image; if False, scrambles it
            
        Returns:
            numpy.ndarray: Scrambled or unscrambled image
        """
        h, w = image.shape
        
        # Generate chaotic sequence using logistic map
        sequence = np.zeros(h * w, dtype=np.float64)
        sequence[0] = key
        
        # Iterate logistic map to generate chaotic sequence
        for i in range(h * w - 1): 
            sequence[i + 1] = 3.99 * sequence[i] * (1 - sequence[i])
        
        # Create permutation array based on sorted sequence
        p = np.argsort(sequence)
        
        if unscramble:
            # For unscrambling, create inverse permutation
            p_inv = np.empty_like(p)
            p_inv[p] = np.arange(len(p))
            p = p_inv
            
        # Apply permutation to flattened image and reshape
        return image.flatten()[p].reshape(h, w)

    def embed(self, cover_image_bytes, watermark_text):
        """
        Embeds encrypted text as an invisible watermark into an image.
        
        The embedding process follows these steps:
        1. Encrypt the watermark text using AES encryption
        2. Convert encrypted text to binary image
        3. Apply chaotic scrambling to the watermark
        4. Perform DWT on the blue channel of the cover image
        5. Apply SVD to both cover and watermark images
        6. Modify singular values of cover image with watermark
        7. Reconstruct the watermarked image
        
        Args:
            cover_image_bytes (bytes): Raw bytes of the cover image
            watermark_text (str): Text to embed as watermark
            
        Returns:
            tuple: (watermarked_image_bytes, keys_dict)
                - watermarked_image_bytes: Bytes of the watermarked image
                - keys_dict: Dictionary containing all keys needed for extraction
        """
        # Generate AES encryption keys
        aes_key = os.urandom(Config.AES_KEY_SIZE)
        aes_iv = os.urandom(Config.AES_IV_SIZE)
        
        # Encrypt the watermark text
        encrypted_text_bytes = Crypto.encrypt(watermark_text.encode('utf-8'), aes_key, aes_iv)
        encrypted_text_b64 = base64.b64encode(encrypted_text_bytes).decode('utf-8')
        
        # Convert cover image bytes to OpenCV format
        cover_image = self._bytes_to_cv2_image(cover_image_bytes)
        
        # Resize image if it exceeds maximum dimension for processing
        processing_image = cover_image
        if cover_image.shape[0] > Config.MAX_DIMENSION or cover_image.shape[1] > Config.MAX_DIMENSION:
            r = Config.MAX_DIMENSION / float(max(cover_image.shape[:2]))
            dim = (int(cover_image.shape[1] * r), int(cover_image.shape[0] * r))
            processing_image = cv2.resize(cover_image, dim, interpolation=cv2.INTER_AREA)

        # Split image into color channels and work with blue channel
        b_channel, g_channel, r_channel = cv2.split(processing_image)
        h, w = b_channel.shape
        
        # Perform 2D Discrete Wavelet Transform on blue channel
        coeffs_cover = pywt.dwt2(b_channel, 'haar')
        LL_cover, (LH, HL, HH) = coeffs_cover  # LL = approximation coefficients
        
        # Convert encrypted text to binary watermark image
        watermark_image = self._text_to_binary_image(encrypted_text_b64[:20], (LL_cover.shape[0], LL_cover.shape[1]))
        
        # Apply chaotic scrambling to the watermark
        scrambled_wm = self.chaotic_scramble(watermark_image, Config.CHAOTIC_KEY)
        
        # Perform SVD on both cover and watermark images
        U_c, s_c, V_t_c = np.linalg.svd(LL_cover, full_matrices=False)
        U_wm, s_wm, V_t_wm = np.linalg.svd(scrambled_wm.astype(np.float32), full_matrices=False)
        
        # Modify singular values of cover image with watermark
        min_len = min(len(s_c), len(s_wm))
        s_watermarked = s_c[:min_len] + Config.ALPHA_DWT_SVD * s_wm[:min_len]
        
        # Reconstruct modified LL coefficients
        modified_LL = U_c[:, :min_len] @ np.diag(s_watermarked) @ V_t_c[:min_len, :]
        
        # Perform inverse DWT to reconstruct blue channel
        reconstructed_b = pywt.idwt2((modified_LL, (LH, HL, HH)), 'haar')
        
        # Clip values to valid range and convert to uint8
        final_b = np.uint8(np.clip(reconstructed_b, 0, 255))
        final_b = final_b[:h, :w]  # Ensure correct dimensions
        
        # Merge channels back together
        watermarked_processed = cv2.merge((final_b, g_channel, r_channel))
        
        # Resize back to original dimensions
        watermarked_image = cv2.resize(watermarked_processed, (cover_image.shape[1], cover_image.shape[0]), interpolation=cv2.INTER_CUBIC)

        # Encode final image to bytes
        _, final_image_bytes = cv2.imencode('.png', watermarked_image)

        # Prepare keys dictionary for extraction
        keys = {
            'aes_key': base64.b64encode(aes_key).decode('utf-8'),
            'aes_iv': base64.b64encode(aes_iv).decode('utf-8'),
            'encrypted_text_b64': encrypted_text_b64,
            'original_LL': LL_cover.tolist(),
            'U_scrambled_wm': U_wm.tolist(),
            'V_t_scrambled_wm': V_t_wm.tolist(),
            'chaotic_key': Config.CHAOTIC_KEY,
            'processed_shape': processing_image.shape[:2]
        }
        return final_image_bytes.tobytes(), keys

    def extract(self, watermarked_image_bytes, keys):
        """
        Extracts and decrypts the embedded watermark from an image.
        
        The extraction process reverses the embedding process:
        1. Resize image to processing dimensions
        2. Perform DWT on blue channel
        3. Compare singular values to detect watermark
        4. Decrypt the extracted text
        
        Args:
            watermarked_image_bytes (bytes): Raw bytes of the watermarked image
            keys (dict): Dictionary containing all keys needed for extraction
            
        Returns:
            str: Extracted and decrypted watermark text
            
        Raises:
            ValueError: If watermark is not found or image is unaltered
        """
        # Convert watermarked image bytes to OpenCV format
        watermarked_image = self._bytes_to_cv2_image(watermarked_image_bytes)
        
        # Resize to processing dimensions
        proc_h, proc_w = keys['processed_shape']
        processing_image = cv2.resize(watermarked_image, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        
        # Extract blue channel
        b_channel_wm, _, _ = cv2.split(processing_image)
        
        # Get original LL coefficients from keys
        original_LL = np.array(keys['original_LL'])
        
        # Perform DWT on watermarked blue channel
        coeffs_wm = pywt.dwt2(b_channel_wm, 'haar')
        LL_wm, _ = coeffs_wm
        
        # Ensure dimensions match original
        h_orig, w_orig = original_LL.shape
        LL_wm = LL_wm[:h_orig, :w_orig]

        # Perform SVD on both original and watermarked LL coefficients
        _, s_c, _ = np.linalg.svd(original_LL)
        _, s_wm_ext, _ = np.linalg.svd(LL_wm)
        
        min_len = min(len(s_c), len(s_wm_ext))
        
        # Check if watermark is present by comparing singular values
        if np.allclose(s_c[:min_len], s_wm_ext[:min_len]):
             raise ValueError("Watermark not found or image is unaltered.")

        # Decrypt the embedded text
        aes_key = base64.b64decode(keys['aes_key'])
        aes_iv = base64.b64decode(keys['aes_iv'])
        encrypted_text_bytes = base64.b64decode(keys['encrypted_text_b64'])
        decrypted_text = Crypto.decrypt(encrypted_text_bytes, aes_key, aes_iv)
        
        return decrypted_text.decode('utf-8')
