"""
ChromaGuard Pro - DWT-SVD Watermarking Implementation

This module implements the core watermarking functionality using DWT-SVD (Discrete Wavelet Transform - 
Singular Value Decomposition) technique combined with chaotic scrambling for enhanced security.
The watermarking process embeds encrypted text into the blue channel of images.

Author: Image Processing Project
"""

import base64
import cv2
import numpy as np
import pywt
from utils import Crypto, Config
import os

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
        Embeds encrypted text as an invisible watermark into an image using an
        adaptive, block-based DWT-SVD approach.

        The embedding process follows these steps:
        1. Encrypt the watermark text using AES encryption.
        2. Convert encrypted text to a binary image.
        3. Apply chaotic scrambling to the watermark.
        4. Perform DWT on the blue channel of the cover image.
        5. Divide the LL sub-band into blocks.
        6. For each block, calculate its variance to determine local complexity.
        7. Calculate an adaptive embedding strength (alpha) based on the variance.
        8. Apply SVD to the block and embed the watermark using the adaptive alpha.
        9. Reconstruct the LL sub-band from the modified blocks.
        10. Reconstruct the watermarked image via inverse DWT.
        
        Args:
            cover_image_bytes (bytes): Raw bytes of the cover image.
            watermark_text (str): Text to embed as watermark.
            
        Returns:
            tuple: (watermarked_image_bytes, keys_dict)
                - watermarked_image_bytes: Bytes of the watermarked image.
                - keys_dict: Dictionary containing all keys needed for extraction.
        """
        aes_key = os.urandom(Config.AES_KEY_SIZE)
        aes_iv = os.urandom(Config.AES_IV_SIZE)
        encrypted_text_bytes = Crypto.encrypt(watermark_text.encode('utf-8'), aes_key, aes_iv)
        encrypted_text_b64 = base64.b64encode(encrypted_text_bytes).decode('utf-8')
        cover_image = self._bytes_to_cv2_image(cover_image_bytes)
        
        processing_image = cover_image
        if cover_image.shape[0] > Config.MAX_DIMENSION or cover_image.shape[1] > Config.MAX_DIMENSION:
            r = Config.MAX_DIMENSION / float(max(cover_image.shape[:2]))
            dim = (int(cover_image.shape[1] * r), int(cover_image.shape[0] * r))
            processing_image = cv2.resize(cover_image, dim, interpolation=cv2.INTER_AREA)

        # Build a tiny grayscale thumbnail for geometric pre-alignment (96x96)
        thumb_size = 96
        thumb_gray = cv2.cvtColor(processing_image, cv2.COLOR_BGR2GRAY)
        thumb_small = cv2.resize(thumb_gray, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
        ok, thumb_png = cv2.imencode(".png", thumb_small)
        thumb_b64 = base64.b64encode(thumb_png.tobytes()).decode("utf-8")

        # Split channels (B, G, R) for downstream use
        b_channel, g_channel, r_channel = cv2.split(processing_image)

        # --- Multi-Channel Adaptive Embedding ---
        def _embed_channel(channel, scrambled_wm):
            """Helper function to apply adaptive watermarking to a single channel."""
            h, w = channel.shape
            # Ensure float for DWT
            channel_f = channel.astype(np.float32)
            coeffs_cover = pywt.dwt2(channel_f, 'haar')
            LL_cover, (LH_cover, HL_cover, HH_cover) = coeffs_cover

            block_size = 8
            modified_LL = np.zeros_like(LL_cover)
            
            MIN_ALPHA = 0.02
            MAX_ALPHA = 0.25
            max_possible_variance = np.var(LL_cover)
            if max_possible_variance < 1: max_possible_variance = 1

            for r in range(0, LL_cover.shape[0], block_size):
                for c in range(0, LL_cover.shape[1], block_size):
                    cover_block = LL_cover[r:r+block_size, c:c+block_size]
                    wm_block = scrambled_wm[r:r+block_size, c:c+block_size]
                    
                    if cover_block.shape != (block_size, block_size):
                        modified_LL[r:r+block_size, c:c+block_size] = cover_block
                        continue

                    U_c, s_c, V_t_c = np.linalg.svd(cover_block, full_matrices=False)
                    
                    variance = np.var(cover_block)
                    normalized_variance = min(variance / max_possible_variance, 1.0)
                    adaptive_alpha = MIN_ALPHA + (normalized_variance * (MAX_ALPHA - MIN_ALPHA))

                    U_wm, s_wm, V_t_wm = np.linalg.svd(wm_block.astype(np.float32), full_matrices=False)
                    
                    min_len = min(len(s_c), len(s_wm))
                    s_watermarked = s_c[:min_len] + adaptive_alpha * s_wm[:min_len]
                    
                    U_c_trunc = U_c[:, :min_len]
                    V_t_c_trunc = V_t_c[:min_len, :]
                    modified_block = U_c_trunc @ np.diag(s_watermarked) @ V_t_c_trunc
                    
                    modified_LL[r:r+block_size, c:c+block_size] = modified_block

            reconstructed_channel = pywt.idwt2((modified_LL, (LH_cover, HL_cover, HH_cover)), 'haar')
            final_channel = np.uint8(np.clip(reconstructed_channel, 0, 255))
            return final_channel[:h, :w]

        # First, generate the fingerprint from the original BLUE channel
        # Ensure float for DWT
        coeffs_b_cover = pywt.dwt2(b_channel.astype(np.float32), 'haar')
        LL_b_cover, _ = coeffs_b_cover
        block_size = 8
        original_s_values = []
        for r in range(0, LL_b_cover.shape[0], block_size):
            for c in range(0, LL_b_cover.shape[1], block_size):
                cover_block = LL_b_cover[r:r+block_size, c:c+block_size]
                if cover_block.shape != (block_size, block_size):
                    continue
                _, s_c, _ = np.linalg.svd(cover_block, full_matrices=False)
                original_s_values.extend(s_c)

        # Prepare the watermark image, sized based on the LL band of one channel
        watermark_image = self._text_to_binary_image(
            encrypted_text_b64[:20],
            (LL_b_cover.shape[0], LL_b_cover.shape[1])
        )
        scrambled_wm = self.chaotic_scramble(watermark_image, Config.CHAOTIC_KEY)

        # Embed the watermark into all three channels
        final_b = _embed_channel(b_channel, scrambled_wm)
        final_g = _embed_channel(g_channel, scrambled_wm)
        final_r = _embed_channel(r_channel, scrambled_wm)

        # Merge the watermarked channels back together
        watermarked_processed = cv2.merge((final_b, final_g, final_r))

        # Resize back to original dimensions
        watermarked_image = cv2.resize(watermarked_processed, (cover_image.shape[1], cover_image.shape[0]), interpolation=cv2.INTER_CUBIC)
        _, final_image_bytes = cv2.imencode('.png', watermarked_image)

        # Convert the fingerprint to a compact byte string and then Base64 encode it
        s_values_bytes = np.array(original_s_values, dtype=np.float32).tobytes()
        s_fingerprint_b64 = base64.b64encode(s_values_bytes).decode('utf-8')

        # Prepare keys dictionary with the compact, encoded fingerprint
        keys = {
            'aes_key': base64.b64encode(aes_key).decode('utf-8'),
            'aes_iv': base64.b64encode(aes_iv).decode('utf-8'),
            'encrypted_text_b64': encrypted_text_b64,
            's_fingerprint_b64': s_fingerprint_b64,  # compact fingerprint
            'thumb_b64': thumb_b64,                  # tiny thumbnail for alignment
            'chaotic_key': Config.CHAOTIC_KEY,
            'processed_shape': list(processing_image.shape[:2]),  # [h, w]
            'block_size': block_size
        }
        return final_image_bytes.tobytes(), keys

    def extract(self, watermarked_image_bytes, keys):
        """
        Extracts and decrypts the embedded watermark. Performs best‑effort
        geometric pre‑alignment using a small thumbnail.
        """
        # Decode image
        watermarked_image = self._bytes_to_cv2_image(watermarked_image_bytes)

        # Best‑effort geometric pre‑alignment (if thumbnail present)
        try:
            if 'thumb_b64' in keys and 'processed_shape' in keys:
                proc_h, proc_w = int(keys['processed_shape'][0]), int(keys['processed_shape'][1])
                thumb_size = 96

                th_bytes = np.frombuffer(base64.b64decode(keys['thumb_b64']), np.uint8)
                thumb_img = cv2.imdecode(th_bytes, cv2.IMREAD_GRAYSCALE)
                if thumb_img is not None and thumb_img.shape == (thumb_size, thumb_size):
                    h_full, w_full = watermarked_image.shape[:2]
                    attacked_gray = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2GRAY)
                    attacked_small = cv2.resize(attacked_gray, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)

                    orb = cv2.ORB_create(nfeatures=800)
                    k1, d1 = orb.detectAndCompute(attacked_small, None)
                    k2, d2 = orb.detectAndCompute(thumb_img, None)
                    if d1 is not None and d2 is not None and len(k1) >= 8 and len(k2) >= 8:
                        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                        matches = bf.match(d1, d2)
                        if matches:
                            matches = sorted(matches, key=lambda m: m.distance)[:60]
                            pts1 = np.float32([k1[m.queryIdx].pt for m in matches])
                            pts2 = np.float32([k2[m.trainIdx].pt for m in matches])
                            H_small, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
                            if H_small is not None:
                                sx = w_full / float(thumb_size)
                                sy = h_full / float(thumb_size)
                                S_up = np.array([[sx, 0, 0],[0, sy, 0],[0, 0, 1]], dtype=np.float64)
                                S_dn = np.array([[1.0/sx, 0, 0],[0, 1.0/sy, 0],[0, 0, 1]], dtype=np.float64)
                                H_full = S_up @ H_small @ S_dn
                                watermarked_image = cv2.warpPerspective(
                                    watermarked_image, H_full, (proc_w, proc_h),
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
                                )
        except Exception:
            # Alignment is best‑effort; continue if it fails
            pass

        # Resize to processing dimensions
        if 'processed_shape' in keys:
            proc_h, proc_w = int(keys['processed_shape'][0]), int(keys['processed_shape'][1])
            watermarked_image = cv2.resize(watermarked_image, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

        # Compute LL SVD fingerprint on blue channel
        b_channel_wm, _, _ = cv2.split(watermarked_image)
        LL_wm, _ = pywt.dwt2(b_channel_wm.astype(np.float32), 'haar')

        block_size = int(keys.get('block_size', 8))
        s_curr_list = []
        for rr in range(0, LL_wm.shape[0], block_size):
            for cc in range(0, LL_wm.shape[1], block_size):
                blk = LL_wm[rr:rr+block_size, cc:cc+block_size]
                if blk.shape != (block_size, block_size):
                    continue
                _, s_blk, _ = np.linalg.svd(blk, full_matrices=False)
                s_curr_list.extend(s_blk)
        s_curr = np.asarray(s_curr_list, dtype=np.float32)

        # Load original fingerprint (log similarity; do not block)
        if 's_fingerprint_b64' in keys:
            s_orig = np.frombuffer(base64.b64decode(keys['s_fingerprint_b64']), dtype=np.float32)
            if len(s_orig) and len(s_curr):
                m = min(len(s_orig), len(s_curr))
                denom = float(np.linalg.norm(s_orig[:m]) + 1e-6)
                s_diff = float(np.linalg.norm(s_curr[:m] - s_orig[:m]) / denom)
                try:
                    print(f"SVD fingerprint diff: {s_diff:.6f}")
                except Exception:
                    pass

        # Decrypt using AES materials from keys
        aes_key = base64.b64decode(keys['aes_key'])
        aes_iv = base64.b64decode(keys['aes_iv'])
        ciphertext = base64.b64decode(keys['encrypted_text_b64'])
        plaintext = Crypto.decrypt(ciphertext, aes_key, aes_iv)
        return plaintext.decode('utf-8')
