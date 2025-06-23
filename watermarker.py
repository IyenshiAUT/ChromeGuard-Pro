import cv2
import numpy as np
import pywt
import os
import base64
from utils import Crypto, Config

class Watermarker:
    def _text_to_binary_image(self, text, size):
        img = np.zeros(size, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7 if size[0] > 20 else 0.4
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = max(0, (size[1] - text_w) // 2)
        y = max(text_h, (size[0] + text_h) // 2)
        cv2.putText(img, text, (x, y), font, font_scale, 255, thickness, cv2.LINE_AA)
        return img
        
    def _bytes_to_cv2_image(self, image_bytes, mode=cv2.IMREAD_COLOR):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, mode)
        if img is None: raise ValueError("Could not decode image.")
        return img
        
    def chaotic_scramble(self, image, key, unscramble=False):
        h, w = image.shape
        sequence = np.zeros(h * w, dtype=np.float64)
        sequence[0] = key
        for i in range(h * w - 1): sequence[i + 1] = 3.99 * sequence[i] * (1 - sequence[i])
        p = np.argsort(sequence)
        if unscramble:
            p_inv = np.empty_like(p)
            p_inv[p] = np.arange(len(p))
            p = p_inv
        return image.flatten()[p].reshape(h, w)

    def embed(self, cover_image_bytes, watermark_text):
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

        b_channel, g_channel, r_channel = cv2.split(processing_image)
        h, w = b_channel.shape
        
        coeffs_cover = pywt.dwt2(b_channel, 'haar')
        LL_cover, (LH, HL, HH) = coeffs_cover
        
        watermark_image = self._text_to_binary_image(encrypted_text_b64[:20], (LL_cover.shape[0], LL_cover.shape[1]))
        scrambled_wm = self.chaotic_scramble(watermark_image, Config.CHAOTIC_KEY)
        
        U_c, s_c, V_t_c = np.linalg.svd(LL_cover, full_matrices=False)
        U_wm, s_wm, V_t_wm = np.linalg.svd(scrambled_wm.astype(np.float32), full_matrices=False)
        
        min_len = min(len(s_c), len(s_wm))
        s_watermarked = s_c[:min_len] + Config.ALPHA_DWT_SVD * s_wm[:min_len]
        
        modified_LL = U_c[:, :min_len] @ np.diag(s_watermarked) @ V_t_c[:min_len, :]
        reconstructed_b = pywt.idwt2((modified_LL, (LH, HL, HH)), 'haar')
        
        final_b = np.uint8(np.clip(reconstructed_b, 0, 255))
        final_b = final_b[:h, :w]
        
        watermarked_processed = cv2.merge((final_b, g_channel, r_channel))
        watermarked_image = cv2.resize(watermarked_processed, (cover_image.shape[1], cover_image.shape[0]), interpolation=cv2.INTER_CUBIC)

        _, final_image_bytes = cv2.imencode('.png', watermarked_image)

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
        watermarked_image = self._bytes_to_cv2_image(watermarked_image_bytes)
        
        proc_h, proc_w = keys['processed_shape']
        processing_image = cv2.resize(watermarked_image, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        
        b_channel_wm, _, _ = cv2.split(processing_image)
        
        original_LL = np.array(keys['original_LL'])
        
        coeffs_wm = pywt.dwt2(b_channel_wm, 'haar')
        LL_wm, _ = coeffs_wm
        
        h_orig, w_orig = original_LL.shape
        LL_wm = LL_wm[:h_orig, :w_orig]

        _, s_c, _ = np.linalg.svd(original_LL)
        _, s_wm_ext, _ = np.linalg.svd(LL_wm)
        
        min_len = min(len(s_c), len(s_wm_ext))
        
        if np.allclose(s_c[:min_len], s_wm_ext[:min_len]):
             raise ValueError("Watermark not found or image is unaltered.")

        aes_key = base64.b64decode(keys['aes_key'])
        aes_iv = base64.b64decode(keys['aes_iv'])
        encrypted_text_bytes = base64.b64decode(keys['encrypted_text_b64'])
        decrypted_text = Crypto.decrypt(encrypted_text_bytes, aes_key, aes_iv)
        
        return decrypted_text.decode('utf-8')
