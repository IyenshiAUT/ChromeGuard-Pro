# ChromaGuard Pro: Secure & Robust Invisible Image Watermarking

---

### **1. The Problem: Protecting Digital Images**

*   **Ownership & Copyright**: How can we prove ownership of a digital image once it's shared online?
*   **Imperceptibility**: The proof of ownership (watermark) must be completely invisible to the naked eye so it doesn't spoil the image.
*   **Robustness**: The watermark must survive common image manipulations, both accidental and malicious. This includes:
    *   JPEG Compression (saving the image again)
    *   Adding noise
    *   Blurring or resizing
    *   Rotation
*   **Security**: The watermark should be impossible for an unauthorized person to read or remove.

---

### **2. The Solution: ChromaGuard Pro**

A comprehensive web application that provides a powerful and user-friendly solution for invisible watermarking.

*   **Hybrid Algorithm**: It combines the **Discrete Wavelet Transform (DWT)** and **Singular Value Decomposition (SVD)** for maximum robustness and imperceptibility.
*   **High Security**: It uses **AES Encryption** and **Chaotic Scrambling** to protect the watermark content.
*   **User-Friendly Interface**: A simple web UI allows anyone to embed, extract, and test watermarks without deep technical knowledge.
*   **Robustness Testing**: Includes a built-in "Attack Simulation" module to instantly test how well the watermark survives various manipulations.

---

### **3. Technical Implementation**

The application is built on a modern Python backend and a clean JavaScript frontend.

*   **Backend**:
    *   **Framework**: Flask (run on a high-performance Uvicorn ASGI server).
    *   **Core Libraries**:
        *   `OpenCV` & `Pillow`: For all image manipulation tasks.
        *   `PyWavelets`: To perform the Discrete Wavelet Transform.
        *   `NumPy`: For high-performance numerical operations on image matrices.

*   **Frontend**:
    *   HTML5, TailwindCSS, and modern Vanilla JavaScript.

*   **The DWT-SVD Embedding Process**:
    1.  The cover image is split into color channels, and the **Blue channel** is selected for watermarking because:
        * The Human Visual System (HVS) is least sensitive to changes in blue wavelengths
        * Blue channel modifications are harder for attackers to detect visually
        * Studies show blue channel can tolerate more data embedding while maintaining imperceptibility
        * Higher robustness against common image processing operations in the blue spectrum
    2.  A **2D-DWT** is applied, decomposing the channel into four frequency sub-bands: `LL`, `LH`, `HL`, and `HH`.
    3.  The watermark is embedded in the `LL` (Approximation) sub-band, as it's the most visually significant and resilient to change.
    4.  **SVD** is applied to the `LL` sub-band. The singular values (which represent image intensity) are modified by adding the watermark data to them.
    5.  An **Inverse DWT** is performed to reconstruct the blue channel, which now invisibly contains the watermark.
    6.  The channels are merged back into the final, watermarked image.

*   **The DWT-SVD Extraction Process**:
     1.  The (possibly attacked) image is split into color channels, and the **Blue channel** is selected for extraction.
     2.  A **2D-DWT** is applied to decompose the blue channel into sub-bands, focusing on the `LL` band.
     3.  **SVD** is performed on the `LL` sub-band to obtain the singular values.
     4.  The original SVD fingerprint (from the embedding step, stored in the keys) is compared with the extracted singular values to recover the embedded watermark data.
     5.  The watermark data is descrambled using the chaotic map and then decrypted using AES (with the key and IV from the keys JSON).
     6.  The original secret text is revealed if extraction is successful.

---

### **4. Advanced Features for Security & Robustness**

ChromaGuard Pro goes beyond a basic algorithm by adding multiple layers of security and robustness.

*   AES Encryption (confidentiality)
        *   Before embedding, the watermark text is encrypted using AES‑CBC with PKCS7 padding.
        *   Even if raw bits are recovered, the message remains unreadable without the key + IV.

*   Chaotic Scrambling (obfuscation)
        *   The binary watermark image is scrambled via a logistic map permutation keyed by a secret seed.
        *   Prevents straightforward pattern recognition or template attacks.

*   Adaptive Block‑Wise DWT‑SVD (imperceptibility + resilience)
        *   We embed in the LL band using 8×8 blocks. Each block’s variance controls a local embedding strength α:
            *   Smooth areas → small α (invisible), textured edges → larger α (robust).
        *   This improves visual quality and survivability under compression/noise.

*   Multi‑Channel Embedding (B, G, R) with a Single Fingerprint
        *   The watermark is applied to all three channels to diversify redundancy.
        *   Keys stay compact by storing a single SVD fingerprint (from the blue channel’s LL) shared across channels.

*   Compact Keys via SVD Fingerprint
        *   We store only the vector of singular values from the cover image’s LL blocks as a float32 byte array (Base64‑encoded).
        *   This drastically reduces key size compared to full‑matrix storage.

*   Best‑Effort Geometric Pre‑Alignment
        *   A tiny 96×96 grayscale thumbnail of the processing image is included in the keys.
        *   During extraction, ORB + BFMatcher + RANSAC estimate a homography to approximately undo mild rotation/scale.

*   Multi‑Band Embedding (Experimental)
        *   We validated embedding across all four sub‑bands (LL, LH, HL, HH) for extreme robustness.
        *   Outcome: Key size grew significantly for a web workflow, so we ship LL‑focused adaptive embedding by default.

---

### **5. Complete Procedure (End‑to‑End Demo)**

Follow these steps to reproduce the full workflow inside the web app.

1) Embed the watermark
     * Open the app in your browser.
     * In the “Embed” panel:
         - Select a cover image (PNG/JPG). Large images are automatically resized for performance.
         - Enter the secret text to embed.
         - Click “Embed”.
     * Output:
         - A watermarked image preview + a download link (filename starts with “ChromaGuard_…”).
         - A Keys JSON blob in the textarea. Keep this safe — you’ll need it for extraction.

2) (Optional) Attack the watermarked image
     * In the “Attack Simulation” panel, pick an attack type and parameters:
         - JPEG: quality 1–100
         - Gaussian noise: standard deviation
         - Salt & pepper: amount
         - Blur: kernel size (odd)
         - Resize: scale factor
         - Rotate: ± angle in degrees
     * Click “Apply Attack”. The attacked image preview appears and is cached in the page for extraction.

3) Send to Extract
     * Click “Send to Extract”.
         - This pre‑loads the attacked image blob (if present) for the Extract form and auto‑fills keys when possible.
         - You can also manually choose a file instead of using the attacked image.

4) Extract the watermark
     * In the “Extract” panel:
         - Ensure the image field is either the attacked blob (auto) or a manually chosen file.
         - Ensure the Keys textarea contains the JSON captured in the Embed step (paste if needed).
         - Click “Extract”.
     * Result: The decrypted secret text appears below if the extraction succeeds.

5) What the Keys contain (summary)
     * aes_key, aes_iv: Base64‑encoded materials for AES‑CBC decryption.
     * encrypted_text_b64: The ciphertext of your message (Base64).
     * s_fingerprint_b64: Compact SVD fingerprint (float32 Base64) from LL blocks.
     * thumb_b64: 96×96 grayscale thumbnail for geometric pre‑alignment.
     * processed_shape: [height, width] used during processing (pre‑resize).
     * block_size: Embedding block size (default 8).

---

### **6. Troubleshooting & Tips**

* Form complains “image is required”
    - Use “Apply Attack” then “Send to Extract”, or manually select a file in the Extract panel.
* JSON parse error / HTML error page
    - The backend returns JSON for errors. If you pasted malformed keys, re‑copy from the Embed step.
* 413 Request Entity Too Large
    - Use a smaller image or JPEG; the server limits uploads. The app auto‑resizes large images internally.
* Extracted text is empty or wrong
    - Ensure you’re using the exact Keys from the Embed output for that image.
    - Severe geometric changes (heavy crop, large rotations) can defeat pre‑alignment; try smaller rotations.
* Debugging aid
    - The server logs a “SVD fingerprint diff” during extraction. Very large diffs indicate strong distortions.

---

### **7. How to Run (Local)**

* Python 3.10+ with packages from `requirements.txt`.
* Start the server (recommended): run the ASGI app with Uvicorn.
    - Module path: `asgi:asgi_app` (Flask app wrapped for ASGI).
* Open the printed local URL in your browser.

Quick steps on Windows (PowerShell):
1. python -m venv .venv ; . .venv/Scripts/Activate.ps1
2. pip install --upgrade pip ; pip install -r requirements.txt
3. uvicorn asgi:asgi_app --reload --port 5000
4. Visit http://127.0.0.1:5000/

---

### **8. Challenges (Observed in Practice)**

* Payload vs. Imperceptibility vs. Robustness trade‑off
    - Higher embedding strength helps survive attacks but risks visible artifacts.
* Geometric attacks
    - Large rotations, crops, or perspective changes can defeat alignment.
* Web payload constraints
    - Very large side keys are impractical. We addressed this via compact fingerprints.
* Browser form quirks
    - Native validation on hidden inputs and blob handling required careful JS logic.

---

### **9. Limitations (Current Release)**

* Best‑effort alignment only
    - ORB+RANSAC over a small thumbnail can’t correct severe transformations.
* Non‑blind extraction
    - Keys are required; this is not a fully blind watermarking scheme.
* LL‑focused embedding
    - Multi‑band mode is not enabled by default due to key size considerations.
* Computational cost
    - Block‑wise SVD is more expensive than simpler schemes, especially on large images.

---

### **10. Further Improvements**

* Authenticity signatures
    - Sign `(s_fingerprint_b64 || encrypted_text_b64)` with Ed25519 and verify at extract time.
* Stronger geometric resilience
    - Add phase correlation or ECC‑based alignment; multi‑scale keypoints; synchronization patterns.
* Color domain exploration
    - Evaluate YCbCr/Lab embedding to improve perceptual quality and robustness.
* Toward blind extraction
    - Explore embedding strategies that reduce reliance on side keys while preserving robustness.
* Re‑enable multi‑band with compression
    - Investigate advanced key compression to ship multi‑band embeddings without bloated JSON.

---

### **11. References**

* R. Gonzalez, R. Woods, “Digital Image Processing.”
* S. Mallat, “A Wavelet Tour of Signal Processing.”
* A. Menezes et al., “Handbook of Applied Cryptography.”
