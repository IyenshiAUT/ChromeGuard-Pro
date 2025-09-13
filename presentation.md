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
    1.  The cover image is split into color channels, and the **Blue channel** is selected for watermarking.
    2.  A **2D-DWT** is applied, decomposing the channel into four frequency sub-bands: `LL`, `LH`, `HL`, and `HH`.
    3.  The watermark is embedded in the `LL` (Approximation) sub-band, as it's the most visually significant and resilient to change.
    4.  **SVD** is applied to the `LL` sub-band. The singular values (which represent image intensity) are modified by adding the watermark data to them.
    5.  An **Inverse DWT** is performed to reconstruct the blue channel, which now invisibly contains the watermark.
    6.  The channels are merged back into the final, watermarked image.

---

### **4. Advanced Features for Security & Robustness**

ChromaGuard Pro goes beyond a basic algorithm by adding multiple layers of security.

*   **AES Encryption**:
    *   Before the watermark text is embedded, it is encrypted using the **AES (Advanced Encryption Standard)** algorithm.
    *   This means even if an attacker could extract the raw watermark data, they could not read the message without the secret encryption key and IV (Initialization Vector).

*   **Chaotic Scrambling**:
    *   The watermark (represented as a small binary image) is scrambled using a **logistic map**, a chaotic function.
    *   This pseudo-randomly shuffles the watermark's pixels before embedding. Without the initial "chaotic key," it is computationally infeasible to reverse the scrambling and reconstruct the watermark image.

*   **Multi-Band Embedding (Experimental)**:
    *   We implemented an enhancement to embed the watermark in **all four DWT sub-bands** (`LL`, `LH`, `HL`, `HH`).
    *   **Benefit**: This created extreme robustness, as an attack would have to destroy information across all frequency domains to remove the watermark.
    *   **Outcome**: While effective, this dramatically increased the size of the extraction keys, making it impractical for a web application. We reverted to the more efficient single-band (`LL`) embedding for stability.
