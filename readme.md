# ChromaGuard Pro: Advanced Invisible Watermarking

Welcome to **ChromaGuard Pro**, a complete, self-contained web application that allows users to embed and extract robust, invisible watermarks in images using a state-of-the-art algorithm.

This application combines a sophisticated, full-frame DWT-SVD watermarking method with a layer of AES encryption for high security. The entire project is served through a user-friendly web interface powered by a single Python script running a Flask server. It also includes a unique module for simulating common image attacks to test the resilience of the embedded watermark.

---

## Features

-   **Invisible Watermarking:** Embeds text data into images in a way that is imperceptible to the human eye, preserving the original quality of the image.
-   **Robust Algorithm:** Uses a powerful hybrid of **Discrete Wavelet Transform (DWT)** and **Singular Value Decomposition (SVD)** across the entire image frame, making the watermark highly resistant to common attacks like JPEG compression and noise.
-   **AES Encryption:** Your secret text is secured with the industry-standard **Advanced Encryption Standard (AES)** before being embedded, ensuring the content of your message remains confidential.
-   **Web-Based UI:** Provides a clean, modern, and responsive user interface for uploading images, embedding text, and extracting the hidden watermark.
-   **Attack Simulation:** Includes a built-in testing suite to apply various attacks (e.g., JPEG compression, noise, blur, rotation) to your watermarked image, allowing you to instantly test its resilience.
-   **Self-Contained & Easy to Run:** The entire application—backend, frontend, and API—is packaged into a single Python script, making it incredibly easy to set up and run.

---

## Technology Stack

-   **Backend:** Flask
-   **Image Processing:** OpenCV, NumPy
-   **Watermarking Algorithm:** PyWavelets (for DWT)
-   **Cryptography:** `cryptography` library (for AES)
-   **Frontend:** HTML, Tailwind CSS, JavaScript
-   **Image Handling:** Pillow (PIL)

---

## Project Structure

For a clean and manageable project, your directory should be organized as follows:

```
/chromaguard_pro/
├── app.py              # The main Flask application
├── watermarker.py      # Core watermarking logic
├── utils.py            # Configuration and Crypto classes
├── requirements.txt    # Project dependencies
└── /templates/
    └── index.html      # The HTML file for the frontend
```

---

## Installation

1.  **Set up the project directory and files** as shown in the structure above. Save the code from the corresponding Canvases into each file.

2.  **Create and activate a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required Python libraries** from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

---

## How to Run the Application

1.  **Navigate to the project directory** (`/chromaguard_pro/`) in your terminal.

2.  **Run the Flask application:**
    ```bash
    python app.py
    ```

3.  You will see output indicating the server is running, typically on `http://127.0.0.1:5000`.

4.  **Open your web browser** and navigate to:
    [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## How to Use the Tool

The web interface is split into two main sections:

### 1. Protect an Image
-   **Upload Original Image:** Click to select the image you want to protect.
-   **Enter Secret Text:** Type the message you want to embed.
-   **Encrypt & Embed:** Click the button to start the process.
-   **Save Your Assets:** The newly protected image will be displayed. You can download it. **Crucially, you must copy and save the unique "Decryption Keys"**, as they are required to get the watermark back.

### 2. Reveal a Watermark
-   **Upload Protected Image:** Select the image that contains your hidden watermark.
-   **Provide Decryption Keys:** Paste the full JSON key string you saved during the embedding step.
-   **Extract & Decrypt:** Click the button to begin the recovery process.
-   **View Result:** Your original secret message will be extracted, decrypted, and displayed.

### 3. Attack Simulation
-   After embedding a watermark, the Attack Simulation module will appear.
-   Select an attack type from the dropdown menu and click "Apply Attack".
-   The attacked image will be displayed. You can then click "Test Extraction on this Image" to automatically send it to the extraction panel and see if your watermark survived.
