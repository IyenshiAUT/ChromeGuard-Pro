/**
 * ChromaGuard Pro - Main JavaScript Application
 * * This file contains all the client-side functionality for the ChromaGuard Pro
 * watermarking web application, including form handling, file uploads, API calls,
 * and attack simulation features.
 * * Author: Image Processing Project
 */

// Utility functions for UI management
function showSpinner(spinnerId) {
    document.getElementById(spinnerId).classList.remove('hidden');
}

function hideSpinner(spinnerId) {
    document.getElementById(spinnerId).classList.add('hidden');
}

// Update file input labels to show selected filename
function updateFileLabel(inputId, labelId) {
    const input = document.getElementById(inputId);
    const label = document.getElementById(labelId);
    if (input.files.length > 0) {
        label.innerHTML = `<svg class="icon text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg> <span>${input.files[0].name}</span>`;
    }
}

// Main application logic
document.addEventListener('DOMContentLoaded', () => {
    // Get form and input elements
    const embedForm = document.getElementById('embed-form');
    const extractForm = document.getElementById('extract-form');
    const coverImageInput = document.getElementById('cover-image-input');
    const extractImageInput = document.getElementById('extract-image-input');

    // Disable native HTML5 validation; we validate in JS
    if (extractForm) extractForm.setAttribute('novalidate', '');

    // Ensure the hidden file input has a name and is NOT required by the browser.
    // We will handle validation manually in our submit event listener.
    if (extractImageInput) {
        if (!extractImageInput.name) extractImageInput.name = 'image';
        extractImageInput.removeAttribute('required');
    }

    // Store watermarked image blob for attack simulation and extraction
    let watermarkedImageBlob = null;
    let attackedImageBlob = null; // Will hold the blob after an attack

    // Helper: ensure we have an attacked blob (fetch from preview <img> if needed)
    async function ensureAttackedBlob() {
        if (attackedImageBlob) return attackedImageBlob;
        const img = document.getElementById('attacked-image-display');
        if (img && img.src && (img.src.startsWith('blob:') || img.src.startsWith('data:'))) {
            try {
                const res = await fetch(img.src);
                attackedImageBlob = await res.blob();
                return attackedImageBlob;
            } catch { /* ignore */ }
        }
        return null;
    }

    // Add event listeners for file input changes
    coverImageInput.addEventListener('change', () => updateFileLabel('cover-image-input', 'cover-image-label'));
    extractImageInput.addEventListener('change', () => {
        updateFileLabel('extract-image-input', 'extract-image-label');
        // If user manually selects a file, clear the attacked image blob
        attackedImageBlob = null;
    });

    // Watermark embedding form submission
    embedForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Prepare form data for embedding
        const formData = new FormData();
        const coverFile = coverImageInput.files[0];
        formData.append('image', coverFile);
        formData.append('text', document.getElementById('watermark-text-input').value);
        // Pass the filename in the form data for proper download naming
        formData.append('filename', coverFile.name);

        // Update UI state
        document.getElementById('embed-button').disabled = true;
        showSpinner('embed-spinner');
        document.getElementById('embed-output-container').classList.add('hidden');
        document.getElementById('attack-section').classList.add('hidden');

        try {
            // Send embedding request to server
            const response = await fetch('/embed', { method: 'POST', body: formData });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Embedding failed.');

            // Display watermarked image
            const imageUrl = `data:image/png;base64,${data.image}`;
            document.getElementById('watermarked-image-display').src = imageUrl;

            // Setup download link
            const downloadLink = document.getElementById('download-watermarked-link');
            downloadLink.href = imageUrl;
            // Set proper filename for download
            downloadLink.download = `ChromaGuard_${data.original_filename}`;

            // Display decryption keys
            document.getElementById('extraction-keys-display').value = JSON.stringify(data.keys, null, 2);
            document.getElementById('embed-output-container').classList.remove('hidden');

            // Store watermarked image blob for attack simulation
            const imageResponse = await fetch(imageUrl);
            watermarkedImageBlob = await imageResponse.blob();
            document.getElementById('attack-section').classList.remove('hidden');
            document.getElementById('attack-output').classList.add('hidden');
            attackedImageBlob = null; // Reset attacked blob on new embed
        } catch (error) {
            alert(`Error: ${error.message}`);
        } finally {
            // Reset UI state
            document.getElementById('embed-button').disabled = false;
            hideSpinner('embed-spinner');
        }
    });

    // Send to Extract button functionality (augmented)
    document.getElementById('send-to-extract-btn')?.addEventListener('click', async () => {
        // Clear manual file to avoid ambiguity
        if (extractImageInput) extractImageInput.value = '';

        // Try to ensure we have a blob (in case page was refreshed)
        await ensureAttackedBlob();

        // Update label to indicate we’ll use the attacked image
        const extractLabel = document.getElementById('extract-image-label');
        if (extractLabel) {
            extractLabel.innerHTML = `<svg class="icon text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg> <span>Using attacked image</span>`;
        }

        // Prefill keys input from display if empty
        const keysInput = document.getElementById('keys-input');
        const keysDisplay = document.getElementById('extraction-keys-display');
        if (keysInput && keysDisplay && !keysInput.value.trim()) {
            keysInput.value = keysDisplay.value.trim();
        }
        keysInput?.scrollIntoView({ behavior: 'smooth' });
    });

    // Watermark extraction form submission
    document.getElementById('extract-form')?.addEventListener('submit', async (e) => {
        e.preventDefault();

        const blob = await ensureAttackedBlob();
        const btn = document.getElementById('extract-button');
        btn && (btn.disabled = true);
        showSpinner('extract-spinner');
        document.getElementById('extract-output-container')?.classList.add('hidden');

        try {
            const formData = new FormData();

            // Prefer attacked blob; fallback to selected file
            if (blob) {
                formData.append('image', blob, 'attacked_image.png'); // ensure a filename is provided
            } else if (extractImageInput?.files?.length) {
                formData.append('image', extractImageInput.files[0]);
            } else {
                alert('Image is required. Select a file or run an attack and click "Send to Extract".');
                // We return here because our manual validation failed.
                return; 
            }

            // Keys: use keys-input; fallback to the displayed keys textarea
            const keysInput = document.getElementById('keys-input');
            const keysDisplay = document.getElementById('extraction-keys-display');
            const keysText = (keysInput?.value || '').trim() || (keysDisplay?.value || '').trim();
            if (!keysText) {
                alert('Keys are required. Paste keys or embed again.');
                return;
            }
            formData.append('keys', keysText);

            // Send request
            const res = await fetch('/extract', { method: 'POST', body: formData });
            const isJson = res.headers.get('content-type')?.includes('application/json');
            const data = isJson ? await res.json() : { error: await res.text() };

            if (!res.ok) {
                console.error('/extract failed', data);
                alert(data.error || 'Extraction failed.');
                return;
            }

            document.getElementById('extracted-text-display').textContent = data.text || '';
            document.getElementById('extract-output-container')?.classList.remove('hidden');
        } catch (err) {
            console.error('Extract error', err);
            alert(`Extraction error. Check console.`);
        } finally {
            btn && (btn.disabled = false);
            hideSpinner('extract-spinner');
        }
    });

    // Copy keys button functionality
    document.getElementById('copy-keys-button').addEventListener('click', () => {
        const keysDisplay = document.getElementById('extraction-keys-display');
        keysDisplay.select();
        document.execCommand('copy');
        const btn = document.getElementById('copy-keys-button');
        btn.textContent = 'Copied!';
        setTimeout(() => { btn.textContent = 'Copy Keys'; }, 2000);
    });

    // Attack simulation functionality
    const attackType = document.getElementById('attack-type');
    const attackParamsDiv = document.getElementById('attack-params');

    // Configuration for different attack parameters
    const attackParamsConfig = {
        jpeg: '<label class="text-sm">Quality (1-100): <input type="number" id="jpeg-quality" value="50" min="1" max="100" class="w-full mt-1 p-1 border rounded"></label>',
        gauss_noise: '<label class="text-sm">Std Dev (1-50): <input type="number" id="noise-std" value="10" min="1" max="50" class="w-full mt-1 p-1 border rounded"></label>',
        salt_pepper_noise: '<label class="text-sm">Amount (0.01-0.2): <input type="number" id="sp-amount" value="0.05" min="0.01" max="0.2" step="0.01" class="w-full mt-1 p-1 border rounded"></label>',
        blur: '<label class="text-sm">Kernel Size (3-15, odd): <input type="number" id="blur-ksize" value="5" min="3" max="15" step="2" class="w-full mt-1 p-1 border rounded"></label>',
        resize: '<label class="text-sm">Scale Factor (0.1-0.9): <input type="number" id="resize-factor" value="0.5" min="0.1" max="0.9" step="0.1" class="w-full mt-1 p-1 border rounded"></label>',
        rotate: '<label class="text-sm">Angle (-45 to 45): <input type="number" id="rotate-angle" value="5" min="-45" max="45" class="w-full mt-1 p-1 border rounded"></label>'
    };

    // Update attack parameters when attack type changes
    attackType.addEventListener('change', () => {
        attackParamsDiv.innerHTML = attackParamsConfig[attackType.value] || '';
    });
    attackType.dispatchEvent(new Event('change'));

    // Apply attack button functionality
    document.getElementById('apply-attack-btn').addEventListener('click', async () => {
        if (!watermarkedImageBlob) {
            alert("Please embed a watermark first.");
            return;
        }

        // Prepare form data for attack
        const formData = new FormData();
        formData.append('image', watermarkedImageBlob, 'watermarked.png');
        formData.append('attack_type', attackType.value);

        // Collect attack parameters based on selected attack type
        let params = {};
        const quality = document.getElementById('jpeg-quality'); if (quality) params.quality = quality.value;
        const std_dev = document.getElementById('noise-std'); if (std_dev) params.std_dev = std_dev.value;
        const amount = document.getElementById('sp-amount'); if (amount) params.amount = amount.value;
        const ksize = document.getElementById('blur-ksize'); if (ksize) params.ksize = ksize.value;
        const factor = document.getElementById('resize-factor'); if (factor) params.factor = factor.value;
        const angle = document.getElementById('rotate-angle'); if (angle) params.angle = angle.value;

        formData.append('params', JSON.stringify(params));

        // Update UI state
        document.getElementById('apply-attack-btn').disabled = true;
        showSpinner('attack-spinner');
        document.getElementById('attack-output').classList.add('hidden');

        try {
            // Send attack request to server
            const response = await fetch('/attack', { method: 'POST', body: formData });
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Attack failed.');
            }

            // Display attacked image and STORE the blob
            attackedImageBlob = await response.blob(); // Store the blob in our variable
            document.getElementById('attacked-image-display').src = URL.createObjectURL(attackedImageBlob);
            document.getElementById('attack-output').classList.remove('hidden');

        } catch (error) {
            alert(`Error: ${error.message}`);
        } finally {
            // Reset UI state
            document.getElementById('apply-attack-btn').disabled = false;
            hideSpinner('attack-spinner');
        }
    });
});