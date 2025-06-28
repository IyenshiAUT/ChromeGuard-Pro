/**
 * ChromaGuard Pro - Main JavaScript Application
 * 
 * This file contains all the client-side functionality for the ChromaGuard Pro
 * watermarking web application, including form handling, file uploads, API calls,
 * and attack simulation features.
 * 
 * Author: Image Processing Project
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
    
    // Store watermarked image blob for attack simulation
    let watermarkedImageBlob = null;

    // Add event listeners for file input changes
    coverImageInput.addEventListener('change', () => updateFileLabel('cover-image-input', 'cover-image-label'));
    extractImageInput.addEventListener('change', () => updateFileLabel('extract-image-input', 'extract-image-label'));

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
        } catch (error) {
            alert(`Error: ${error.message}`);
        } finally {
            // Reset UI state
            document.getElementById('embed-button').disabled = false;
            hideSpinner('embed-spinner');
        }
    });

    // Watermark extraction form submission
    extractForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Prepare form data for extraction
        const formData = new FormData();
        formData.append('image', extractImageInput.files[0]);
        formData.append('keys', document.getElementById('keys-input').value);
        
        // Update UI state
        document.getElementById('extract-button').disabled = true;
        showSpinner('extract-spinner');
        document.getElementById('extract-output-container').classList.add('hidden');

        try {
            // Send extraction request to server
            const response = await fetch('/extract', { method: 'POST', body: formData });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Extraction failed.');

            // Display extracted text
            document.getElementById('extracted-text-display').textContent = data.text;
            document.getElementById('extract-output-container').classList.remove('hidden');
        } catch (error) {
            alert(`Error: ${error.message}`);
        } finally {
            // Reset UI state
            document.getElementById('extract-button').disabled = false;
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
            
            // Display attacked image
            const attackedBlob = await response.blob();
            document.getElementById('attacked-image-display').src = URL.createObjectURL(attackedBlob);
            document.getElementById('attack-output').classList.remove('hidden');
            
            // Setup button to send attacked image to extraction
            document.getElementById('send-to-extract-btn').onclick = () => {
                const dataTransfer = new DataTransfer();
                dataTransfer.items.add(new File([attackedBlob], 'attacked_image.png', { type: 'image/png' }));
                extractImageInput.files = dataTransfer.files;
                updateFileLabel('extract-image-input', 'extract-image-label');
                document.getElementById('keys-input').scrollIntoView({ behavior: 'smooth' });
            };
        } catch (error) {
            alert(`Error: ${error.message}`);
        } finally {
            // Reset UI state
            document.getElementById('apply-attack-btn').disabled = false;
            hideSpinner('attack-spinner');
        }
    });
}); 