"""
ChromaGuard Pro - Utility Functions and Configuration

This module provides configuration settings and cryptographic utilities for the
watermarking system. It includes AES encryption/decryption functions and system
configuration parameters.

Author: Image Processing Project
"""

import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding

class Config:
    """
    Configuration class containing all system parameters for the watermarking process.
    
    This class defines the key parameters that control the behavior of the DWT-SVD
    watermarking algorithm, including embedding strength, image processing limits,
    and cryptographic settings.
    """
    
    # DWT-SVD specific parameters
    ALPHA_DWT_SVD = 0.25  # Embedding strength factor for singular value modification
    MAX_DIMENSION = 1024  # Maximum image dimension for processing (performance optimization)
    CHAOTIC_KEY = 0.01    # Initial value for logistic map in chaotic scrambling

    # Cryptographic parameters
    AES_KEY_SIZE = 16     # AES-128 key size in bytes
    AES_IV_SIZE = 16      # AES initialization vector size in bytes

class Crypto:
    """
    Cryptographic utility class providing AES encryption and decryption functionality.
    
    This class implements AES encryption in CBC mode with PKCS7 padding for securing
    the watermark data before embedding. The encryption ensures that even if the
    watermark is detected, the content remains confidential without the proper keys.
    """
    
    @staticmethod
    def encrypt(data_bytes, key, iv):
        """
        Encrypts data using AES-128 in CBC mode with PKCS7 padding.
        
        This method encrypts the watermark text before embedding to ensure
        confidentiality. The encryption uses AES-128 in CBC mode which provides
        both confidentiality and integrity protection.
        
        Args:
            data_bytes (bytes): Raw data to encrypt
            key (bytes): AES encryption key (16 bytes for AES-128)
            iv (bytes): Initialization vector (16 bytes)
            
        Returns:
            bytes: Encrypted data
            
        Note:
            The key and IV should be randomly generated for each watermark
            to ensure security. These are stored with the watermark for decryption.
        """
        # Create PKCS7 padder for proper padding
        padder = padding.PKCS7(algorithms.AES.block_size).padder()
        
        # Apply padding to the data
        padded_data = padder.update(data_bytes) + padder.finalize()
        
        # Create AES cipher in CBC mode
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        
        # Encrypt the padded data
        return encryptor.update(padded_data) + encryptor.finalize()

    @staticmethod
    def decrypt(encrypted_data, key, iv):
        """
        Decrypts data using AES-128 in CBC mode with PKCS7 padding.
        
        This method decrypts the extracted watermark data using the same
        key and IV that were used during encryption. The decryption process
        reverses the encryption to recover the original watermark text.
        
        Args:
            encrypted_data (bytes): Encrypted data to decrypt
            key (bytes): AES decryption key (must match encryption key)
            iv (bytes): Initialization vector (must match encryption IV)
            
        Returns:
            bytes: Decrypted data
            
        Note:
            The key and IV must be exactly the same as those used during
            encryption for successful decryption.
        """
        # Create AES cipher in CBC mode
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        
        # Decrypt the data
        decrypted_padded = decryptor.update(encrypted_data) + decryptor.finalize()
        
        # Remove PKCS7 padding
        unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
        return unpadder.update(decrypted_padded) + unpadder.finalize()
