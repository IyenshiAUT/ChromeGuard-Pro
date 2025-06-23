import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding

class Config:
    # DWT-SVD specific
    ALPHA_DWT_SVD = 0.25 
    MAX_DIMENSION = 1024
    CHAOTIC_KEY = 0.01

    # General
    AES_KEY_SIZE = 16 
    AES_IV_SIZE = 16

class Crypto:
    @staticmethod
    def encrypt(data_bytes, key, iv):
        padder = padding.PKCS7(algorithms.AES.block_size).padder()
        padded_data = padder.update(data_bytes) + padder.finalize()
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        return encryptor.update(padded_data) + encryptor.finalize()

    @staticmethod
    def decrypt(encrypted_data, key, iv):
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        decrypted_padded = decryptor.update(encrypted_data) + decryptor.finalize()
        unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
        return unpadder.update(decrypted_padded) + unpadder.finalize()
