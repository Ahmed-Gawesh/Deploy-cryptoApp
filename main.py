import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from fastapi.middleware.cors import CORSMiddleware
from keras import layers, models, ops 
import logging
import os
import base64
from typing import List

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define input and output models
class PlaintextInput(BaseModel):
    plaintext: List[str]  

class EncryptionOutput(BaseModel):
    results: List[dict]  

class CiphertextInput(BaseModel):
    ciphertexts: List[dict]  
    
class DecryptionOutput(BaseModel):
    plaintexts: List[str]  

class Config:
    P_LEN = 32
    K_LEN = 32
    C_LEN = 32

config = Config()

class AsymmetricEncryption:
    def __init__(self, config):
        self.config = config
        self.key_gen = models.load_model("key_gen.keras")
        self.alice = models.load_model("alice.keras")
        self.bob = models.load_model("bob.keras")
        self.pvk_inputs = np.random.choice([-1, 1], size=(1, config.K_LEN)).astype(np.float32)
        self.pub_key = self.key_gen.predict(self.pvk_inputs, verbose=0)

encryption_system = AsymmetricEncryption(config)

def text_to_tensor(text: str, p_len: int) -> tuple[np.ndarray, int]:
    def char_to_binary(ch: str) -> list[int]:
        return [int(bit) for bit in format(ord(ch), "08b")]
    binary = np.array([char_to_binary(ch) for ch in text]).flatten()
    pad = (p_len - len(binary) % p_len) % p_len
    tensor = np.concatenate([(binary * 2) - 1, np.zeros(pad)])
    return tensor, pad

def tensor_to_text(tensor: np.ndarray, pad: int) -> str:
    def binary_to_char(binary: np.ndarray) -> str:
        return chr(int("".join([str(bit) for bit in binary]), 2))
    binary = np.round((tensor + 1) / 2.0).astype(int).flatten()
    binarys = [binary[i: i + 8] for i in range(0, len(binary) - pad, 8)]
    return "".join(map(binary_to_char, binarys))

def numpy_to_base64(arr: np.ndarray) -> str:
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode('ascii')

def asymmetric_encryption(encryption_system: AsymmetricEncryption, plaintext: str) -> tuple[str, int]:
    tensor, pad = text_to_tensor(plaintext, encryption_system.config.P_LEN)
    p_inputs = np.array(tensor).reshape(-1, encryption_system.config.P_LEN).astype(np.float32)
    batch_size = p_inputs.shape[0]
    pub_key_tiled = np.tile(encryption_system.pub_key, (batch_size, 1))
    alice_output = encryption_system.alice.predict([p_inputs, pub_key_tiled], verbose=0)
    ciphertext_base64 = numpy_to_base64(alice_output.flatten())
    return ciphertext_base64, pad

def asymmetric_encryption_list(encryption_system: AsymmetricEncryption, plaintexts: List[str]) -> List[dict]:
    results = []
    for plaintext in plaintexts:
        ciphertext, pad = asymmetric_encryption(encryption_system, plaintext)
        results.append({"ciphertext": ciphertext, "pad": pad})
    return results

def asymmetric_decryption(encryption_system: AsymmetricEncryption, ciphertext_base64: str, pad: int) -> str:
    c_len = encryption_system.config.C_LEN
    bytes_data = base64.b64decode(ciphertext_base64)
    dtype = np.float32
    total_elements = len(bytes_data) // np.dtype(dtype).itemsize
    if total_elements % c_len != 0:
        raise ValueError("Ciphertext length is not compatible with C_LEN")
    batch_size = total_elements // c_len
    alice_output = np.frombuffer(bytes_data, dtype=dtype).reshape(batch_size, c_len)
    pvk_inputs_tiled = np.tile(encryption_system.pvk_inputs, (batch_size, 1))
    bob_output = encryption_system.bob.predict([alice_output, pvk_inputs_tiled], verbose=0)
    plaintext_bob = tensor_to_text(bob_output, pad)
    return plaintext_bob

def asymmetric_decryption_list(encryption_system: AsymmetricEncryption, ciphertexts: List[dict]) -> List[str]:
    plaintexts = []
    for item in ciphertexts:
        plaintext = asymmetric_decryption(encryption_system, item["ciphertext"], item["pad"])
        plaintexts.append(plaintext)
    return plaintexts

# Encryption endpoint
@app.post("/encrypt", response_model=EncryptionOutput)
async def encrypt(input: PlaintextInput):
    results = asymmetric_encryption_list(encryption_system, input.plaintext)
    return {"results": results}

# Decryption endpoint
@app.post("/decrypt", response_model=DecryptionOutput)
async def decrypt(input: CiphertextInput):
    plaintexts = asymmetric_decryption_list(encryption_system, input.ciphertexts)
    return {"plaintexts": plaintexts}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

