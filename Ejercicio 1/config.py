import os
import json
from dotenv import load_dotenv

def load_config():
    load_dotenv()
    try:
        latent_dims = json.loads(os.getenv("LATENT_DIMS", "[2, 3, 4]"))
        arch_configs = json.loads(os.getenv("ARCH_CONFIGS", "[[32, 16]]"))
        lr_configs = json.loads(os.getenv("LR_CONFIGS", "[0.001]"))
        epochs_configs = json.loads(os.getenv("EPOCHS_CONFIGS", "[1000]"))
        batch_size = int(os.getenv("BATCH_SIZE", "32"))
    except Exception as e:
        print(f"Error leyendo .env, usando valores de respaldo. Detalles: {e}")
        latent_dims = [2, 3, 4]
        arch_configs = [[16, 8], [32, 16], [64, 32], [32, 16, 8], [64, 32, 16]]
        lr_configs = [0.1, 0.01, 0.001, 0.0005]
        epochs_configs = [500, 1000, 1500, 2000]
        batch_size = 32
        
    return latent_dims, arch_configs, lr_configs, epochs_configs, batch_size