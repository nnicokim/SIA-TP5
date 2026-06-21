import os
import json
from dotenv import load_dotenv

def load_seed():
    """Semilla del RNG (reproducibilidad). Configurable por SEED en .env."""
    load_dotenv(dotenv_path=".env")
    try:
        return int(os.getenv("SEED", "42"))
    except (ValueError, TypeError):
        return 42

def load_config_vae():
    load_dotenv(dotenv_path=".env")
    try:
        latent_dims = json.loads(os.getenv("VAE_LATENT_DIMS", "[2]"))
        arch_configs = json.loads(os.getenv("VAE_ARCH_CONFIGS", "[[64, 32], [128, 64], [128, 64, 32]]"))
        lr_configs = json.loads(os.getenv("VAE_LR_CONFIGS", "[0.001, 0.01, 0.0005]"))
        epochs_configs = json.loads(os.getenv("VAE_EPOCHS_CONFIGS", "[500, 1000, 1500]"))
        batch_size = int(os.getenv("BATCH_SIZE", "32"))
        use_cuda = os.getenv("USE_CUDA", "False").lower() in ('true', '1', 't')
        seed = int(os.getenv("SEED", "42"))
    except Exception as e:
        print(f"Error leyendo .env de VAE: {e}. Usando defaults.")
        latent_dims, arch_configs, lr_configs, epochs_configs, batch_size, use_cuda, seed = [2], [[128, 64]], [0.001], [1500], 32, False, 42

    return latent_dims, arch_configs, lr_configs, epochs_configs, batch_size, use_cuda, seed