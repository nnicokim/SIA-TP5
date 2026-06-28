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


def load_config_experimento():
    """
    Carga la configuracion del experimento nuevo (experimento.py / graficos.py).
    Devuelve un dict con: arquitecturas, learning rates, betas, baselines,
    parametros de early stopping, latente, batch, seed y use_cuda.
    """
    load_dotenv(dotenv_path=".env")
    try:
        cfg = {
            "latent_dim": json.loads(os.getenv("VAE_LATENT_DIMS", "[2]"))[0],
            "archs": json.loads(os.getenv("VAE_ARCH_CONFIGS", "[[16, 8], [32, 16], [64, 32], [32, 16, 8], [64, 32, 16]]")),
            "lrs": json.loads(os.getenv("VAE_LR_CONFIGS", "[0.1, 0.01, 0.001, 0.0005]")),
            "betas": json.loads(os.getenv("VAE_BETA_CONFIGS", "[0.001, 0.01, 0.1, 1.0]")),
            "lr_base": float(os.getenv("VAE_LR_BASE", "0.001")),
            "beta_base": float(os.getenv("VAE_BETA_BASE", "0.01")),
            "max_epochs": int(os.getenv("ES_MAX_EPOCHS", "5000")),
            "patience": int(os.getenv("ES_PATIENCE", "200")),
            "min_delta": float(os.getenv("ES_MIN_DELTA", "0.0001")),
            "batch_size": int(os.getenv("BATCH_SIZE", "32")),
            "use_cuda": os.getenv("USE_CUDA", "False").lower() in ("true", "1", "t"),
            "seed": int(os.getenv("SEED", "42")),
        }
    except Exception as e:
        print(f"Error leyendo .env del experimento: {e}. Usando defaults.")
        cfg = {
            "latent_dim": 2,
            "archs": [[16, 8], [32, 16], [64, 32], [32, 16, 8], [64, 32, 16]],
            "lrs": [0.1, 0.01, 0.001, 0.0005],
            "betas": [0.001, 0.01, 0.1, 1.0],
            "lr_base": 0.001, "beta_base": 0.01,
            "max_epochs": 5000, "patience": 200, "min_delta": 0.0001,
            "batch_size": 32, "use_cuda": False, "seed": 42,
        }
    return cfg