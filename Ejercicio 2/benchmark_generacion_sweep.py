import os
import time
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Ejercicio 1')))
from data_loader_vae import cargar_datos_kanji
from models_vae import build_vae
from config_vae import load_seed


def medir_lote(decoder, latent_dim, n_kanjis, reps, warmup):
    """Tiempo promedio de generar n_kanjis de una sola vez (un forward del decoder)."""
    z = np.random.normal(loc=0.0, scale=1.0, size=(n_kanjis, latent_dim))
    for _ in range(warmup):
        _ = decoder.predict(z, verbose=0)
    start = time.time()
    for _ in range(reps):
        _ = decoder.predict(z, verbose=0)
    end = time.time()
    return (end - start) / reps  # segundos por lote


def main():
    print("--- BARRIDO DE GENERACION: CPU vs GPU en funcion del tamano del lote ---")
    arch = [128, 64, 32]
    latent_dim = 2
    lr = 0.01
    seed = load_seed()

    X_train = cargar_datos_kanji("symbol.h")
    input_dim = X_train.shape[1]

    np.random.seed(seed)
    _, _, decoder_cpu = build_vae(input_dim, arch, latent_dim, lr, use_cuda=False)
    np.random.seed(seed)
    _, _, decoder_gpu = build_vae(input_dim, arch, latent_dim, lr, use_cuda=True)

    lotes = [1, 10, 100, 1000, 10000, 100000]
    print(f"\n{'N':>8} | {'CPU lote':>12} | {'GPU lote':>12} | {'CPU x kanji':>12} | {'GPU x kanji':>12} | ganador")
    print("-" * 90)

    for n in lotes:
        reps = max(3, min(2000, int(200000 / n)))
        warmup = max(2, reps // 10)
        t_cpu = medir_lote(decoder_cpu, latent_dim, n, reps, warmup)
        t_gpu = medir_lote(decoder_gpu, latent_dim, n, reps, warmup)
        ganador = "GPU" if t_gpu < t_cpu else "CPU"
        factor = (t_cpu / t_gpu) if t_gpu < t_cpu else (t_gpu / t_cpu)
        print(f"{n:>8} | {t_cpu*1e3:>9.3f} ms | {t_gpu*1e3:>9.3f} ms | "
              f"{t_cpu/n*1e6:>9.3f} us | {t_gpu/n*1e6:>9.3f} us | {ganador} ({factor:.1f}x)")


if __name__ == "__main__":
    main()
