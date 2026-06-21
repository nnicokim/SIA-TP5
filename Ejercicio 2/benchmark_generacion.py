import os
import time
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Ejercicio 1')))
from data_loader_vae import cargar_datos_kanji
from models_vae import build_vae
from config_vae import load_seed


def medir_generacion(decoder, latent_dim, n_reps=5000, n_warmup=300):
    """
    Mide el tiempo promedio de generar UN kanji: un vector z (1, latent_dim)
    pasado por el decoder -> imagen 10x10.

    - Warmup (no se cronometra): absorbe la creacion del contexto CUDA y la
      primera asignacion de tensores intermedios en GPU. Son costos de "una sola
      vez" que no forman parte del costo estable de generar.
    - Promedio sobre n_reps: generar un kanji toma microsegundos, asi que el ruido
      de medicion domina una sola corrida. Promediar muchas repeticiones da un
      numero estable y comparable.

    En GPU cada llamada incluye el viaje completo: subir z (Host->Device),
    lanzar los kernels de cada capa, y bajar la imagen (Device->Host). Ese es el
    costo real de "generar un kanji y tenerlo de vuelta en CPU".
    """
    z = np.random.normal(loc=0.0, scale=1.0, size=(1, latent_dim))

    # Warmup (descartado)
    for _ in range(n_warmup):
        _ = decoder.predict(z, verbose=0)

    # Medicion
    start = time.time()
    for _ in range(n_reps):
        _ = decoder.predict(z, verbose=0)
    end = time.time()

    return (end - start) / n_reps  # segundos por kanji generado


def main():
    print("--- BENCHMARK DE GENERACION VAE (1 kanji): CPU vs GPU ---")

    # Misma config optima que el benchmark de entrenamiento.
    arch = [128, 64, 32]
    latent_dim = 2
    lr = 0.01

    seed = load_seed()
    print(f"Semilla del RNG: {seed} (configurable por SEED en .env)")

    X_train = cargar_datos_kanji("symbol.h")
    input_dim = X_train.shape[1]

    # El costo de generar NO depende de si el modelo esta entrenado: las
    # operaciones (matmuls + activaciones) y sus tamanos son identicos con pesos
    # entrenados o random. Por eso construimos los modelos y medimos directo.

    # 1. CPU
    print("\n[1/2] Midiendo generacion en CPU (NumPy puro)...")
    np.random.seed(seed)  # misma init que la GPU -> comparacion justa
    _, _, decoder_cpu = build_vae(input_dim, arch, latent_dim, lr, use_cuda=False)
    t_cpu = medir_generacion(decoder_cpu, latent_dim)
    print(f"[CPU] Tiempo por kanji: {t_cpu * 1e6:.2f} us")

    # 2. GPU
    print("\n[2/2] Midiendo generacion en GPU (CUDA)...")
    np.random.seed(seed)
    _, _, decoder_gpu = build_vae(input_dim, arch, latent_dim, lr, use_cuda=True)
    t_gpu = medir_generacion(decoder_gpu, latent_dim)
    print(f"[GPU] Tiempo por kanji: {t_gpu * 1e6:.2f} us")

    # 3. GRAFICO: comparacion de tiempo de generar 1 kanji
    labels = ['CPU (NumPy)', 'GPU (CUDA)']
    tiempos_us = [t_cpu * 1e6, t_gpu * 1e6]  # microsegundos

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, tiempos_us, color=['#ff9999', '#66b3ff'])
    plt.title("Tiempo de Generacion de 1 Kanji (z -> imagen 10x10)")
    plt.ylabel("Tiempo por generacion (microsegundos)")

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f"{yval:.1f} us",
                 va='bottom', ha='center')

    ratio = t_gpu / t_cpu if t_cpu > 0 else 0
    if ratio > 1:
        veredicto = f"GPU {ratio:.1f}x mas LENTA"
    else:
        veredicto = f"GPU {1 / ratio:.1f}x mas rapida"
    plt.text(0.5, max(tiempos_us) * 0.85, veredicto, ha='center', fontsize=13,
             bbox=dict(facecolor='yellow', alpha=0.6))

    plt.savefig("benchmark_generacion_kanji.png", dpi=300)
    plt.close()

    # 4. JUSTIFICACION (en texto, por consola)
    print("\n" + "=" * 60)
    print("RESULTADO")
    print("=" * 60)
    print(f"CPU: {t_cpu * 1e6:.1f} us/kanji   |   GPU: {t_gpu * 1e6:.1f} us/kanji")
    if ratio > 1:
        print(f"-> Generando de a 1 kanji, la GPU es {ratio:.1f}x MAS LENTA que la CPU.")
        print("   Causa: el decoder es minusculo ([2->32->64->128->100], batch=1).")
        print("   Por cada generacion la GPU paga costos FIJOS que no puede amortizar:")
        print("     - lanzamiento de ~12 kernels CUDA (uno por matmul/bias/activacion),")
        print("     - 2 transferencias por PCIe (subir z, bajar la imagen),")
        print("     - marshalling de ctypes en cada llamada.")
        print("   La CPU resuelve 4 matmuls diminutos en NumPy sin nada de eso.")
        print("   La GPU empezaria a ganar cuando el batch sea lo bastante grande")
        print("   como para que el computo paralelo supere a esos costos fijos")
        print("   (generar miles/decenas de miles de kanjis de una sola vez).")
    print("\nRevisa 'benchmark_generacion_kanji.png'.")


if __name__ == "__main__":
    main()
