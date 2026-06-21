import os
import time
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Ejercicio 1')))
from utils import ProgressCallback
from data_loader_vae import cargar_datos_kanji
from models_vae import build_vae
from config_vae import load_seed

def main():
    print("--- BENCHMARK DE RENDIMIENTO VAE: CPU vs GPU ---")
    
    # Configuracion Optima
    arch = [128, 64, 32]
    latent_dim = 2
    lr = 0.01
    epochs = 1000
    batch_size = 32
    
    seed = load_seed()
    print(f"Semilla del RNG: {seed} (configurable por SEED en .env)")

    X_train = cargar_datos_kanji("symbol.h")
    input_dim = X_train.shape[1]

    # 1. RUN CPU
    print("\n[1/2] Corriendo modelo en CPU (NumPy puro)...")
    np.random.seed(seed)  # misma init que la GPU -> comparacion justa y reproducible
    vae_cpu, encoder_cpu, decoder_cpu = build_vae(input_dim, arch, latent_dim, lr, use_cuda=False)
    
    start_cpu = time.time()
    spinner_cpu = ProgressCallback(1, 1, "CPU", epochs)
    hist_cpu = vae_cpu.fit(X_train, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[spinner_cpu])
    end_cpu = time.time()
    time_cpu = end_cpu - start_cpu
    print(f"\n[CPU] Tiempo de ejecución: {time_cpu:.2f} segundos")
    
    # 2. RUN GPU
    print("\n[2/2] Corriendo modelo en GPU (CUDA)...")
    np.random.seed(seed)  # re-siembra: la GPU parte de la misma init que la CPU
    vae_gpu, encoder_gpu, decoder_gpu = build_vae(input_dim, arch, latent_dim, lr, use_cuda=True)
    
    start_gpu = time.time()
    spinner_gpu = ProgressCallback(1, 1, "GPU", epochs)
    hist_gpu = vae_gpu.fit(X_train, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[spinner_gpu])
    end_gpu = time.time()
    time_gpu = end_gpu - start_gpu
    print(f"\n[GPU] Tiempo de ejecución: {time_gpu:.2f} segundos")
    
    # 3. GRAFICO DE COMPARACION DE TIEMPO
    labels = ['CPU (NumPy)', 'GPU (CUDA)']
    times = [time_cpu, time_gpu]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, times, color=['#ff9999','#66b3ff'])
    plt.title("Comparación de Tiempo de Entrenamiento (1000 Épocas)")
    plt.ylabel("Tiempo (segundos)")
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f"{yval:.2f}s", va='bottom', ha='center')
        
    speedup = time_cpu / time_gpu if time_gpu > 0 else 0
    plt.text(0.5, max(times)*0.8, f"Speedup: {speedup:.2f}x", ha='center', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))
    
    plt.savefig("benchmark_cpu_vs_gpu_time.png", dpi=300)
    plt.close()
    
    # 4. GRAFICO DE CONVERGENCIA (LOSS)
    plt.figure(figsize=(10, 6))
    plt.plot(hist_cpu.history['loss'], label=f"CPU Loss", color='#ff9999')
    plt.plot(hist_gpu.history['loss'], label=f"GPU Loss", color='#66b3ff')
    plt.title("Evolución del Error (Loss) durante el Entrenamiento")
    plt.xlabel("Épocas")
    plt.ylabel("Loss Total")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("benchmark_cpu_vs_gpu_loss.png", dpi=300)
    plt.close()
    
    print("\nBenchmark finalizado.")
    print("Revisa 'benchmark_cpu_vs_gpu_time.png' para los tiempos.")
    print("Revisa 'benchmark_cpu_vs_gpu_loss.png' para comprobar que aprenden idéntico.")

if __name__ == "__main__":
    main()
