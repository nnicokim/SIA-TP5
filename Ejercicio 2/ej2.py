import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt

# --- SILENCIADOR DE TENSORFLOW ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
import logging



import time

sys.path.append(os.path.abspath('../Ejercicio 1'))
from utils import ProgressCallback

from config_vae import load_config_vae
from data_loader_vae import cargar_datos_kanji
from models_vae import build_vae

def main():
    start_time = time.time()
    print("--- Iniciando Grid Search Dinámico de VAE (Kanjis) ---")
    
    latent_dims, arch_configs, lr_configs, epochs_configs, batch_size, use_cuda, seed = load_config_vae()

    # Semilla fija (configurable por SEED en .env) -> resultados reproducibles
    # corrida a corrida: misma init de pesos y mismo ruido de reparametrizacion.
    np.random.seed(seed)
    print(f">>> Semilla del RNG fijada en {seed} (reproducible). <<<")

    if use_cuda:
        print(">>> AVISO: Usando backend de CUDA activado desde el entorno. <<<")
    
    X_train = cargar_datos_kanji("symbol.h")
    input_dim = X_train.shape[1]
    
    dim_fija = latent_dims[0] 
    epocas_test = 1000

    mejor_arch = None
    mejor_loss_arch = float('inf')
    mejor_lr = None
    mejor_loss_lr = float('inf')

    # ==========================================================
    # TEST 1: IMPACTO DE LA ARQUITECTURA (Buscando el óptimo)
    # ==========================================================
    print("\n[1/4] Testeando Arquitecturas (con LR base de 0.001)...")
    plt.figure(figsize=(10, 6))
    for i, arch in enumerate(arch_configs):
        vae, _, _ = build_vae(input_dim, arch, dim_fija, learning_rate=0.001, use_cuda=use_cuda)
        spinner = ProgressCallback(len(arch_configs), i+1, f"Arch {arch}", epocas_test)
        
        hist = vae.fit(X_train, epochs=epocas_test, batch_size=batch_size, verbose=0, callbacks=[spinner])
        
        loss_final = hist.history['loss'][-1]
        plt.plot(hist.history['loss'], label=f"Capas: {arch} (Loss: {loss_final:.2f})")
        
        if loss_final < mejor_loss_arch:
            mejor_loss_arch = loss_final
            mejor_arch = arch
            
    plt.title("Impacto de la Arquitectura en la Pérdida del VAE")
    plt.xlabel("Épocas")
    plt.ylabel("Loss Total (Reconstrucción + KL)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("grafico_1_arquitecturas.png", dpi=300)
    plt.close()
    
    print(f"\n >>> Arquitectura Óptima Encontrada: {mejor_arch} (Loss: {mejor_loss_arch:.2f})")

    # ==========================================================
    # TEST 2: IMPACTO DEL LEARNING RATE (Usando la mejor arquitectura)
    # ==========================================================
    print(f"\n[2/4] Testeando Learning Rates sobre la arquitectura {mejor_arch}...")
    plt.figure(figsize=(10, 6))
    for i, lr in enumerate(lr_configs):
        vae, _, _ = build_vae(input_dim, mejor_arch, dim_fija, lr, use_cuda=use_cuda)
        spinner = ProgressCallback(len(lr_configs), i+1, f"LR {lr}", epocas_test)
        
        hist = vae.fit(X_train, epochs=epocas_test, batch_size=batch_size, verbose=0, callbacks=[spinner])
        
        loss_final = hist.history['loss'][-1]
        plt.plot(hist.history['loss'], label=f"LR: {lr} (Loss: {loss_final:.2f})")

        if not np.isnan(loss_final) and loss_final < mejor_loss_lr:
            mejor_loss_lr = loss_final
            mejor_lr = lr

    plt.title(f"Impacto del LR en la Convergencia (Arch GANADORA={mejor_arch})")
    plt.xlabel("Épocas")
    plt.ylabel("Loss Total")
    plt.ylim(top=mejor_loss_arch * 3) 
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("grafico_2_learning_rates.png", dpi=300)
    plt.close()

    if mejor_lr is None:
        print("\n [WARNING] Todos los LRs testeados divergieron (Loss=NaN). Usando el primero por defecto.")
        mejor_lr = lr_configs[0]

    print(f"\n >>> Learning Rate Óptimo Encontrado: {mejor_lr} (Loss: {mejor_loss_lr:.2f})")


    # ==========================================================
    # TEST 3: ENTRENAMIENTO FINAL CON HIPERPARÁMETROS ÓPTIMOS
    # ==========================================================
    max_epocas = max(epochs_configs)
    print(f"\n[3/4] Entrenando el Modelo Final por {max_epocas} épocas...")
    print(f"      -> Usando Arquitectura: {mejor_arch}")
    print(f"      -> Usando Learning Rate: {mejor_lr}")
    
    modelo_final, encoder, decoder = build_vae(input_dim, mejor_arch, dim_fija, mejor_lr, use_cuda=use_cuda)
    spinner_final = ProgressCallback(1, 1, "Modelo Final", max_epocas)
    
    hist_final = modelo_final.fit(X_train, epochs=max_epocas, batch_size=batch_size, verbose=0, callbacks=[spinner_final])
    
    plt.figure(figsize=(10, 6))
    plt.plot(hist_final.history['loss'], color='black', linewidth=2, label="Pérdida de Entrenamiento")
    
    for ep in epochs_configs:
        if ep <= max_epocas:
            loss_en_ep = hist_final.history['loss'][ep-1]
            plt.axvline(x=ep, color='red', linestyle='--', alpha=0.5)
            plt.scatter(ep, loss_en_ep, color='red', zorder=5)
            plt.annotate(f"  Ep {ep}\n  Loss: {loss_en_ep:.1f}", (ep, loss_en_ep), color='red', weight='bold')

    plt.title(f"Curva de Aprendizaje - Modelo Ganador (Arch {mejor_arch} | LR {mejor_lr})")
    plt.xlabel("Épocas")
    plt.ylabel("Loss Total")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("grafico_3_epocas_optimo.png", dpi=300)
    plt.close()

    # ==========================================================
    # CONSIGNAS B y C: ESPACIO LATENTE Y GENERACIÓN
    # ==========================================================
    print("\n[4/4] Generando gráficos de la Consigna con el MODELO ÓPTIMO...")
    
    # Consigna B
    z_mean, _, _ = encoder.predict(X_train, verbose=0)
    plt.figure(figsize=(8, 8))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=range(32), cmap='tab20', s=200, edgecolors='k', alpha=0.8)
    for i in range(32):
        plt.annotate(str(i+1), (z_mean[i, 0]+0.05, z_mean[i, 1]+0.05), fontsize=9, weight='bold')
    
    plt.axhline(0, color='grey', linestyle='--', alpha=0.5)
    plt.axvline(0, color='grey', linestyle='--', alpha=0.5)
    plt.title(f"CONSIGNA B: Espacio Latente 2D Continuo (Arch {mejor_arch})")
    plt.xlabel("Dimensión Latente 1 (\u03BC)")
    plt.ylabel("Dimensión Latente 2 (\u03BC)")
    plt.grid(True, alpha=0.3)
    plt.savefig("consigna_b_espacio_latente.png", dpi=300)
    plt.close()

    # Consigna C
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    fig.suptitle(f"CONSIGNA C: Kanjis Sintéticos Generados por el VAE Óptimo", fontsize=14)
    
    for i in range(5):
        z_sample = np.random.normal(loc=0.0, scale=1.0, size=(1, dim_fija))
        nuevo_kanji_raw = decoder.predict(z_sample, verbose=0)
        nuevo_kanji_img = np.round(nuevo_kanji_raw).reshape(10, 10)
        
        axes[i].imshow(nuevo_kanji_img, cmap='gray_r')
        axes[i].set_title(f"Z=({z_sample[0][0]:.1f}, {z_sample[0][1]:.1f})")
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig("consigna_c_kanjis_generados.png", dpi=300)
    plt.close()
    
    
    end_time = time.time()
    print(f"\n[TIME] El proceso completo tomó {end_time - start_time:.2f} segundos.")
    print("\n¡Proceso Completado Exitosamente!")

if __name__ == "__main__":
    main()