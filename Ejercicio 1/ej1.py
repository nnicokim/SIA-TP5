import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import itertools
from dotenv import load_dotenv

# Componentes de TensorFlow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# ==========================================
# 0. CARGA DE CONFIGURACIÓN (.env) Y DATOS (font.h)
# ==========================================
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

def cargar_datos_font(filepath="font.h"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró el archivo '{filepath}'.")
    with open(filepath, 'r') as file:
        contenido = file.read()
    contenido = re.sub(r'//.*?\n|/\*.*?\*/', '', contenido, flags=re.S)
    hex_valores = re.findall(r'0x[0-9a-fA-F]+', contenido)
    if len(hex_valores) != 224:
        raise ValueError(f"Se esperaban 224 valores hex, se encontraron {len(hex_valores)}.")
    
    datos_binarios = []
    for hv in hex_valores:
        bits_str = format(int(hv, 16), '05b')
        datos_binarios.extend([int(b) for b in bits_str])
        
    return np.array(datos_binarios, dtype='float32').reshape(32, 35)

X_train = cargar_datos_font("font.h")
input_dim = 35 

# ==========================================
# 1. FUNCIÓN CONSTRUCTORA DINÁMICA
# ==========================================
def build_autoencoder(hidden_layers, latent_dim, learning_rate=0.001):
    inputs = Input(shape=(input_dim,))
    x = inputs
    for units in hidden_layers:
        x = Dense(units, activation='relu')(x)
    latent_space = Dense(latent_dim, activation='linear', name="latent_space")(x)
    
    x = latent_space
    for units in reversed(hidden_layers):
        x = Dense(units, activation='relu')(x)
    outputs = Dense(input_dim, activation='sigmoid')(x)
    
    autoencoder = Model(inputs, outputs)
    encoder = Model(inputs, latent_space)
    
    decoder_input = Input(shape=(latent_dim,))
    x_dec = decoder_input
    for i in range(len(hidden_layers) + 1):
        x_dec = autoencoder.layers[-(len(hidden_layers) + 1) + i](x_dec)
    decoder = Model(decoder_input, x_dec)
    
    autoencoder.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return autoencoder, encoder, decoder

# ==========================================
# 2. GRID SEARCH GENERAL
# ==========================================
best_loss = float('inf')
best_params = None
best_models = None
best_2d_models = None
best_2d_loss = float('inf')

history_dict = {}
registro_metricas = []

total_combinaciones = len(latent_dims) * len(arch_configs) * len(lr_configs) * len(epochs_configs)
print(f"Iniciando Grid Search Optimizador ({total_combinaciones} combinaciones)...")

for lat_dim, arch, lr, ep in itertools.product(latent_dims, arch_configs, lr_configs, epochs_configs):
    param_str = f"Dim:{lat_dim} | Arch:{arch} | LR:{lr} | Ep:{ep}"
    
    ae, enc, dec = build_autoencoder(arch, latent_dim=lat_dim, learning_rate=lr)
    history = ae.fit(X_train, X_train, epochs=ep, batch_size=batch_size, verbose=0)
    final_loss = history.history['loss'][-1]
    
    # Prevenir NaN si el LR explotó
    if np.isnan(final_loss):
        final_loss = 1.0
        
    history_dict[param_str] = history.history['loss']
    registro_metricas.append({
        'latent_dim': lat_dim,
        'arch': str(arch),
        'lr': lr,
        'epochs': ep,
        'loss': final_loss
    })
    
    if final_loss < best_loss:
        best_loss = final_loss
        best_params = {'latent_dim': lat_dim, 'arch': arch, 'lr': lr, 'epochs': ep}
        best_models = (ae, enc, dec)
        
    if lat_dim == 2 and final_loss < best_2d_loss:
        best_2d_loss = final_loss
        best_2d_models = (ae, enc, dec)

print(f"\n--- ¡VALORES ÓPTIMOS ENCONTRADOS! ---")
print(f"-> Espacio Latente: {best_params['latent_dim']}D")
print(f"-> Capas Ocultas: {best_params['arch']}")
print(f"-> Learning Rate: {best_params['lr']}")
print(f"-> Épocas: {best_params['epochs']}")
print(f"-> Pérdida Mínima Absoluta (MSE): {best_loss:.5f}")

autoencoder_opt, encoder_opt, decoder_opt = best_models

# ==========================================
# 3. BLOQUE DE HISTOGRAMAS Y GRÁFICOS BASE
# ==========================================

# Gráfico 1: Curvas de Pérdidas Globales
plt.figure(figsize=(12, 6))
opt_key = f"Dim:{best_params['latent_dim']} | Arch:{best_params['arch']} | LR:{best_params['lr']} | Ep:{best_params['epochs']}"
for params, loss_history in history_dict.items():
    if params == opt_key:
        plt.plot(loss_history, label="Configuración Óptima", linewidth=3, color='black', zorder=10)
    else:
        plt.plot(loss_history, alpha=0.1, linewidth=0.5)
plt.title("Historial de Pérdidas en el Espacio de Soluciones")
plt.xlabel("Épocas")
plt.ylabel("Pérdida (MSE) - Escala Logarítmica")
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("1_comparativa_hiperparametros.png", dpi=300)
plt.close()

# Gráfico 2: Dispersión Espacio Latente 2D
_, encoder_2d, _ = best_2d_models
latent_points = encoder_2d.predict(X_train, verbose=0)
plt.figure(figsize=(8, 6))
plt.scatter(latent_points[:, 0], latent_points[:, 1], c=range(32), cmap='jet', s=130, edgecolors='black')
for i in range(32):
    plt.annotate(str(i), (latent_points[i, 0]+0.04, latent_points[i, 1]+0.04), fontsize=9, weight='bold')
plt.title("Mapeo de Caracteres en Espacio Latente 2D Reconstruido")
plt.xlabel("Dimensión Latente 1")
plt.ylabel("Dimensión Latente 2")
plt.grid(True, alpha=0.3)
plt.savefig("2_espacio_latente_2d.png", dpi=300)
plt.close()

# Gráfico 3: Generación Sintética
centro_latente = np.mean(encoder_opt.predict(X_train, verbose=0), axis=0)
punto_sintetico = centro_latente + np.random.uniform(-1.0, 1.0, size=(best_params['latent_dim'],))
nueva_letra_raw = decoder_opt.predict(punto_sintetico.reshape(1, best_params['latent_dim']), verbose=0)
nueva_letra_img = np.round(nueva_letra_raw).reshape(7, 5)
plt.figure(figsize=(3, 4))
plt.imshow(nueva_letra_img, cmap='gray_r')
plt.title("Nueva Letra Inventada")
plt.axis('off')
plt.savefig("3_nueva_letra_generada.png", dpi=300)
plt.close()

# Gráfico 4: Denoising Autoencoder
dae, _, _ = build_autoencoder(best_params['arch'], latent_dim=best_params['latent_dim'], learning_rate=best_params['lr'])
X_train_ruido = np.copy(X_train)
mascara = np.random.rand(*X_train.shape) < 0.15
X_train_ruido[mascara] = 1 - X_train_ruido[mascara]
dae.fit(X_train_ruido, X_train, epochs=best_params['epochs'], batch_size=batch_size, verbose=0)

niveles_ruido = [0.10, 0.25, 0.50]
fig, axes = plt.subplots(len(niveles_ruido), 3, figsize=(9, 9))
for idx, p in enumerate(niveles_ruido):
    ent_r = np.copy(X_train[1:2])
    masc_r = np.random.rand(*ent_r.shape) < p
    ent_r[masc_r] = 1 - ent_r[masc_r]
    pred = np.round(dae.predict(ent_r, verbose=0))
    axes[idx, 0].imshow(X_train[1].reshape(7, 5), cmap='gray_r')
    axes[idx, 0].axis('off')
    axes[idx, 1].imshow(ent_r.reshape(7, 5), cmap='gray_r')
    axes[idx, 1].axis('off')
    axes[idx, 2].imshow(pred.reshape(7, 5), cmap='gray_r')
    axes[idx, 2].axis('off')
plt.tight_layout()
plt.savefig("4_resultado_denoising.png", dpi=300)
plt.close()

# ==========================================
# 4. CUADROS COMPARATIVOS INDEPENDIENTES (REQUERIDOS)
# ==========================================

# --- Gráfico 5: Comparativa de ÉPOCAS ---
plt.figure(figsize=(8, 5))
eps_eje = sorted(list(set([r['epochs'] for r in registro_metricas])))
min_loss_ep = [min([r['loss'] for r in registro_metricas if r['epochs'] == e]) for e in eps_eje]
plt.plot([str(e) for e in eps_eje], min_loss_ep, marker='s', linewidth=2.5, color='forestgreen', markersize=8)
plt.title("Impacto de la Cantidad de Épocas en el Límite de Error", weight='bold')
plt.xlabel("Épocas de Entrenamiento")
plt.ylabel("Mínimo MSE Alcanzado")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("5_comparativa_epochs.png", dpi=300)
plt.close()
print("[OK] Guardada la comparación aislada de Épocas en '5_comparativa_epochs.png'")

# --- Gráfico 6: Comparativa de LEARNING RATES ---
plt.figure(figsize=(8, 5))
lrs_eje = sorted(list(set([r['lr'] for r in registro_metricas])))
min_loss_lr = [min([r['loss'] for r in registro_metricas if r['lr'] == l]) for l in lrs_eje]
plt.plot([str(l) for l in lrs_eje], min_loss_lr, marker='o', linewidth=2.5, color='crimson', markersize=8)
plt.title("Impacto de la Tasa de Aprendizaje (LR) en la Estabilidad", weight='bold')
plt.xlabel("Learning Rate Evaluados")
plt.ylabel("Mínimo MSE Alcanzado")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("6_comparativa_lr.png", dpi=300)
plt.close()
print("[OK] Guardada la comparación aislada de LR en '6_comparativa_lr.png'")

# --- Gráfico 7: Comparativa de ARQUITECTURAS ---
plt.figure(figsize=(9, 5))
archs_eje = sorted(list(set([r['arch'] for r in registro_metricas])))
min_loss_arch = [min([r['loss'] for r in registro_metricas if r['arch'] == a]) for a in archs_eje]
plt.barh(archs_eje, min_loss_arch, color='darkorchid', edgecolor='black', height=0.5)
plt.title("Impacto de la Capacidad de las Capas Ocultas (2D vs 3D/4D de profundidad)", weight='bold')
plt.xlabel("Mínimo MSE Alcanzado")
plt.ylabel("Topología de Capas (Encoder)")
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig("7_comparativa_architectures.png", dpi=300)
plt.close()
print("[OK] Guardada la comparación aislada de Arquitecturas en '7_comparativa_architectures.png'")

# --- Gráfico 8: Comparativa de ESPACIOS LATENTES ---
plt.figure(figsize=(7, 5))
dims_eje = sorted(list(set([r['latent_dim'] for r in registro_metricas])))
min_loss_dim = [min([r['loss'] for r in registro_metricas if r['latent_dim'] == d]) for d in dims_eje]
plt.bar([f"{d}D" for d in dims_eje], min_loss_dim, color='dodgerblue', edgecolor='black', width=0.4)
plt.title("Impacto del Tamaño del Bottleneck (Espacio Latente)", weight='bold')
plt.xlabel("Dimensiones del Espacio Latente")
plt.ylabel("Mínimo MSE Alcanzado")
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("8_comparativa_latent_dims.png", dpi=300)
plt.close()
print("[OK] Guardada la comparación aislada de Espacio Latente en '8_comparativa_latent_dims.png'")
