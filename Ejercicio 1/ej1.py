import os
import sys
import warnings
import itertools
import numpy as np

# Configuración inicial y silenciamiento de TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore', category=UserWarning)

import tensorflow as tf

# Importación de nuestros propios módulos
from config import load_config
from data_loader import cargar_datos_font
from models import build_autoencoder
from utils import ProgressCallback
import visualization as vis

# 1. Carga de datos y variables
latent_dims, arch_configs, lr_configs, epochs_configs, batch_size = load_config()
X_train = cargar_datos_font("font.h")
input_dim = X_train.shape[1]

# 2. Inicialización de variables de métricas
best_loss = float('inf')
best_params = None
best_models = None
best_2d_models = None
best_2d_loss = float('inf')
history_dict = {}
registro_metricas = []
total_combinaciones = len(latent_dims) * len(arch_configs) * len(lr_configs) * len(epochs_configs)

# 3. Grid Search Principal
print(f"\n--- Iniciando Grid Search Clásico ({total_combinaciones} combinaciones) ---")
contador = 0

for lat_dim, arch, lr, ep in itertools.product(latent_dims, arch_configs, lr_configs, epochs_configs):
    contador += 1
    param_str = f"Dim:{lat_dim} | Arch:{arch} | LR:{lr}"
    
    ae, enc, dec = build_autoencoder(input_dim, arch, lat_dim, lr)
    spinner_cb = ProgressCallback(contador, total_combinaciones, param_str, ep)
    
    history = ae.fit(X_train, X_train, epochs=ep, batch_size=batch_size, verbose=0, callbacks=[spinner_cb])
    final_loss = history.history['loss'][-1]
    if np.isnan(final_loss): final_loss = 1.0
        
    history_dict[f"{param_str} | Ep:{ep}"] = history.history['loss']
    registro_metricas.append({'latent_dim': lat_dim, 'arch': str(arch), 'lr': lr, 'epochs': ep, 'loss': final_loss})
    
    if final_loss < best_loss:
        best_loss = final_loss
        best_params = {'latent_dim': lat_dim, 'arch': arch, 'lr': lr, 'epochs': ep}
        best_models = (ae, enc, dec)
        
    if lat_dim == 2 and final_loss < best_2d_loss:
        best_2d_loss = final_loss
        best_2d_models = (ae, enc, dec)
        
    tf.keras.backend.clear_session()

# Limpieza visual de terminal
sys.stdout.write("\r" + " " * 100 + "\r")
sys.stdout.flush()

print(f"\n--- ¡VALORES ÓPTIMOS ENCONTRADOS! ---")
print(f"-> Espacio Latente: {best_params['latent_dim']}D")
print(f"-> Capas Ocultas: {best_params['arch']}")
print(f"-> LR: {best_params['lr']} | Épocas: {best_params['epochs']}")
print(f"-> Pérdida Mínima (MSE): {best_loss:.5f}\n")

# 4. Generación de Gráficos (Delegados al módulo)
print("Generando archivos de gráficos...")
autoencoder_opt, encoder_opt, decoder_opt = best_models
_, encoder_2d, _ = best_2d_models

vis.plot_loss_history(history_dict, best_params)
vis.plot_latent_space_2d(encoder_2d, X_train)
vis.plot_synthetic_generation(encoder_opt, decoder_opt, X_train, best_params['latent_dim'])

# Para Denoising, entrenamos un modelo específico con el setup óptimo y ruido aplicado
dae, _, _ = build_autoencoder(input_dim, best_params['arch'], best_params['latent_dim'], best_params['lr'])
X_train_ruido = np.copy(X_train)
mascara = np.random.rand(*X_train.shape) < 0.15
X_train_ruido[mascara] = 1 - X_train_ruido[mascara]
dae.fit(X_train_ruido, X_train, epochs=best_params['epochs'], batch_size=batch_size, verbose=0)
vis.plot_denoising(dae, X_train)

# Gráficos comparativos independientes
vis.plot_comparative_metrics(registro_metricas)

print("¡Ejecución modular terminada! Revisá tu directorio para ver los 8 gráficos.")