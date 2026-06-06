import matplotlib.pyplot as plt
import numpy as np

def plot_loss_history(history_dict, best_params):
    plt.figure(figsize=(12, 6))
    opt_key = f"Dim:{best_params['latent_dim']} | Arch:{best_params['arch']} | LR:{best_params['lr']} | Ep:{best_params['epochs']}"
    for params, loss_history in history_dict.items():
        if params == opt_key:
            plt.plot(loss_history, label="Configuración Óptima", linewidth=3, color='black', zorder=10)
        else:
            plt.plot(loss_history, alpha=0.08, linewidth=0.5)
    plt.title("Historial de Pérdidas en el Espacio de Soluciones")
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida (MSE) - Escala Logarítmica")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig("1_comparativa_hiperparametros.png", dpi=300)
    plt.close()

def plot_latent_space_2d(encoder_2d, X_train):
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

def plot_synthetic_generation(encoder, decoder, X_train, latent_dim):
    centro_latente = np.mean(encoder.predict(X_train, verbose=0), axis=0)
    punto_sintetico = centro_latente + np.random.uniform(-1.0, 1.0, size=(latent_dim,))
    nueva_letra_raw = decoder.predict(punto_sintetico.reshape(1, latent_dim), verbose=0)
    nueva_letra_img = np.round(nueva_letra_raw).reshape(7, 5)
    
    plt.figure(figsize=(3, 4))
    plt.imshow(nueva_letra_img, cmap='gray_r')
    plt.title("Nueva Letra Inventada")
    plt.axis('off')
    plt.savefig("3_nueva_letra_generada.png", dpi=300)
    plt.close()

def plot_denoising(dae_model, X_train):
    niveles_ruido = [0.10, 0.25, 0.50]
    fig, axes = plt.subplots(len(niveles_ruido), 3, figsize=(9, 9))
    for idx, p in enumerate(niveles_ruido):
        ent_r = np.copy(X_train[1:2])
        masc_r = np.random.rand(*ent_r.shape) < p
        ent_r[masc_r] = 1 - ent_r[masc_r]
        pred = np.round(dae_model.predict(ent_r, verbose=0))
        
        axes[idx, 0].imshow(X_train[1].reshape(7, 5), cmap='gray_r')
        axes[idx, 0].set_title("Original")
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(ent_r.reshape(7, 5), cmap='gray_r')
        axes[idx, 1].set_title(f"Ruido: {int(p*100)}%")
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(pred.reshape(7, 5), cmap='gray_r')
        axes[idx, 2].set_title("Reconstruido")
        axes[idx, 2].axis('off')
        
    plt.tight_layout()
    plt.savefig("4_resultado_denoising.png", dpi=300)
    plt.close()

def plot_comparative_metrics(registro_metricas):
    # ÉPOCAS
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

    # LEARNING RATES
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

    # ARQUITECTURAS
    plt.figure(figsize=(9, 5))
    archs_eje = sorted(list(set([r['arch'] for r in registro_metricas])))
    min_loss_arch = [min([r['loss'] for r in registro_metricas if r['arch'] == a]) for a in archs_eje]
    plt.barh(archs_eje, min_loss_arch, color='darkorchid', edgecolor='black', height=0.5)
    plt.title("Impacto de la Capacidad de las Capas Ocultas", weight='bold')
    plt.xlabel("Mínimo MSE Alcanzado")
    plt.ylabel("Topología de Capas (Encoder)")
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig("7_comparativa_architectures.png", dpi=300)
    plt.close()

    # LATENT DIMS
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