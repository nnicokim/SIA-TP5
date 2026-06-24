# plot.py
import json
import matplotlib.pyplot as plt

def main():
    # 1. Cargamos el archivo generado por train.py
    try:
        with open('resultados_grid_search.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: No se encontró el archivo 'resultados_grid_search.json'. ¡Corré train.py primero!")
        return

    # 2. Definimos tus valores fijos (Baseline)
    base_lr = 0.001
    base_epocas = 1500
    base_lambda = 0.1
    base_arch = "[64, 32]"  # Recordá que las listas se guardaron como strings

    # =====================================================================
    # GRÁFICO 1: Variando Arquitecturas (Gráfico de Barras)
    # Fijos: LR, Épocas, Lambda
    # =====================================================================
    datos_arch = [d for d in data if d['learning_rate'] == base_lr and d['epocas'] == base_epocas and d['lambda'] == base_lambda]
    
    if datos_arch:
        # Ordenamos de peor a mejor loss
        datos_arch = sorted(datos_arch, key=lambda x: x['loss_final'], reverse=True)
        x_arch = [d['arquitectura'] for d in datos_arch]
        y_arch = [d['loss_final'] for d in datos_arch]

        plt.figure(figsize=(10, 5))
        plt.bar(x_arch, y_arch, color='skyblue', edgecolor='black')
        plt.title(f"Comparación de Arquitecturas\n(Fijos: LR={base_lr}, Épocas={base_epocas}, Lambda={base_lambda})", fontweight='bold')
        plt.xlabel("Arquitecturas (Capas Ocultas)")
        plt.ylabel("Pérdida Total (Loss Final)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("grafico_1_arquitecturas.png")
        plt.show()

    # =====================================================================
    # GRÁFICO 2: Variando Learning Rate (Gráfico de Líneas)
    # Fijos: Arquitectura, Épocas, Lambda
    # =====================================================================
    datos_lr = [d for d in data if d['arquitectura'] == base_arch and d['epocas'] == base_epocas and d['lambda'] == base_lambda]
    
    if datos_lr:
        datos_lr = sorted(datos_lr, key=lambda x: x['learning_rate'])
        x_lr = [str(d['learning_rate']) for d in datos_lr] # Lo pasamos a str para que el eje X espacie parejo
        y_lr = [d['loss_final'] for d in datos_lr]

        plt.figure(figsize=(10, 5))
        plt.plot(x_lr, y_lr, marker='o', color='crimson', linewidth=2, markersize=8)
        plt.title(f"Efecto del Learning Rate\n(Fijos: Arch={base_arch}, Épocas={base_epocas}, Lambda={base_lambda})", fontweight='bold')
        plt.xlabel("Learning Rate")
        plt.ylabel("Pérdida Total (Loss Final)")
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("grafico_2_learning_rate.png")
        plt.show()

    # =====================================================================
    # GRÁFICO 3: Variando Cantidad de Épocas (Gráfico de Líneas)
    # Fijos: Arquitectura, LR, Lambda
    # =====================================================================
    datos_epocas = [d for d in data if d['arquitectura'] == base_arch and d['learning_rate'] == base_lr and d['lambda'] == base_lambda]
    
    if datos_epocas:
        datos_epocas = sorted(datos_epocas, key=lambda x: x['epocas'])
        x_epocas = [d['epocas'] for d in datos_epocas]
        y_epocas = [d['loss_final'] for d in datos_epocas]

        plt.figure(figsize=(10, 5))
        plt.plot(x_epocas, y_epocas, marker='s', color='forestgreen', linewidth=2, markersize=8)
        plt.title(f"Evolución según Cantidad de Épocas\n(Fijos: Arch={base_arch}, LR={base_lr}, Lambda={base_lambda})", fontweight='bold')
        plt.xlabel("Cantidad de Épocas")
        plt.ylabel("Pérdida Total (Loss Final)")
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("grafico_3_epocas.png")
        plt.show()

    # =====================================================================
    # GRÁFICO 4: Variando Penalización KL (Lambda) (Gráfico de Líneas)
    # Fijos: Arquitectura, LR, Épocas
    # =====================================================================
    datos_lambda = [d for d in data if d['arquitectura'] == base_arch and d['learning_rate'] == base_lr and d['epocas'] == base_epocas]
    
    if datos_lambda:
        datos_lambda = sorted(datos_lambda, key=lambda x: x['lambda'])
        x_lambda = [str(d['lambda']) for d in datos_lambda]
        y_lambda = [d['loss_final'] for d in datos_lambda]

        plt.figure(figsize=(10, 5))
        plt.plot(x_lambda, y_lambda, marker='D', color='darkorange', linewidth=2, markersize=8)
        plt.title(f"Impacto del Peso KL (Lambda)\n(Fijos: Arch={base_arch}, LR={base_lr}, Épocas={base_epocas})", fontweight='bold')
        plt.xlabel("Valor de Lambda")
        plt.ylabel("Pérdida Total (Loss Final)")
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("grafico_4_lambda.png")
        plt.show()

if __name__ == "__main__":
    main()