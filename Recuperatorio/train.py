# train.py
import numpy as np
from data import get_dataset
from encoder import VariationalEncoder
from decoder import VariationalDecoder
from loss import calculate_total_loss, bce_derivative, kl_derivatives
import os
import ast
from dotenv import load_dotenv

# Cargamos el archivo .env
load_dotenv()

# Leemos las variables como strings y las transformamos a listas de Python con ast.literal_eval
arquitecturas = ast.literal_eval(os.getenv("ARCH_CONFIGS"))
learning_rates = ast.literal_eval(os.getenv("LR_CONFIGS"))
epocas_list = ast.literal_eval(os.getenv("EPOCHS_CONFIGS"))
latent_dims = ast.literal_eval(os.getenv("LATENT_DIMS_CONFIGS"))
lambdas = ast.literal_eval(os.getenv("LAMBDA_CONFIGS"))

# Estos son números simples, así que solo usamos int()
batch_size = int(os.getenv("BATCH_SIZE"))
seed = int(os.getenv("SEED"))

def main():
    # 1. Cargamos nuestros 32 Kanjis
    X = get_dataset()
    
    # 2. Definimos las listas para el Grid Search
    arquitecturas = [[64, 32], [128, 64, 32]]
    learning_rates = [0.01, 0.005]
    epocas = 1000
    lambda_reg = 1.0  # El peso de la penalización KL
    
    mejor_loss = float('inf')
    mejores_hiperparametros = None
    
    # 3. Bucle de Grid Search
    for arch in arquitecturas:
        for lr in learning_rates:
            print(f"\n--- Entrenando con Arquitectura: {arch} | LR: {lr} ---")
            
            # Instanciamos una red nueva desde cero para esta prueba
            encoder = VariationalEncoder(input_dim=100, hidden_dims=arch, latent_dim=2)
            decoder = VariationalDecoder(latent_dim=2, hidden_dims=arch, output_dim=100)
            
            # Bucle de Entrenamiento (Épocas)
            for epoch in range(epocas):
                
                # --- A. FORWARD PASS (Hacia adelante) ---
                z, mu, log_var, cache_enc = encoder.forward(X)
                x_recon, cache_dec = decoder.forward(z)
                
                # --- B. CÁLCULO DE PÉRDIDA (El Juez) ---
                loss, bce, kl = calculate_total_loss(X, x_recon, mu, log_var, lambda_reg)
                
                # --- C. BACKWARD PASS (Hacia atrás / Aprendizaje) ---
                # Calculamos las chispas iniciales
                dl_dxrecon = bce_derivative(X, x_recon)
                dkl_dmu, dkl_dlogvar = kl_derivatives(mu, log_var)
                
                # El Decoder viaja hacia atrás y le pasa el error de Z al Encoder
                dl_dz = decoder.backward(dl_dxrecon, cache_dec, lr)
                
                # El Encoder viaja hacia atrás actualizando sus pesos
                encoder.backward(dl_dz, dkl_dmu, dkl_dlogvar, cache_enc, lr)
                
                # Imprimimos el progreso cada 200 épocas para no saturar la consola
                if epoch % 200 == 0:
                    print(f"  Época {epoch:4d} | Loss Total: {loss:.4f} (BCE: {bce:.4f}, KL: {kl:.4f})")
            
            # 4. Al terminar las épocas de esta configuración, evaluamos si es la mejor
            if loss < mejor_loss:
                mejor_loss = loss
                mejores_hiperparametros = {'arch': arch, 'lr': lr}
                print(f"-> ¡Nuevo mejor modelo encontrado! Loss final: {mejor_loss:.4f}")

    # 5. Resultados finales
    print("\n==================================================")
    print("RESULTADOS DEL GRID SEARCH:")
    print(f"Mejor Arquitectura : {mejores_hiperparametros['arch']}")
    print(f"Mejor Learning Rate: {mejores_hiperparametros['lr']}")
    print(f"Menor Loss Obtenida: {mejor_loss:.4f}")
    print("==================================================")

if __name__ == "__main__":
    main()