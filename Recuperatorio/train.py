# train.py
import numpy as np
import os
import ast
import json
from dotenv import load_dotenv

from data import get_dataset
from encoder import VariationalEncoder
from decoder import VariationalDecoder
from loss import calculate_total_loss, bce_derivative, kl_derivatives

def main():
    # 1. Cargamos el archivo .env y leemos las variables
    load_dotenv()

    arquitecturas = ast.literal_eval(os.getenv("ARCH_CONFIGS"))
    learning_rates = ast.literal_eval(os.getenv("LR_CONFIGS"))
    epocas_list = ast.literal_eval(os.getenv("EPOCHS_CONFIGS"))
    latent_dims = ast.literal_eval(os.getenv("LATENT_DIMS_CONFIGS"))
    lambdas = ast.literal_eval(os.getenv("LAMBDA_CONFIGS"))

    seed = int(os.getenv("SEED"))

    # 2. Cargamos nuestros 32 Kanjis
    X = get_dataset()
    
    mejor_loss = float('inf')
    mejores_hiperparametros = None
    
    # NUEVO: Lista para almacenar el historial de todas las pruebas
    historial_resultados = []
    
    # 3. MEGA BUCLE DE GRID SEARCH
    for arch in arquitecturas:
        for lr in learning_rates:
            for epocas in epocas_list:
                for l_dim in latent_dims:
                    for lambda_reg in lambdas:
                        print(f"\n--- Probando | Arch: {arch} | LR: {lr} | Épocas: {epocas} | Z-Dim: {l_dim} | Lambda: {lambda_reg} ---")
                        
                        encoder = VariationalEncoder(input_dim=100, hidden_dims=arch, latent_dim=l_dim, seed=seed)
                        decoder = VariationalDecoder(latent_dim=l_dim, hidden_dims=arch, output_dim=100, seed=seed)
                        
                        for epoch in range(epocas):
                            # Forward, Pérdida y Backward
                            z, mu, log_var, cache_enc = encoder.forward(X)
                            x_recon, cache_dec = decoder.forward(z)
                            
                            loss, bce, kl = calculate_total_loss(X, x_recon, mu, log_var, lambda_reg)
                            
                            dl_dxrecon = bce_derivative(X, x_recon)
                            dkl_dmu, dkl_dlogvar = kl_derivatives(mu, log_var)
                            
                            dl_dz = decoder.backward(dl_dxrecon, cache_dec, lr)
                            encoder.backward(dl_dz, dkl_dmu, dkl_dlogvar, cache_enc, lr)
                            
                            if epoch > 0 and epoch % (epocas // 5) == 0:
                                print(f"  Época {epoch:4d} | Loss Total: {loss:.4f}")
                        
                        # NUEVO: Guardamos el resultado final de esta configuración
                        # Convertimos los valores a tipos estándar de Python para que JSON no de error
                        historial_resultados.append({
                            'arquitectura': str(arch),
                            'learning_rate': float(lr),
                            'epocas': int(epocas),
                            'latent_dim': int(l_dim),
                            'lambda': float(lambda_reg),
                            'loss_final': float(loss),
                            'bce_final': float(bce),
                            'kl_final': float(kl)
                        })

                        if loss < mejor_loss:
                            mejor_loss = loss
                            mejores_hiperparametros = {
                                'arch': arch, 
                                'lr': lr, 
                                'epocas': epocas, 
                                'latent_dim': l_dim, 
                                'lambda': lambda_reg
                            }
                            print(f"-> ¡NUEVO MEJOR MODELO! Loss final: {mejor_loss:.4f}")

    # NUEVO: Al terminar todos los bucles, exportamos el historial a un archivo JSON
    with open('resultados_grid_search.json', 'w') as f:
        json.dump(historial_resultados, f, indent=4)
    print("\n[INFO] Todos los resultados fueron guardados en 'resultados_grid_search.json'")

    # 5. Resultados finales por consola
    print("\n" + "="*50)
    print("RESULTADOS DEL GRID SEARCH:")
    print(f"Mejor Arquitectura : {mejores_hiperparametros['arch']}")
    print(f"Mejor Learning Rate: {mejores_hiperparametros['lr']}")
    print(f"Mejor Cant. Épocas : {mejores_hiperparametros['epocas']}")
    print(f"Mejor Dim. Latente : {mejores_hiperparametros['latent_dim']}")
    print(f"Mejor Lambda (KL)  : {mejores_hiperparametros['lambda']}")
    print(f"MENOR LOSS OBTENIDA: {mejor_loss:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()