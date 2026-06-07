import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# =====================================================================
# 1. DEFINICIÓN DEL MODELO ÓPTIMO (EJERCICIO 1)
# =====================================================================
class OptimalAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: 100 (10x10) -> 64 -> 32 -> 4 (Cuello de botella 4D)
        self.encoder = nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )
        
        # Decoder: 4 -> 32 -> 64 -> 100 (10x10)
        self.decoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 100),
            nn.Sigmoid()  # Para que los píxeles estén entre 0 y 1
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed

# =====================================================================
# 2. CARGA DE DATOS 
# =====================================================================
# IMPORTANTE: Reemplazá esto por la carga de tu verdadero dataset de letras
# Aquí creo un dataset de prueba aleatorio de 30 "letras" de 10x10 para que el código funcione.
num_letras = 30
X_train_np = np.random.rand(num_letras, 100).astype(np.float32) 
X_train = torch.tensor(X_train_np)

# =====================================================================
# 3. ENTRENAMIENTO CON PARÁMETROS ÓPTIMOS
# =====================================================================
model = OptimalAutoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # LR Óptimo

epochs = 1500  # Épocas Óptimas

print(f"--- Iniciando Entrenamiento ({epochs} épocas) ---")
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, X_train)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 300 == 0:
        print(f"Época [{epoch+1}/{epochs}] - Loss MSE: {loss.item():.5f}")

print("--- Entrenamiento Finalizado ---\n")

# =====================================================================
# 4. GRÁFICO 1: MAPEO DEL ESPACIO LATENTE (4D a 2D con PCA)
# =====================================================================
model.eval() # Modo evaluación
with torch.no_grad():
    # Obtenemos los vectores 4D de todas las letras
    latentes_4d = model.encoder(X_train).numpy()

# Usamos PCA para reducir de 4 dimensiones a 2 dimensiones visualizables
pca = PCA(n_components=2)
latentes_2d = pca.fit_transform(latentes_4d)

plt.figure(figsize=(10, 8))
plt.scatter(latentes_2d[:, 0], latentes_2d[:, 1], c='royalblue', s=150, edgecolors='black')

# Etiquetamos cada punto con el número de la letra
for i in range(len(latentes_2d)):
    plt.annotate(str(i), (latentes_2d[i, 0] + 0.02, latentes_2d[i, 1] + 0.02), fontsize=10)

plt.title("Mapeo de Letras en Espacio Latente (4D reducido a 2D con PCA)", fontsize=14)
plt.xlabel(f"Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}% varianza)")
plt.ylabel(f"Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}% varianza)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("mapa_latente_letras_4d.png", bbox_inches='tight')

# =====================================================================
# 5. GRÁFICO 2: GENERACIÓN DE UNA LETRA DESDE 4D
# =====================================================================
with torch.no_grad():
    # Creamos un tensor aleatorio de tamaño (1, 4) para inyectar en el Decoder.
    # Los valores se generan con una distribución normal estándar.
    z_random = torch.randn(1, 4) 
    
    # Pasamos el vector 4D por el decoder
    letra_generada = model.decoder(z_random)
    
    # Reformateamos el vector de 100 a una imagen de 10x10
    imagen_generada = letra_generada.view(10, 10).numpy()

plt.figure(figsize=(5, 5))
plt.imshow(imagen_generada, cmap='gray_r') # gray_r para que el fondo sea blanco y trazos negros
plt.title(f"Letra Generada desde 4D\nVector Z: [{z_random[0][0]:.2f}, {z_random[0][1]:.2f}, {z_random[0][2]:.2f}, {z_random[0][3]:.2f}]", fontsize=12)
plt.axis('off')
plt.savefig("nueva_letra_generada_4d.png", bbox_inches='tight')
