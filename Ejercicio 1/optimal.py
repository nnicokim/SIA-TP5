import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# =====================================================================
# 1. DEFINICIÓN DEL AUTOENCODER EN NUMPY PURO
# =====================================================================
class NumPyAutoencoder:
    def __init__(self):
        # Inicialización de pesos usando He (Kaiming) Normal para ReLU
        # Dimensiones: 100 -> 64 -> 32 -> 4 (Latente) -> 32 -> 64 -> 100
        self.W1 = np.random.randn(100, 64) * np.sqrt(2.0 / 100)
        self.b1 = np.zeros((1, 64))
        
        self.W2 = np.random.randn(64, 32) * np.sqrt(2.0 / 64)
        self.b2 = np.zeros((1, 32))
        
        self.W3 = np.random.randn(32, 4) * np.sqrt(2.0 / 32)  # Espacio Latente 4D
        self.b3 = np.zeros((1, 4))
        
        self.W4 = np.random.randn(4, 32) * np.sqrt(2.0 / 4)
        self.b4 = np.zeros((1, 32))
        
        self.W5 = np.random.randn(32, 64) * np.sqrt(2.0 / 32)
        self.b5 = np.zeros((1, 64))
        
        self.W6 = np.random.randn(64, 100) * np.sqrt(1.0 / 64) # Xavier para Sigmoid
        self.b6 = np.zeros((1, 100))
        
        # Parámetros para el optimizador Adam
        self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, 
                       self.W4, self.b4, self.W5, self.b5, self.W6, self.b6]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    # Funciones de activación y sus derivadas
    def relu(self, x): return np.maximum(0, x)
    def drelu(self, x): return (x > 0).astype(float)
    def sigmoid(self, x): return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

    def forward_encoder(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self.relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self.relu(z2)
        z3 = a2 @ self.W3 + self.b3  # Activación lineal para el bottleneck
        return z3, (X, z1, a1, z2, a2, z3)

    def forward_decoder(self, z3):
        z4 = z3 @ self.W4 + self.b4
        a4 = self.relu(z4)
        z5 = a4 @ self.W5 + self.b5
        a5 = self.relu(z5)
        z6 = a5 @ self.W6 + self.b6
        a6 = self.sigmoid(z6)
        return a6, (z4, a4, z5, a5, z6, a6)

    def train_step(self, X, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        # 1. FORWARD PASS
        latent, enc_cache = self.forward_encoder(X)
        output, dec_cache = self.forward_decoder(latent)
        
        X, z1, a1, z2, a2, z3 = enc_cache
        z4, a4, z5, a5, z6, a6 = dec_cache
        
        # Cálculo del error MSE
        loss = np.mean((a6 - X) ** 2)
        
        # 2. BACKWARD PASS (Derivadas manuales)
        da6 = 2 * (a6 - X) / a6.size
        dz6 = da6 * a6 * (1 - a6)  # Derivada Sigmoid
        dW6 = a5.T @ dz6
        db6 = np.sum(dz6, axis=0, keepdims=True)
        
        da5 = dz6 @ self.W6.T
        dz5 = da5 * self.drelu(z5)
        dW5 = a4.T @ dz5
        db5 = np.sum(dz5, axis=0, keepdims=True)
        
        da4 = dz5 @ self.W5.T
        dz4 = da4 * self.drelu(z4)
        dW4 = z3.T @ dz4
        db4 = np.sum(dz4, axis=0, keepdims=True)
        
        da3 = dz4 @ self.W4.T
        dz3 = da3 * 1.0  # Derivada Lineal
        dW3 = a2.T @ dz3
        db3 = np.sum(dz3, axis=0, keepdims=True)
        
        da2 = dz3 @ self.W3.T
        dz2 = da2 * self.drelu(z2)
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.drelu(z1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        grads = [dW1, db1, dW2, db2, dW3, db3, dW4, db4, dW5, db5, dW6, db6]
        
        # 3. OPTIMIZADOR ADAM (Actualización de pesos)
        self.t += 1
        for i in range(len(self.params)):
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * grads[i]
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * (grads[i] ** 2)
            
            m_hat = self.m[i] / (1 - beta1 ** self.t)
            v_hat = self.v[i] / (1 - beta2 ** self.t)
            
            self.params[i] -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
            
        return loss

# =====================================================================
# 2. CARGA DE DATOS (MOCK DATA)
# =====================================================================
# Reemplazá esto por tu verdadero X_train de letras (num_muestras, 100)
num_letras = 30
X_train = np.random.rand(num_letras, 100).astype(np.float32)

# =====================================================================
# 3. ENTRENAMIENTO
# =====================================================================
model = NumPyAutoencoder()
epochs = 1500

print(f"--- Iniciando Entrenamiento NumPy ({epochs} épocas) ---")
for epoch in range(epochs):
    current_loss = model.train_step(X_train, lr=0.001)
    
    if (epoch + 1) % 300 == 0:
        print(f"Época [{epoch+1}/{epochs}] - Loss MSE: {current_loss:.5f}")
print("--- Entrenamiento Finalizado ---\n")

# =====================================================================
# 4. GRÁFICO 1: MAPEO DEL ESPACIO LATENTE (4D a 2D con PCA)
# =====================================================================
# Pasamos las letras por el encoder para extraer las coordenadas 4D
latentes_4d, _ = model.forward_encoder(X_train)

pca = PCA(n_components=2)
latentes_2d = pca.fit_transform(latentes_4d)

plt.figure(figsize=(10, 8))
plt.scatter(latentes_2d[:, 0], latentes_2d[:, 1], c='royalblue', s=150, edgecolors='black')

for i in range(len(latentes_2d)):
    plt.annotate(str(i), (latentes_2d[i, 0] + 0.02, latentes_2d[i, 1] + 0.02), fontsize=10)

plt.title("Mapeo de Letras en Espacio Latente (4D reducido a 2D con PCA - NumPy)", fontsize=14)
plt.xlabel(f"Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}% varianza)")
plt.ylabel(f"Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}% varianza)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("mapa_latente_letras_4d.png", bbox_inches='tight')
plt.show()

# =====================================================================
# 5. GRÁFICO 2: GENERACIÓN DE UNA LETRA DESDE 4D
# =====================================================================
# Generamos un vector Z aleatorio 4D
z_random = np.random.randn(1, 4) 
letra_generada, _ = model.forward_decoder(z_random)
imagen_generada = letra_generada.reshape(10, 10)

plt.figure(figsize=(5, 5))
plt.imshow(imagen_generada, cmap='gray_r')
plt.title(f"Letra Generada desde 4D (NumPy)\nVector Z: [{z_random[0][0]:.2f}, {z_random[0][1]:.2f}, {z_random[0][2]:.2f}, {z_random[0][3]:.2f}]", fontsize=12)
plt.axis('off')
plt.savefig("nueva_letra_generada_4d.png", bbox_inches='tight')
plt.show()