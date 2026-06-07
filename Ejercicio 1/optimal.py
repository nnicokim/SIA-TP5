import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# =====================================================================
# 1. DEFINICIÓN DEL MODELO ÓPTIMO EN KERAS
# =====================================================================
input_img = layers.Input(shape=(100,))
# Encoder
encoded = layers.Dense(64, activation='relu')(input_img)
encoded = layers.Dense(32, activation='relu')(encoded)
latent_space = layers.Dense(4, activation='linear')(encoded) # Bottleneck 4D

# Decoder
decoded = layers.Dense(32, activation='relu')(latent_space)
decoded = layers.Dense(64, activation='relu')(decoded)
output_img = layers.Dense(100, activation='sigmoid')(decoded)

# Modelos completos y divididos
autoencoder = models.Model(input_img, output_img)
encoder = models.Model(input_img, latent_space)

# Creamos un decoder aislado para generar nuevas letras luego
decoder_input = layers.Input(shape=(4,))
deco_layer1 = autoencoder.layers[-3](decoder_input)
deco_layer2 = autoencoder.layers[-2](deco_layer1)
deco_output = autoencoder.layers[-1](deco_layer2)
decoder = models.Model(decoder_input, deco_output)

autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# =====================================================================
# 2. CARGA DE DATOS 
# =====================================================================
# IMPORTANTE: Reemplazá esto por tu verdadero X_train
num_letras = 30
X_train = np.random.rand(num_letras, 100).astype(np.float32)

# =====================================================================
# 3. ENTRENAMIENTO
# =====================================================================
epochs = 1500
print(f"--- Iniciando Entrenamiento ({epochs} épocas) ---")
# El verbose=0 evita que te imprima 1500 líneas en la consola
history = autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=len(X_train), verbose=0) 
print(f"Loss Final (MSE): {history.history['loss'][-1]:.5f}")
print("--- Entrenamiento Finalizado ---\n")

# =====================================================================
# 4. GRÁFICO 1: MAPEO DEL ESPACIO LATENTE (4D a 2D con PCA)
# =====================================================================
latentes_4d = encoder.predict(X_train, verbose=0)

pca = PCA(n_components=2)
latentes_2d = pca.fit_transform(latentes_4d)

plt.figure(figsize=(10, 8))
plt.scatter(latentes_2d[:, 0], latentes_2d[:, 1], c='royalblue', s=150, edgecolors='black')

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
# Generamos un vector Z aleatorio 4D
z_random = np.random.randn(1, 4) 
letra_generada = decoder.predict(z_random, verbose=0)
imagen_generada = letra_generada.reshape(10, 10)

plt.figure(figsize=(5, 5))
plt.imshow(imagen_generada, cmap='gray_r')
plt.title(f"Letra Generada desde 4D\nVector Z: [{z_random[0][0]:.2f}, {z_random[0][1]:.2f}, {z_random[0][2]:.2f}, {z_random[0][3]:.2f}]", fontsize=12)
plt.axis('off')
plt.savefig("nueva_letra_generada_4d.png", bbox_inches='tight')
