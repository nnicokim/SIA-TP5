# decoder.py
import numpy as np

class VariationalDecoder:
    def __init__(self, latent_dim=2, hidden_dims=[64, 32], output_dim=100, seed=42):
        self.latent_dim = latent_dim
        
        # Instanciamos el generador moderno
        self.rng = np.random.default_rng(seed)
        
        decoder_dims = hidden_dims[::-1]
        
        self.weights = []
        self.biases = []
        
        prev_dim = latent_dim
        for h_dim in decoder_dims:
            # Usamos standard_normal con tuplas
            W = self.rng.standard_normal((prev_dim, h_dim)) * np.sqrt(2.0 / prev_dim)
            b = np.zeros((1, h_dim))
            self.weights.append(W)
            self.biases.append(b)
            prev_dim = h_dim
            
        self.w_out = self.rng.standard_normal((prev_dim, output_dim)) * np.sqrt(2.0 / prev_dim)
        self.b_out = np.zeros((1, output_dim))

    def _relu(self, x):
        return np.maximum(0, x)

    def _sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, z):
        activations = [z]
        current_input = z
        
        for i in range(len(self.weights)):
            z_layer = np.dot(current_input, self.weights[i]) + self.biases[i]
            a_layer = self._relu(z_layer)
            activations.append(a_layer)
            current_input = a_layer
            
        x_recon_raw = np.dot(current_input, self.w_out) + self.b_out
        x_recon = self._sigmoid(x_recon_raw)
        
        cache = {
            'activations': activations,
            'x_recon_raw': x_recon_raw,
            'x_recon': x_recon
        }
        
        return x_recon, cache
    
    def backward(self, dl_dx_recon_raw, cache, learning_rate):
        """
        dl_dx_recon_raw: La chispa inicial que viene de loss.py
        """
        # Recuperamos la memoria del viaje de ida
        activations = cache['activations']
        
        # 1. Derivadas de la capa de salida
        # La activación anterior (la última capa oculta) es la penúltima de la lista
        a_prev = activations[-1]
        
        # dw = (Entrada de la capa)^T * (Error)
        dw_out = np.dot(a_prev.T, dl_dx_recon_raw)
        db_out = np.sum(dl_dx_recon_raw, axis=0, keepdims=True)
        
        # Pasamos el error hacia atrás: da = (Error) * (Pesos de salida)^T
        da = np.dot(dl_dx_recon_raw, self.w_out.T)
        
        # 2. Retropropagación por las capas ocultas dinámicas (vamos en reversa)
        for i in reversed(range(len(self.weights))):
            # Derivada de la función ReLU: Es 1 si la activación fue > 0, sino es 0
            a_current = activations[i+1]
            dz = da * (a_current > 0).astype(float)
            
            a_prev = activations[i]
            dw = np.dot(a_prev.T, dz)
            db = np.sum(dz, axis=0, keepdims=True)
            
            # Pasamos el error a la capa anterior
            da = np.dot(dz, self.weights[i].T)
            
            # --- ACTUALIZACIÓN DE PESOS (Descenso de Gradiente) ---
            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db
            
        # Actualizamos los pesos de la capa de salida al final
        self.w_out -= learning_rate * dw_out
        self.b_out -= learning_rate * db_out
        
        # Devolvemos da, que en este punto es EXACTAMENTE el error de Z (dl_dz)
        # Esto es lo que le pasaremos al Encoder.
        dl_dz = da
        return dl_dz