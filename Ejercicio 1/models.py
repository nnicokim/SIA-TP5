import numpy as np

# =====================================================================
# MOTOR INTERNO: AUTOENCODER DINÁMICO EN NUMPY PURO
# =====================================================================
class DynamicNumPyAutoencoder:
    def __init__(self, input_dim, hidden_layers, latent_dim, learning_rate=0.001):
        self.lr = learning_rate
        
        # Definimos las dimensiones de cada paso
        enc_sizes = [input_dim] + list(hidden_layers) + [latent_dim]
        dec_sizes = [latent_dim] + list(reversed(hidden_layers)) + [input_dim]
        
        # Inicialización de pesos y bias para el Encoder
        self.enc_weights = []
        self.enc_biases = []
        for i in range(len(enc_sizes) - 1):
            w = np.random.randn(enc_sizes[i], enc_sizes[i+1]) * np.sqrt(2.0 / enc_sizes[i])
            b = np.zeros((1, enc_sizes[i+1]))
            self.enc_weights.append(w)
            self.enc_biases.append(b)
            
        # Inicialización de pesos y bias para el Decoder
        self.dec_weights = []
        self.dec_biases = []
        for i in range(len(dec_sizes) - 1):
            factor = 1.0 if i == len(dec_sizes) - 2 else 2.0  # Xavier para la última (Sigmoid)
            w = np.random.randn(dec_sizes[i], dec_sizes[i+1]) * np.sqrt(factor / dec_sizes[i])
            b = np.zeros((1, dec_sizes[i+1]))
            self.dec_weights.append(w)
            self.dec_biases.append(b)
            
        # Listas unificadas para el Optimizador Adam
        self.all_weights = self.enc_weights + self.dec_weights
        self.all_biases = self.enc_biases + self.dec_biases
        
        self.m_w = [np.zeros_like(w) for w in self.all_weights]
        self.v_w = [np.zeros_like(w) for w in self.all_weights]
        self.m_b = [np.zeros_like(b) for b in self.all_biases]
        self.v_b = [np.zeros_like(b) for b in self.all_biases]
        self.t = 0

    def relu(self, x): return np.maximum(0, x)
    def sigmoid(self, x): return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

    def _forward_encoder(self, X):
        a = X
        enc_a, enc_z = [a], []
        for i in range(len(self.enc_weights)):
            z = a @ self.enc_weights[i] + self.enc_biases[i]
            enc_z.append(z)
            a = z if i == len(self.enc_weights) - 1 else self.relu(z)
            enc_a.append(a)
        return a, (enc_z, enc_a)

    def _forward_decoder(self, latent):
        a = latent
        dec_a, dec_z = [a], []
        for i in range(len(self.dec_weights)):
            z = a @ self.dec_weights[i] + self.dec_biases[i]
            dec_z.append(z)
            a = self.sigmoid(z) if i == len(self.dec_weights) - 1 else self.relu(z)
            dec_a.append(a)
        return a, (dec_z, dec_a)

    def train_step(self, X):
        # 1. Forward Pass completo
        latent, (enc_z, enc_a) = self._forward_encoder(X)
        output, (dec_z, dec_a) = self._forward_decoder(latent)
        
        loss = np.mean((output - X) ** 2)
        
        # 2. Backward Pass Dinámico (Derivadas encadenadas en reversa)
        dw_enc, db_enc = [None]*len(self.enc_weights), [None]*len(self.enc_biases)
        dw_dec, db_dec = [None]*len(self.dec_weights), [None]*len(self.dec_biases)
        
        # Gradiente inicial de la pérdida MSE
        da = 2 * (output - X) / output.size
        
        # Backprop del Decoder
        for i in reversed(range(len(self.dec_weights))):
            z = dec_z[i]
            a_prev = dec_a[i]
            dz = da * dec_a[i+1] * (1 - dec_a[i+1]) if i == len(self.dec_weights) - 1 else da * (z > 0)
            
            dw_dec[i] = a_prev.T @ dz
            db_dec[i] = np.sum(dz, axis=0, keepdims=True)
            da = dz @ self.dec_weights[i].T
            
        # Backprop del Encoder
        for i in reversed(range(len(self.enc_weights))):
            z = enc_z[i]
            a_prev = enc_a[i]
            dz = da * 1.0 if i == len(self.enc_weights) - 1 else da * (z > 0)
            
            dw_enc[i] = a_prev.T @ dz
            db_enc[i] = np.sum(dz, axis=0, keepdims=True)
            if i > 0: da = dz @ self.enc_weights[i].T
            
        # 3. Actualización de Parámetros (Adam Optimizer)
        self.t += 1
        all_dw = dw_enc + dw_dec
        all_db = db_enc + db_dec
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        
        for i in range(len(self.all_weights)):
            # Pesos
            self.m_w[i] = beta1 * self.m_w[i] + (1 - beta1) * all_dw[i]
            self.v_w[i] = beta2 * self.v_w[i] + (1 - beta2) * (all_dw[i] ** 2)
            self.all_weights[i] -= self.lr * (self.m_w[i] / (1 - beta1**self.t)) / (np.sqrt(self.v_w[i] / (1 - beta2**self.t)) + epsilon)
            # Bias
            self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * all_db[i]
            self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * (all_db[i] ** 2)
            self.all_biases[i] -= self.lr * (self.m_b[i] / (1 - beta1**self.t)) / (np.sqrt(self.v_b[i] / (1 - beta2**self.t)) + epsilon)
            
        return loss

# =====================================================================
# WRAPPERS "ESPEJO" PARA PASAR DESAPERCEBIDO ANTE EL SCRIPT ORIGINAL
# =====================================================================
class KerasHistoryMimic:
    def __init__(self, losses): 
        self.history = {'loss': losses}

class AutoencoderWrapper:
    def __init__(self, core): 
        self.core = core
        
    def fit(self, X, y, epochs=1, batch_size=None, verbose=0, callbacks=None):
        losses = []
        for epoch in range(epochs):
            # Ejecuta el paso de entrenamiento en NumPy
            loss = self.core.train_step(X)
            losses.append(loss)
            
            # Si le pasaste tu ProgressCallback, lo dispara al final de la época
            if callbacks:
                for callback in callbacks:
                    if hasattr(callback, 'on_epoch_end'):
                        callback.on_epoch_end(epoch, logs={'loss': loss})
                        
        # Imprime un salto de línea al final del entrenamiento para que no se pisen los textos en consola
        print() 
        return KerasHistoryMimic(losses)

    # <<-- AGREGADO CLAVE: Para poder hacer la reconstrucción directa desde el autoencoder completo
    def predict(self, X, verbose=0):
        latent, _ = self.core._forward_encoder(X)
        output, _ = self.core._forward_decoder(latent)
        return output

class SubModelWrapper:
    def __init__(self, predict_fn): 
        self.predict_fn = predict_fn
        
    def predict(self, X, verbose=0): 
        return self.predict_fn(X)

# =====================================================================
# FUNCTION REEMPLAZO (TU NUEVA FUNCIÓN GANADORA)
# =====================================================================
def build_autoencoder(input_dim, hidden_layers, latent_dim, learning_rate=0.001):
    # Instanciamos el motor en NumPy
    core_model = DynamicNumPyAutoencoder(input_dim, hidden_layers, latent_dim, learning_rate)
    
    # Envolvemos los componentes para simular los objetos de Keras (.fit, .predict)
    autoencoder = AutoencoderWrapper(core_model)
    encoder = SubModelWrapper(lambda X: core_model._forward_encoder(X)[0])
    decoder = SubModelWrapper(lambda Z: core_model._forward_decoder(Z)[0])
    
    return autoencoder, encoder, decoder