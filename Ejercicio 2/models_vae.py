import numpy as np

# =====================================================================
# MOTOR INTERNO: VARIATIONAL AUTOENCODER (VAE) EN NUMPY PURO
# =====================================================================
class DynamicNumPyVAE:
    def __init__(self, input_dim, hidden_layers, latent_dim, learning_rate=0.001, kl_weight=0.01):
        self.lr = learning_rate
        self.latent_dim = latent_dim
        # Peso beta de la divergencia KL (antes hardcodeado en 0.01).
        # Es el hiperparametro que controla el posterior collapse: beta alto -> KL -> 0.
        self.kl_weight = kl_weight
        
        # Capas ocultas del Encoder y Decoder
        enc_sizes = [input_dim] + list(hidden_layers)
        dec_sizes = [latent_dim] + list(reversed(hidden_layers)) + [input_dim]
        
        # 1. Inicialización Encoder (Capas ocultas)
        self.enc_weights = []
        self.enc_biases = []
        for i in range(len(enc_sizes) - 1):
            w = np.random.randn(enc_sizes[i], enc_sizes[i+1]) * np.sqrt(2.0 / enc_sizes[i])
            b = np.zeros((1, enc_sizes[i+1]))
            self.enc_weights.append(w)
            self.enc_biases.append(b)
            
        # 2. Hojas de salida del Encoder (z_mean y z_log_var)
        self.W_mean = np.random.randn(enc_sizes[-1], latent_dim) * np.sqrt(2.0 / enc_sizes[-1])
        self.b_mean = np.zeros((1, latent_dim))
        
        self.W_log_var = np.random.randn(enc_sizes[-1], latent_dim) * np.sqrt(2.0 / enc_sizes[-1])
        self.b_log_var = np.zeros((1, latent_dim))
        
        # 3. Inicialización Decoder
        self.dec_weights = []
        self.dec_biases = []
        for i in range(len(dec_sizes) - 1):
            factor = 1.0 if i == len(dec_sizes) - 2 else 2.0  # Xavier para la salida Sigmoid
            w = np.random.randn(dec_sizes[i], dec_sizes[i+1]) * np.sqrt(factor / dec_sizes[i])
            b = np.zeros((1, dec_sizes[i+1]))
            self.dec_weights.append(w)
            self.dec_biases.append(b)
            
        # Unificación de todos los parámetros para el optimizador Adam
        self.all_weights = self.enc_weights + [self.W_mean, self.W_log_var] + self.dec_weights
        self.all_biases = self.enc_biases + [self.b_mean, self.b_log_var] + self.dec_biases
        
        self.m_w = [np.zeros_like(w) for w in self.all_weights]
        self.v_w = [np.zeros_like(w) for w in self.all_weights]
        self.m_b = [np.zeros_like(b) for b in self.all_biases]
        self.v_b = [np.zeros_like(b) for b in self.all_biases]
        self.t = 0

    def relu(self, x): return np.maximum(0, x)
    def sigmoid(self, x): return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

    def _forward_encoder(self, X):
        a = X
        enc_z, enc_a = [], [a]
        for i in range(len(self.enc_weights)):
            z_layer = a @ self.enc_weights[i] + self.enc_biases[i]
            enc_z.append(z_layer)
            a = self.relu(z_layer)
            enc_a.append(a)
            
        # Proyección al espacio latente distribucional
        z_mean = a @ self.W_mean + self.b_mean
        z_log_var = a @ self.W_log_var + self.b_log_var
        
        # Truco de Reparametrización: z = mu + sigma * epsilon
        epsilon = np.random.normal(size=z_mean.shape)
        z = z_mean + np.exp(0.5 * z_log_var) * epsilon
        
        return z_mean, z_log_var, z, epsilon, enc_z, enc_a

    def _forward_decoder(self, z):
        a = z
        dec_z, dec_a = [], [a]
        for i in range(len(self.dec_weights)):
            z_layer = a @ self.dec_weights[i] + self.dec_biases[i]
            dec_z.append(z_layer)
            a = self.sigmoid(z_layer) if i == len(self.dec_weights) - 1 else self.relu(z_layer)
            dec_a.append(a)
        return a, dec_z, dec_a

    def train_step(self, X):
        # --- 1. FORWARD PASS ---
        z_mean, z_log_var, z, epsilon, enc_z, enc_a = self._forward_encoder(X)
        output, dec_z, dec_a = self._forward_decoder(z)
        
        # Pérdida 1: Reconstrucción (Binary Crossentropy replicando Keras reduce_sum)
        output_clipped = np.clip(output, 1e-12, 1.0 - 1e-12)
        bce_elementwise = - (X * np.log(output_clipped) + (1.0 - X) * np.log(1.0 - output_clipped))
        reconstruction_loss = np.sum(np.mean(bce_elementwise, axis=1))
        
        # Pérdida 2: Divergencia KL sumada
        kl_elementwise = -0.5 * (1.0 + z_log_var - np.square(z_mean) - np.exp(z_log_var))
        kl_loss = np.sum(kl_elementwise)
        
        total_loss = reconstruction_loss + (self.kl_weight * kl_loss)

        # --- 2. BACKWARD PASS (Gradientes del VAE) ---
        input_dim = X.shape[1]
        # Gradiente directo de la combinación BCE + Salida Sigmoid (dividido la dimensión)
        da = (output - X) / input_dim 
        
        dw_dec, db_dec = [None]*len(self.dec_weights), [None]*len(self.dec_biases)
        for i in reversed(range(len(self.dec_weights))):
            z_layer = dec_z[i]
            a_prev = dec_a[i]
            dz = da if i == len(self.dec_weights) - 1 else da * (z_layer > 0)
            
            dw_dec[i] = a_prev.T @ dz
            db_dec[i] = np.sum(dz, axis=0, keepdims=True)
            da = dz @ self.dec_weights[i].T
            
        # da ahora representa el gradiente que llegó al vector latente 'z' (da_z)
        da_z = da
        
        # Gradientes hacia las ramas distribucionales (Combinando KL y el paso del Decoder)
        beta = self.kl_weight
        d_z_mean = beta * z_mean + da_z
        d_z_log_var = 0.5 * beta * (np.exp(z_log_var) - 1.0) + da_z * (0.5 * np.exp(0.5 * z_log_var) * epsilon)
        
        # Gradientes de los pesos de las ramas
        a_enc_last = enc_a[-1]
        dW_mean = a_enc_last.T @ d_z_mean
        db_mean = np.sum(d_z_mean, axis=0, keepdims=True)
        
        dW_log_var = a_enc_last.T @ d_z_log_var
        db_log_var = np.sum(d_z_log_var, axis=0, keepdims=True)
        
        # Unimos el gradiente que regresa al cuerpo del encoder
        da = d_z_mean @ self.W_mean.T + d_z_log_var @ self.W_log_var.T
        
        # Backprop a través de las capas ocultas del Encoder
        dw_enc, db_enc = [None]*len(self.enc_weights), [None]*len(self.enc_biases)
        for i in reversed(range(len(self.enc_weights))):
            z_layer = enc_z[i]
            a_prev = enc_a[i]
            dz = da * (z_layer > 0)
            
            dw_enc[i] = a_prev.T @ dz
            db_enc[i] = np.sum(dz, axis=0, keepdims=True)
            da = dz @ self.enc_weights[i].T
            
        # --- 3. OPTIMIZADOR ADAM ---
        self.t += 1
        all_dw = dw_enc + [dW_mean, dW_log_var] + dw_dec
        all_db = db_enc + [db_mean, db_log_var] + db_dec
        beta1, beta2, epsilon_adam = 0.9, 0.999, 1e-8
        
        for i in range(len(self.all_weights)):
            # Pesos
            self.m_w[i] = beta1 * self.m_w[i] + (1 - beta1) * all_dw[i]
            self.v_w[i] = beta2 * self.v_w[i] + (1 - beta2) * (all_dw[i] ** 2)
            m_hat = self.m_w[i] / (1 - beta1**self.t)
            v_hat = self.v_w[i] / (1 - beta2**self.t)
            self.all_weights[i] -= self.lr * m_hat / (np.sqrt(v_hat) + epsilon_adam)
            # Biases
            self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * all_db[i]
            self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * (all_db[i] ** 2)
            m_hat_b = self.m_b[i] / (1 - beta1**self.t)
            v_hat_b = self.v_b[i] / (1 - beta2**self.t)
            self.all_biases[i] -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + epsilon_adam)
            
        return total_loss, reconstruction_loss, kl_loss

# =====================================================================
# WRAPPERS "ESPEJO" PARA EL MUNDO EXTERIOR
# =====================================================================
class KerasVAEHistoryMimic:
    def __init__(self, t_loss, r_loss, k_loss):
        self.history = {
            'loss': t_loss,
            'reconstruction_loss': r_loss,
            'kl_loss': k_loss
        }

class VAEWrapper:
    def __init__(self, core):
        self.core = core
    def fit(self, X, y=None, epochs=1, batch_size=None, verbose=0, callbacks=None):
        t_losses, r_losses, k_losses = [], [], []
        for epoch in range(epochs):
            t_l, r_l, k_l = self.core.train_step(X)
            t_losses.append(t_l)
            r_losses.append(r_l)
            k_losses.append(k_l)
            
            if callbacks:
                for callback in callbacks:
                    if hasattr(callback, 'on_epoch_end'):
                        # Se le pasan todas las métricas en tiempo real al ProgressCallback del spinner
                        callback.on_epoch_end(epoch, logs={'loss': t_l, 'reconstruction_loss': r_l, 'kl_loss': k_l})
        print()
        return KerasVAEHistoryMimic(t_losses, r_losses, k_losses)
        
    def predict(self, X, verbose=0):
        # Mapeo completo de reconstrucción
        _, _, z, _, _, _ = self.core._forward_encoder(X)
        output, _, _ = self.core._forward_decoder(z)
        return output

class SubModelWrapper:
    def __init__(self, predict_fn):
        self.predict_fn = predict_fn
    def predict(self, X, verbose=0):
        return self.predict_fn(X)

# =====================================================================
# FUNCIÓN DE CONSTRUCCIÓN COMPATIBLE
# =====================================================================
def build_vae(input_dim, hidden_layers, latent_dim, learning_rate=0.001, use_cuda=False, kl_weight=0.01):
    if use_cuda:
        from models_cuda_vae import DynamicCUDAVAE
        core_model = DynamicCUDAVAE(input_dim, hidden_layers, latent_dim, learning_rate, kl_weight)
    else:
        core_model = DynamicNumPyVAE(input_dim, hidden_layers, latent_dim, learning_rate, kl_weight)
    
    vae = VAEWrapper(core_model)
    # El encoder de Keras devuelve una lista: [z_mean, z_log_var, z]
    encoder = SubModelWrapper(lambda X: list(core_model._forward_encoder(X)[:3]))
    # El decoder recibe z y devuelve la imagen sintética
    decoder = SubModelWrapper(lambda Z: core_model._forward_decoder(Z)[0])
    
    return vae, encoder, decoder