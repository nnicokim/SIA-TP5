# encoder.py
import numpy as np

class VariationalEncoder:
    def __init__(self, input_dim=100, hidden_dims=[64, 32], latent_dim=2, seed=42):
        self.latent_dim = latent_dim
        
        # 1. Instanciamos el generador moderno de NumPy
        self.rng = np.random.default_rng(seed)
        
        self.weights = []
        self.biases = []
        
        prev_dim = input_dim
        for h_dim in hidden_dims:
            # 2. Usamos standard_normal pasándole una tupla con las dimensiones
            W = self.rng.standard_normal((prev_dim, h_dim)) * np.sqrt(2.0 / prev_dim)
            b = np.zeros((1, h_dim))
            self.weights.append(W)
            self.biases.append(b)
            prev_dim = h_dim
            
        self.w_mu = self.rng.standard_normal((prev_dim, latent_dim)) * np.sqrt(2.0 / prev_dim)
        self.b_mu = np.zeros((1, latent_dim))
        
        self.w_logvar = self.rng.standard_normal((prev_dim, latent_dim)) * np.sqrt(2.0 / prev_dim)
        self.b_logvar = np.zeros((1, latent_dim))

    def _relu(self, x):
        return np.maximum(0, x)

    def forward(self, X):
        activations = [X]
        current_input = X
        
        for i in range(len(self.weights)):
            z_layer = np.dot(current_input, self.weights[i]) + self.biases[i]
            a_layer = self._relu(z_layer)
            activations.append(a_layer)
            current_input = a_layer
            
        mu = np.dot(current_input, self.w_mu) + self.b_mu
        log_var = np.dot(current_input, self.w_logvar) + self.b_logvar
        
        sigma = np.exp(0.5 * log_var)
        
        # 3. Generamos el épsilon con el generador moderno
        epsilon = self.rng.standard_normal(mu.shape)
        z = mu + sigma * epsilon
        
        cache = {
            'activations': activations, 
            'mu': mu,
            'log_var': log_var,
            'sigma': sigma,
            'epsilon': epsilon,
            'z': z
        }
        
        return z, mu, log_var, cache
    
    def backward(self, dl_dz, dkl_dmu, dkl_dlogvar, cache, learning_rate):
        """
        dl_dz: El error que nos acaba de devolver el Decoder.
        dkl_dmu, dkl_dlogvar: Los castigos del espacio latente que vienen de loss.py.
        """
        activations = cache['activations']
        sigma = cache['sigma']
        epsilon = cache['epsilon'] # ¡Acá usamos el ruido que guardamos!
        
        # =================================================================
        # 1. EL TRUCO DE REPARAMETRIZACIÓN (Las derivadas de tu imagen)
        # Ecuación: Z = mu + sigma * epsilon
        # =================================================================
        
        # Derivada respecto a mu (dz/dmu = 1) -> El error pasa directo
        dl_dmu_recon = dl_dz * 1.0
        
        # Derivada respecto a sigma (dz/dsigma = epsilon) -> Multiplicamos por el ruido
        dl_dsigma = dl_dz * epsilon
        
        # Como nuestra red escupe log_var y no sigma, necesitamos una derivada más:
        # sigma = exp(0.5 * log_var) -> dsigma/dlogvar = 0.5 * sigma
        dl_dlogvar_recon = dl_dsigma * 0.5 * sigma
        
        # =================================================================
        # 2. LA SUMA DE LOS DOS ERRORES (Reconstrucción + Multa KL)
        # =================================================================
        # (Acá asumimos que el factor lambda está incluido en el dkl)
        d_mu = dl_dmu_recon + dkl_dmu
        d_logvar = dl_dlogvar_recon + dkl_dlogvar
        
        # 3. Derivadas de las capas de salida del Encoder (mu y log_var)
        a_prev = activations[-1]
        
        dw_mu = np.dot(a_prev.T, d_mu)
        db_mu = np.sum(d_mu, axis=0, keepdims=True)
        
        dw_logvar = np.dot(a_prev.T, d_logvar)
        db_logvar = np.sum(d_logvar, axis=0, keepdims=True)
        
        # Pasamos el error hacia atrás, juntando los caminos de mu y log_var
        da = np.dot(d_mu, self.w_mu.T) + np.dot(d_logvar, self.w_logvar.T)
        
        # 4. Retropropagación por las capas ocultas (igual que en el decoder)
        for i in reversed(range(len(self.weights))):
            a_current = activations[i+1]
            dz = da * (a_current > 0).astype(float) # Derivada de ReLU
            
            a_prev = activations[i]
            dw = np.dot(a_prev.T, dz)
            db = np.sum(dz, axis=0, keepdims=True)
            
            da = np.dot(dz, self.weights[i].T)
            
            # --- ACTUALIZACIÓN DE PESOS ---
            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db
            
        # Actualizamos las ramas de salida
        self.w_mu -= learning_rate * dw_mu
        self.b_mu -= learning_rate * db_mu
        self.w_logvar -= learning_rate * dw_logvar
        self.b_logvar -= learning_rate * db_logvar