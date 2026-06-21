import numpy as np
import os

try:
    from cuda_backend.cuda_manager import (
        CUDATensor, matmul, matmul_T1, matmul_T2, add_bias, matrix_add,
        relu_forward, relu_backward, sigmoid_forward, sum_rows, 
        generate_normal, reparameterize, bce_backward, kl_grad, adam_step, compute_loss
    )
    CUDA_AVAILABLE = True
except ImportError as e:
    print(f"CUDA ImportError: {e}")
    CUDA_AVAILABLE = False

class DynamicCUDAVAE:
    def __init__(self, input_dim, hidden_layers, latent_dim, learning_rate=0.001):
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA backend not available. Please compile libbackend.so in cuda_backend/")
            
        self.lr = learning_rate
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers
        
        self.enc_sizes = [input_dim] + list(hidden_layers)
        self.dec_sizes = [latent_dim] + list(reversed(hidden_layers)) + [input_dim]
        
        # --- 1. Inicialización de Pesos en GPU ---
        self.enc_weights = []
        self.enc_biases = []
        for i in range(len(self.enc_sizes) - 1):
            w = np.random.randn(self.enc_sizes[i], self.enc_sizes[i+1]) * np.sqrt(2.0 / self.enc_sizes[i])
            b = np.zeros((1, self.enc_sizes[i+1]))
            self.enc_weights.append(CUDATensor(w.shape, w))
            self.enc_biases.append(CUDATensor(b.shape, b))
            
        w_mean = np.random.randn(self.enc_sizes[-1], latent_dim) * np.sqrt(2.0 / self.enc_sizes[-1])
        self.W_mean = CUDATensor(w_mean.shape, w_mean)
        self.b_mean = CUDATensor((1, latent_dim), np.zeros((1, latent_dim)))
        
        w_log_var = np.random.randn(self.enc_sizes[-1], latent_dim) * np.sqrt(2.0 / self.enc_sizes[-1])
        self.W_log_var = CUDATensor(w_log_var.shape, w_log_var)
        self.b_log_var = CUDATensor((1, latent_dim), np.zeros((1, latent_dim)))
        
        self.dec_weights = []
        self.dec_biases = []
        for i in range(len(self.dec_sizes) - 1):
            factor = 1.0 if i == len(self.dec_sizes) - 2 else 2.0
            w = np.random.randn(self.dec_sizes[i], self.dec_sizes[i+1]) * np.sqrt(factor / self.dec_sizes[i])
            b = np.zeros((1, self.dec_sizes[i+1]))
            self.dec_weights.append(CUDATensor(w.shape, w))
            self.dec_biases.append(CUDATensor(b.shape, b))
            
        self.all_weights = self.enc_weights + [self.W_mean, self.W_log_var] + self.dec_weights
        self.all_biases = self.enc_biases + [self.b_mean, self.b_log_var] + self.dec_biases
        
        # --- 2. Inicialización de momentos Adam en GPU ---
        self.m_w = [CUDATensor(w.shape, np.zeros(w.shape)) for w in self.all_weights]
        self.v_w = [CUDATensor(w.shape, np.zeros(w.shape)) for w in self.all_weights]
        self.m_b = [CUDATensor(b.shape, np.zeros(b.shape)) for b in self.all_biases]
        self.v_b = [CUDATensor(b.shape, np.zeros(b.shape)) for b in self.all_biases]
        self.t = 0
        self.batch_size = None

    def _allocate_intermediate_tensors(self, bs):
        self.batch_size = bs
        
        self.X_gpu = CUDATensor((bs, self.input_dim))
        
        # Encoder Forward
        self.enc_z = [CUDATensor((bs, s)) for s in self.enc_sizes[1:]]
        self.enc_a = [CUDATensor((bs, s)) for s in self.enc_sizes]
        
        self.z_mean = CUDATensor((bs, self.latent_dim))
        self.z_log_var = CUDATensor((bs, self.latent_dim))
        self.epsilon = CUDATensor((bs, self.latent_dim))
        self.z = CUDATensor((bs, self.latent_dim))
        
        # Decoder Forward
        self.dec_z = [CUDATensor((bs, s)) for s in self.dec_sizes[1:]]
        self.dec_a = [CUDATensor((bs, s)) for s in self.dec_sizes]
        
        # Backward Tensors
        self.dec_dz = [CUDATensor((bs, s)) for s in self.dec_sizes[1:]]
        self.dec_da = [CUDATensor((bs, s)) for s in self.dec_sizes]
        
        self.dw_dec = [CUDATensor(w.shape) for w in self.dec_weights]
        self.db_dec = [CUDATensor(b.shape) for b in self.dec_biases]
        
        self.da_z = CUDATensor((bs, self.latent_dim))
        self.d_z_mean = CUDATensor((bs, self.latent_dim))
        self.d_z_log_var = CUDATensor((bs, self.latent_dim))
        
        self.dW_mean = CUDATensor(self.W_mean.shape)
        self.db_mean = CUDATensor(self.b_mean.shape)
        self.dW_log_var = CUDATensor(self.W_log_var.shape)
        self.db_log_var = CUDATensor(self.b_log_var.shape)
        
        self.da_from_mean = CUDATensor((bs, self.enc_sizes[-1]))
        self.da_from_log_var = CUDATensor((bs, self.enc_sizes[-1]))
        self.enc_da_last = CUDATensor((bs, self.enc_sizes[-1]))
        
        self.enc_dz = [CUDATensor((bs, s)) for s in self.enc_sizes[1:]]
        self.enc_da = [CUDATensor((bs, s)) for s in self.enc_sizes]
        
        self.dw_enc = [CUDATensor(w.shape) for w in self.enc_weights]
        self.db_enc = [CUDATensor(b.shape) for b in self.enc_biases]
        
        self.all_dw = self.dw_enc + [self.dW_mean, self.dW_log_var] + self.dw_dec
        self.all_db = self.db_enc + [self.db_mean, self.db_log_var] + self.db_dec

    def _forward_encoder(self, X_np):
        bs = X_np.shape[0]
        if self.batch_size != bs:
            self._allocate_intermediate_tensors(bs)
            
        self.X_gpu.set_data(X_np)
        
        curr_a = self.X_gpu
        self.enc_a[0] = self.X_gpu
        
        for i in range(len(self.enc_weights)):
            matmul(curr_a, self.enc_weights[i], self.enc_z[i])
            add_bias(self.enc_z[i], self.enc_biases[i])
            relu_forward(self.enc_z[i], self.enc_a[i+1])
            curr_a = self.enc_a[i+1]
            
        a_enc_last = curr_a
        
        matmul(a_enc_last, self.W_mean, self.z_mean)
        add_bias(self.z_mean, self.b_mean)
        
        matmul(a_enc_last, self.W_log_var, self.z_log_var)
        add_bias(self.z_log_var, self.b_log_var)
        
        generate_normal(self.epsilon, seed=np.random.randint(0, 1000000))
        reparameterize(self.z_mean, self.z_log_var, self.epsilon, self.z)
        
        return self.z_mean.get_data(), self.z_log_var.get_data(), self.z.get_data(), None, None, None

    def _forward_decoder(self, z_np):
        bs = z_np.shape[0]
        if self.batch_size != bs:
            self._allocate_intermediate_tensors(bs)
            
        self.z.set_data(z_np)
        curr_a = self.z
        self.dec_a[0] = self.z
        
        for i in range(len(self.dec_weights)):
            matmul(curr_a, self.dec_weights[i], self.dec_z[i])
            add_bias(self.dec_z[i], self.dec_biases[i])
            if i == len(self.dec_weights) - 1:
                sigmoid_forward(self.dec_z[i], self.dec_a[i+1])
            else:
                relu_forward(self.dec_z[i], self.dec_a[i+1])
            curr_a = self.dec_a[i+1]
            
        return curr_a.get_data(), None, None

    def train_step(self, X):
        bs = X.shape[0]
        if self.batch_size != bs:
            self._allocate_intermediate_tensors(bs)
            
        self.X_gpu.set_data(X)
        
        # --- 1. FORWARD PASS ---
        # Encoder
        curr_a = self.X_gpu
        self.enc_a[0] = self.X_gpu
        for i in range(len(self.enc_weights)):
            matmul(curr_a, self.enc_weights[i], self.enc_z[i])
            add_bias(self.enc_z[i], self.enc_biases[i])
            relu_forward(self.enc_z[i], self.enc_a[i+1])
            curr_a = self.enc_a[i+1]
            
        a_enc_last = curr_a
        
        # Latent
        matmul(a_enc_last, self.W_mean, self.z_mean)
        add_bias(self.z_mean, self.b_mean)
        
        matmul(a_enc_last, self.W_log_var, self.z_log_var)
        add_bias(self.z_log_var, self.b_log_var)
        
        generate_normal(self.epsilon, seed=np.random.randint(0, 1000000))
        reparameterize(self.z_mean, self.z_log_var, self.epsilon, self.z)
        
        # Decoder
        curr_a = self.z
        self.dec_a[0] = self.z
        for i in range(len(self.dec_weights)):
            matmul(curr_a, self.dec_weights[i], self.dec_z[i])
            add_bias(self.dec_z[i], self.dec_biases[i])
            if i == len(self.dec_weights) - 1:
                sigmoid_forward(self.dec_z[i], self.dec_a[i+1])
            else:
                relu_forward(self.dec_z[i], self.dec_a[i+1])
            curr_a = self.dec_a[i+1]
            
        output = curr_a
        
        # Losses
        bce_loss, kl_loss = compute_loss(self.X_gpu, output, self.z_mean, self.z_log_var, self.latent_dim)
        total_loss = bce_loss + 0.01 * kl_loss
        
        # --- 2. BACKWARD PASS ---
        # Salida del Decoder (BCE + Sigmoid combinados)
        bce_backward(output, self.X_gpu, self.dec_dz[-1])
        
        # Decoder Backward Loop
        da = self.dec_da[-1] # No se usa en la última capa porque dz se calcula directo de BCE
        for i in reversed(range(len(self.dec_weights))):
            z_layer = self.dec_z[i]
            a_prev = self.dec_a[i]
            
            # dz se calculó en la iteración anterior o en la pérdida
            if i < len(self.dec_weights) - 1:
                relu_backward(da, z_layer, self.dec_dz[i])
                
            dz = self.dec_dz[i]
            
            # dw = a_prev.T @ dz
            matmul_T1(a_prev, dz, self.dw_dec[i])
            
            # db = sum(dz)
            sum_rows(dz, self.db_dec[i])
            
            # da = dz @ w.T
            if i > 0:
                matmul_T2(dz, self.dec_weights[i], self.dec_da[i])
                da = self.dec_da[i]
            else:
                matmul_T2(dz, self.dec_weights[i], self.da_z)
                
        # Latent Backward
        kl_grad(self.z_mean, self.z_log_var, self.epsilon, self.da_z, self.d_z_mean, self.d_z_log_var, 0.01)
        
        # dw_mean = a_enc_last.T @ d_z_mean
        matmul_T1(a_enc_last, self.d_z_mean, self.dW_mean)
        sum_rows(self.d_z_mean, self.db_mean)
        
        matmul_T1(a_enc_last, self.d_z_log_var, self.dW_log_var)
        sum_rows(self.d_z_log_var, self.db_log_var)
        
        # da_enc_last = d_z_mean @ W_mean.T + d_z_log_var @ W_log_var.T
        matmul_T2(self.d_z_mean, self.W_mean, self.da_from_mean)
        matmul_T2(self.d_z_log_var, self.W_log_var, self.da_from_log_var)
        matrix_add(self.da_from_mean, self.da_from_log_var, self.enc_da_last)
        
        # Encoder Backward Loop
        da = self.enc_da_last
        for i in reversed(range(len(self.enc_weights))):
            z_layer = self.enc_z[i]
            a_prev = self.enc_a[i]
            
            relu_backward(da, z_layer, self.enc_dz[i])
            dz = self.enc_dz[i]
            
            matmul_T1(a_prev, dz, self.dw_enc[i])
            sum_rows(dz, self.db_enc[i])
            
            if i > 0:
                matmul_T2(dz, self.enc_weights[i], self.enc_da[i])
                da = self.enc_da[i]
                
        # --- 3. ADAM OPTIMIZER ---
        self.t += 1
        beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
        
        for i in range(len(self.all_weights)):
            adam_step(self.all_weights[i], self.all_dw[i], self.m_w[i], self.v_w[i], 
                      beta1, beta2, eps_adam, self.lr, self.t)
            adam_step(self.all_biases[i], self.all_db[i], self.m_b[i], self.v_b[i], 
                      beta1, beta2, eps_adam, self.lr, self.t)
            
        return total_loss, bce_loss, kl_loss
