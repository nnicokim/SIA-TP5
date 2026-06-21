import ctypes
import numpy as np
import os

lib_path = os.path.join(os.path.dirname(__file__), 'libbackend.so')
if not os.path.exists(lib_path):
    raise FileNotFoundError(f"No se encontró el backend CUDA compilado en {lib_path}. Ejecuta 'make' en la carpeta cuda_backend.")

lib = ctypes.CDLL(lib_path)

# ---------------------------------------------------------
# Tipos y Definiciones
# ---------------------------------------------------------
c_float_p = ctypes.POINTER(ctypes.c_float)
c_int = ctypes.c_int
c_float = ctypes.c_float

# Memoria
lib.alloc_gpu.argtypes = [c_int]
lib.alloc_gpu.restype = c_float_p
lib.free_gpu.argtypes = [c_float_p]

lib.copy_to_gpu.argtypes = [c_float_p, c_float_p, c_int]
lib.copy_to_host.argtypes = [c_float_p, c_float_p, c_int]

# Matmul
lib.matmul.argtypes = [c_float_p, c_float_p, c_float_p, c_int, c_int, c_int]
lib.matmul_T1.argtypes = [c_float_p, c_float_p, c_float_p, c_int, c_int, c_int]
lib.matmul_T2.argtypes = [c_float_p, c_float_p, c_float_p, c_int, c_int, c_int]

# Element-wise & Bias
lib.add_bias.argtypes = [c_float_p, c_float_p, c_int, c_int]
lib.matrix_add.argtypes = [c_float_p, c_float_p, c_float_p, c_int]
lib.relu_forward.argtypes = [c_float_p, c_float_p, c_int]
lib.relu_backward.argtypes = [c_float_p, c_float_p, c_float_p, c_int]
lib.sigmoid_forward.argtypes = [c_float_p, c_float_p, c_int]
lib.sum_rows.argtypes = [c_float_p, c_float_p, c_int, c_int]

# VAE
lib.generate_normal.argtypes = [c_float_p, c_int, ctypes.c_ulonglong]
lib.reparameterize.argtypes = [c_float_p, c_float_p, c_float_p, c_float_p, c_int]
lib.bce_backward.argtypes = [c_float_p, c_float_p, c_float_p, c_int, c_int]
lib.kl_grad.argtypes = [c_float_p, c_float_p, c_float_p, c_float_p, c_float_p, c_float_p, c_float, c_int]

# Adam
lib.adam_step.argtypes = [c_float_p, c_float_p, c_float_p, c_float_p, c_float, c_float, c_float, c_float, c_int, c_int]

# Loss
lib.compute_loss_gpu.argtypes = [c_float_p, c_float_p, c_float_p, c_float_p, c_int, c_int, c_int, c_float_p, c_float_p]

# ---------------------------------------------------------
# CUDATensor Wrapper (Maneja punteros en la GPU)
# ---------------------------------------------------------
class CUDATensor:
    def __init__(self, shape, init_data=None):
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape
        self.size = int(np.prod(shape))
        self.ptr = lib.alloc_gpu(self.size)
        
        if init_data is not None:
            self.set_data(init_data)

    def set_data(self, np_array):
        assert np_array.shape == self.shape
        arr = np.ascontiguousarray(np_array, dtype=np.float32)
        arr_ptr = arr.ctypes.data_as(c_float_p)
        lib.copy_to_gpu(self.ptr, arr_ptr, self.size)

    def get_data(self):
        arr = np.empty(self.shape, dtype=np.float32)
        arr_ptr = arr.ctypes.data_as(c_float_p)
        lib.copy_to_host(arr_ptr, self.ptr, self.size)
        return arr

    def __del__(self):
        if hasattr(self, 'ptr') and self.ptr:
            lib.free_gpu(self.ptr)
            self.ptr = None
            
# ---------------------------------------------------------
# Funciones helper para cálculos
# ---------------------------------------------------------
def matmul(A: CUDATensor, B: CUDATensor, out: CUDATensor):
    m, k = A.shape
    k2, n = B.shape
    lib.matmul(A.ptr, B.ptr, out.ptr, m, n, k)

def matmul_T1(A_T: CUDATensor, B: CUDATensor, out: CUDATensor):
    k, m = A_T.shape
    k2, n = B.shape
    lib.matmul_T1(A_T.ptr, B.ptr, out.ptr, m, n, k)

def matmul_T2(A: CUDATensor, B_T: CUDATensor, out: CUDATensor):
    m, k = A.shape
    n, k2 = B_T.shape
    lib.matmul_T2(A.ptr, B_T.ptr, out.ptr, m, n, k)

def add_bias(Z: CUDATensor, b: CUDATensor):
    m, n = Z.shape
    lib.add_bias(Z.ptr, b.ptr, m, n)

def matrix_add(A: CUDATensor, B: CUDATensor, out: CUDATensor):
    assert A.size == B.size == out.size
    lib.matrix_add(A.ptr, B.ptr, out.ptr, A.size)

def relu_forward(Z: CUDATensor, A: CUDATensor):
    lib.relu_forward(Z.ptr, A.ptr, Z.size)

def relu_backward(dA: CUDATensor, Z: CUDATensor, dZ: CUDATensor):
    lib.relu_backward(dA.ptr, Z.ptr, dZ.ptr, Z.size)

def sigmoid_forward(Z: CUDATensor, A: CUDATensor):
    lib.sigmoid_forward(Z.ptr, A.ptr, Z.size)

def sum_rows(dZ: CUDATensor, db: CUDATensor):
    m, n = dZ.shape
    lib.sum_rows(dZ.ptr, db.ptr, m, n)

def generate_normal(epsilon: CUDATensor, seed: int):
    lib.generate_normal(epsilon.ptr, epsilon.size, seed)

def reparameterize(mu: CUDATensor, logvar: CUDATensor, epsilon: CUDATensor, z: CUDATensor):
    lib.reparameterize(mu.ptr, logvar.ptr, epsilon.ptr, z.ptr, mu.size)

def bce_backward(output: CUDATensor, X: CUDATensor, dZ: CUDATensor):
    m, n = output.shape
    lib.bce_backward(output.ptr, X.ptr, dZ.ptr, output.size, n)

def kl_grad(mu: CUDATensor, logvar: CUDATensor, epsilon: CUDATensor, da_z: CUDATensor, d_mu: CUDATensor, d_logvar: CUDATensor, beta: float):
    lib.kl_grad(mu.ptr, logvar.ptr, epsilon.ptr, da_z.ptr, d_mu.ptr, d_logvar.ptr, beta, mu.size)

def adam_step(W: CUDATensor, dW: CUDATensor, m: CUDATensor, v: CUDATensor, beta1: float, beta2: float, eps: float, lr: float, t: int):
    lib.adam_step(W.ptr, dW.ptr, m.ptr, v.ptr, beta1, beta2, eps, lr, t, W.size)

def compute_loss(X: CUDATensor, output: CUDATensor, mu: CUDATensor, logvar: CUDATensor, latent_dim: int):
    # We will pass CPU pointers for bce_out and kl_out to easily get the result back
    bce_out = np.array([0.0], dtype=np.float32)
    kl_out = np.array([0.0], dtype=np.float32)
    
    # Actually wait! The compute_loss_gpu function expects GPU pointers. 
    # Let's allocate temporary GPU tensors for loss, run kernel, then copy back to CPU.
    bce_gpu = CUDATensor((1,))
    kl_gpu = CUDATensor((1,))
    
    batch_size, input_dim = X.shape
    lib.compute_loss_gpu(X.ptr, output.ptr, mu.ptr, logvar.ptr, batch_size, input_dim, latent_dim, bce_gpu.ptr, kl_gpu.ptr)
    
    bce_val = bce_gpu.get_data()[0]
    kl_val = kl_gpu.get_data()[0]
    return float(bce_val), float(kl_val)
