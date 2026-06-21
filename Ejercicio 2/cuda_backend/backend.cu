#include <cuda_runtime.h>
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
#include <cmath>

extern "C" {

// ==========================================================
// MEMORY MANAGEMENT
// ==========================================================
float* alloc_gpu(int size) {
    float* ptr;
    cudaMalloc((void**)&ptr, size * sizeof(float));
    return ptr;
}

void free_gpu(float* ptr) {
    cudaFree(ptr);
}

void copy_to_gpu(float* dest, const float* src, int size) {
    cudaMemcpy(dest, src, size * sizeof(float), cudaMemcpyHostToDevice);
}

void copy_to_host(float* dest, const float* src, int size) {
    cudaMemcpy(dest, src, size * sizeof(float), cudaMemcpyDeviceToHost);
}

// ==========================================================
// MATMUL KERNELS
// ==========================================================
// C = A @ B
// A: (m x k), B: (k x n), C: (m x n)
__global__ void matmul_kernel(const float* A, const float* B, float* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

void matmul(const float* A, const float* B, float* C, int m, int n, int k) {
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    matmul_kernel<<<grid, block>>>(A, B, C, m, n, k);
}

// C = A.T @ B
// A: (k x m), B: (k x n), C: (m x n)
__global__ void matmul_T1_kernel(const float* A, const float* B, float* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            // A_T[row, i] = A[i, row]
            sum += A[i * m + row] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

void matmul_T1(const float* A_T, const float* B, float* C, int m, int n, int k) {
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    matmul_T1_kernel<<<grid, block>>>(A_T, B, C, m, n, k);
}

// C = A @ B.T
// A: (m x k), B: (n x k), C: (m x n)
__global__ void matmul_T2_kernel(const float* A, const float* B, float* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            // B_T[i, col] = B[col, i] -> since B is n x k, row col is col x i
            sum += A[row * k + i] * B[col * k + i];
        }
        C[row * n + col] = sum;
    }
}

void matmul_T2(const float* A, const float* B_T, float* C, int m, int n, int k) {
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    matmul_T2_kernel<<<grid, block>>>(A, B_T, C, m, n, k);
}

// ==========================================================
// VECTOR / ELEMENT-WISE KERNELS
// ==========================================================

__global__ void add_bias_kernel(float* C, const float* b, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m * n) {
        int col = idx % n;
        C[idx] += b[col];
    }
}
void add_bias(float* C, const float* b, int m, int n) {
    int size = m * n;
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    add_bias_kernel<<<blocks, threads>>>(C, b, m, n);
}

__global__ void matrix_add_kernel(const float* A, const float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) C[idx] = A[idx] + B[idx];
}
void matrix_add(const float* A, const float* B, float* C, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    matrix_add_kernel<<<blocks, threads>>>(A, B, C, size);
}

__global__ void relu_kernel(const float* Z, float* A, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) A[idx] = fmaxf(0.0f, Z[idx]);
}
void relu_forward(const float* Z, float* A, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(Z, A, size);
}

__global__ void relu_backward_kernel(const float* dA, const float* Z, float* dZ, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dZ[idx] = Z[idx] > 0.0f ? dA[idx] : 0.0f;
    }
}
void relu_backward(const float* dA, const float* Z, float* dZ, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    relu_backward_kernel<<<blocks, threads>>>(dA, Z, dZ, size);
}

__global__ void sigmoid_kernel(const float* Z, float* A, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = fmaxf(-50.0f, fminf(50.0f, Z[idx]));
        A[idx] = 1.0f / (1.0f + expf(-val));
    }
}
void sigmoid_forward(const float* Z, float* A, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    sigmoid_kernel<<<blocks, threads>>>(Z, A, size);
}

// For sum_rows (used in db = np.sum(dZ, axis=0))
__global__ void sum_rows_kernel(const float* dZ, float* db, int m, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < n) {
        float sum = 0.0f;
        for (int row = 0; row < m; ++row) {
            sum += dZ[row * n + col];
        }
        db[col] = sum;
    }
}
void sum_rows(const float* dZ, float* db, int m, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    sum_rows_kernel<<<blocks, threads>>>(dZ, db, m, n);
}

// ==========================================================
// VAE SPECIFIC KERNELS
// ==========================================================

__global__ void generate_normal_kernel(float* epsilon, int size, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        epsilon[idx] = curand_normal(&state);
    }
}
void generate_normal(float* epsilon, int size, unsigned long long seed) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    generate_normal_kernel<<<blocks, threads>>>(epsilon, size, seed);
}

__global__ void reparameterize_kernel(const float* mu, const float* logvar, const float* epsilon, float* z, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        z[idx] = mu[idx] + expf(0.5f * logvar[idx]) * epsilon[idx];
    }
}
void reparameterize(const float* mu, const float* logvar, const float* epsilon, float* z, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    reparameterize_kernel<<<blocks, threads>>>(mu, logvar, epsilon, z, size);
}

// Gradient of BCE + Sigmoid (combined for stability)
__global__ void bce_backward_kernel(const float* output, const float* X, float* dZ, int size, int input_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dZ[idx] = (output[idx] - X[idx]) / (float)input_dim;
    }
}
void bce_backward(const float* output, const float* X, float* dZ, int size, int input_dim) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    bce_backward_kernel<<<blocks, threads>>>(output, X, dZ, size, input_dim);
}

// Gradients for KL Divergence and Reparameterization
__global__ void kl_grad_kernel(const float* mu, const float* logvar, const float* epsilon, const float* da_z,
                               float* d_mu, float* d_logvar, float beta, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_mu[idx] = beta * mu[idx] + da_z[idx];
        d_logvar[idx] = 0.5f * beta * (expf(logvar[idx]) - 1.0f) + da_z[idx] * (0.5f * expf(0.5f * logvar[idx]) * epsilon[idx]);
    }
}
void kl_grad(const float* mu, const float* logvar, const float* epsilon, const float* da_z,
             float* d_mu, float* d_logvar, float beta, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    kl_grad_kernel<<<blocks, threads>>>(mu, logvar, epsilon, da_z, d_mu, d_logvar, beta, size);
}

// ==========================================================
// ADAM OPTIMIZER KERNEL
// ==========================================================
__global__ void adam_kernel(float* w, const float* dw, float* m, float* v, 
                            float beta1, float beta2, float eps, float lr, int t, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * dw[idx];
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * (dw[idx] * dw[idx]);
        
        float m_hat = m[idx] / (1.0f - powf(beta1, t));
        float v_hat = v[idx] / (1.0f - powf(beta2, t));
        
        w[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}
void adam_step(float* w, const float* dw, float* m, float* v, 
               float beta1, float beta2, float eps, float lr, int t, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    adam_kernel<<<blocks, threads>>>(w, dw, m, v, beta1, beta2, eps, lr, t, size);
}

// ==========================================================
// LOSS CALCULATION (Reduction)
// ==========================================================
// BCE elementwise
__global__ void bce_loss_kernel(const float* X, const float* output, float* loss, int size, int input_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // float32-safe clip: 1e-12f es menor que el epsilon de float32 (~1.2e-7),
        // por lo que 1.0f - 1e-12f == 1.0f y el clip NO actuaba. Cuando un sigmoid
        // satura a 1.0 exacto y X=1, daba 0*log(0)=NaN. 1e-7f sí es representable.
        float out_clipped = fmaxf(1e-7f, fminf(1.0f - 1e-7f, output[idx]));
        float bce = -(X[idx] * logf(out_clipped) + (1.0f - X[idx]) * logf(1.0f - out_clipped));
        atomicAdd(loss, bce / (float)input_dim); // match CPU np.mean(axis=1)
    }
}
// KL elementwise
__global__ void kl_loss_kernel(const float* mu, const float* logvar, float* loss, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float kl = -0.5f * (1.0f + logvar[idx] - (mu[idx] * mu[idx]) - expf(logvar[idx]));
        atomicAdd(loss, kl);
    }
}

void compute_loss_gpu(const float* X, const float* output, const float* mu, const float* logvar, 
                      int batch_size, int input_dim, int latent_dim, float* bce_out, float* kl_out) {
    // Zero out the loss pointers (assuming they are GPU pointers)
    float zero = 0.0f;
    cudaMemcpy(bce_out, &zero, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(kl_out, &zero, sizeof(float), cudaMemcpyHostToDevice);
    
    int threads = 256;
    
    int size_bce = batch_size * input_dim;
    int blocks_bce = (size_bce + threads - 1) / threads;
    bce_loss_kernel<<<blocks_bce, threads>>>(X, output, bce_out, size_bce, input_dim);
    
    int size_kl = batch_size * latent_dim;
    int blocks_kl = (size_kl + threads - 1) / threads;
    kl_loss_kernel<<<blocks_kl, threads>>>(mu, logvar, kl_out, size_kl);
}

} // extern "C"
