# loss.py
import numpy as np

def binary_cross_entropy(X, x_recon):
    """
    OJO 1: Error de Reconstrucción (BCE)
    Mide qué tan idéntica es la salida del Decoder (x_recon) respecto al Kanji original (X).
    Añadimos un pequeño épsilon (1e-10) para evitar que log(0) rompa el código (NaN).
    """
    eps = 1e-10
    x_recon = np.clip(x_recon, eps, 1.0 - eps)
    
    # Fórmula clásica de BCE para datos binarios aplicados a matrices
    bce_por_pixel = - (X * np.log(x_recon) + (1.0 - X) * np.log(1.0 - x_recon))
    
    # Sumamos el error de los 100 píxeles y promediamos por la cantidad de Kanjis (32)
    return np.mean(np.sum(bce_por_pixel, axis=1))

def bce_derivative(X, x_recon):
    """
    Derivada de la pérdida de reconstrucción respecto a la entrada del paso anterior.
    Esta es la 'señal de error' inicial que se le inyectará al Decoder en el backward pass.
    """
    eps = 1e-10
    x_recon = np.clip(x_recon, eps, 1.0 - eps)
    
    # Derivada analítica combinada de la BCE con la activación Sigmoide final
    # Simplificación matemática elegante: (x_recon - X) / Batch_Size
    return (x_recon - X) / X.shape[0]

def kl_divergence(mu, log_var):
    """
    OJO 2: Divergencia KL
    Mide cuánto se aparta la distribución del Encoder de una Normal Estándar (0, 1).
    Usa exactamente la fórmula de tu apunte:
    mín L = 0.5 * sum( exp(log_var) + mu^2 - 1 - log_var )
    """
    kl_por_dimension = 0.5 * (np.exp(log_var) + mu**2 - 1.0 - log_var)
    
    # Sumamos las dimensiones latentes (2) y promediamos por la cantidad de muestras (32)
    return np.mean(np.sum(kl_por_dimension, axis=1))

def kl_derivatives(mu, log_var):
    """
    Derivadas directas de la Divergencia KL respecto a 'mu' y 'log_var'.
    Se usan para actualizar las capas finales del Encoder.
    """
    batch_size = mu.shape[0]
    
    # Derivada respecto a la media (mu)
    dkl_dmu = mu / batch_size
    
    # Derivada respecto al logaritmo de la varianza (log_var)
    dkl_dlogvar = 0.5 * (np.exp(log_var) - 1.0) / batch_size
    
    return dkl_dmu, dkl_dlogvar

def calculate_total_loss(X, x_recon, mu, log_var, lambda_reg=1.0):
    """
    Une las dos fuerzas mediante el factor lambda de regularización.
    """
    recon_loss = binary_cross_entropy(X, x_recon)
    kl_loss = kl_divergence(mu, log_var)
    
    total_loss = recon_loss + lambda_reg * kl_loss
    return total_loss, recon_loss, kl_loss