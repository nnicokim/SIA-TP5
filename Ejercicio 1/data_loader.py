import os
import re
import numpy as np

def cargar_datos_font(filepath="font.h"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró el archivo '{filepath}'.")
    with open(filepath, 'r') as file:
        contenido = file.read()
    
    # Limpieza de comentarios y extracción hexadecimal
    contenido = re.sub(r'//.*?\n|/\*.*?\*/', '', contenido, flags=re.S)
    hex_valores = re.findall(r'0x[0-9a-fA-F]+', contenido)
    
    if len(hex_valores) != 224:
        raise ValueError(f"Se esperaban 224 valores hex, se encontraron {len(hex_valores)}.")
    
    datos_binarios = []
    for hv in hex_valores:
        bits_str = format(int(hv, 16), '05b')
        datos_binarios.extend([int(b) for b in bits_str])
        
    # Devolvemos la matriz lista para entrenar (32 caracteres, 35 píxeles)
    return np.array(datos_binarios, dtype='float32').reshape(32, 35)