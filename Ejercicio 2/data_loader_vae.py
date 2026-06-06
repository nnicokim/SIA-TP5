import os
import re
import numpy as np

def cargar_datos_kanji(filepath="symbol.h"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró '{filepath}'.")
    
    # Agregamos encoding='utf-8' para que no explote al leer los Kanjis en los comentarios
    with open(filepath, 'r', encoding='utf-8') as file:
        contenido = file.read()
        
    contenido = re.sub(r'//.*?\n|/\*.*?\*/', '', contenido, flags=re.S)
    numeros = re.findall(r'\b[01]\b', contenido)
    
    if len(numeros) != 3200:
        raise ValueError(f"Se esperaban 3200 valores (32 kanjis de 10x10), se leyeron {len(numeros)}.")
    
    datos_binarios = [int(n) for n in numeros]
    return np.array(datos_binarios, dtype='float32').reshape(32, 100)