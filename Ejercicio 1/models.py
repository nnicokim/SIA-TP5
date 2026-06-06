from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

def build_autoencoder(input_dim, hidden_layers, latent_dim, learning_rate=0.001):
    # --- ENCODER ---
    inputs = Input(shape=(input_dim,))
    x = inputs
    for units in hidden_layers:
        x = Dense(units, activation='relu')(x)
    latent_space = Dense(latent_dim, activation='linear', name="latent_space")(x)
    
    # --- DECODER ---
    x = latent_space
    for units in reversed(hidden_layers):
        x = Dense(units, activation='relu')(x)
    outputs = Dense(input_dim, activation='sigmoid')(x)
    
    # --- ENSAMBLAJE ---
    autoencoder = Model(inputs, outputs)
    encoder = Model(inputs, latent_space)
    
    # Generar un modelo Decoder independiente para inferencia sintética
    decoder_input = Input(shape=(latent_dim,))
    x_dec = decoder_input
    for i in range(len(hidden_layers) + 1):
        x_dec = autoencoder.layers[-(len(hidden_layers) + 1) + i](x_dec)
    decoder = Model(decoder_input, x_dec)
    
    autoencoder.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return autoencoder, encoder, decoder