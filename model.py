import tensorflow as tf
from tensorflow.keras import layers

def downsample(filters, size, apply_batchnorm=True):
    """Utility for Generator Encoder blocks [cite: 141, 145]"""
    result = tf.keras.Sequential()
    # Generator uses 3x3 filters with stride 2 
    result.add(layers.Conv2D(filters, size, strides=2, padding='same', use_bias=False))
    if apply_batchnorm:
        result.add(layers.BatchNormalization()) # [cite: 141]
    result.add(layers.ReLU()) # [cite: 149]
    return result

def upsample(filters, size, apply_dropout=False):
    """Improved Decoder block to prevent checkerboard artifacts"""
    result = tf.keras.Sequential()
    # FIX: Replace Conv2DTranspose with UpSampling2D + Conv2D
    result.add(layers.UpSampling2D(size=(2, 2))) 
    result.add(layers.Conv2D(filters, size, strides=1, padding='same', use_bias=False))
    result.add(layers.BatchNormalization())
    if apply_dropout:
        result.add(layers.Dropout(0.5))
    result.add(layers.ReLU())
    return result

def Generator():
    """Extended U-Net Architecture with 6 blocks [cite: 146, 150]"""
    # Paper uses 128x128x3 input [cite: 147]
    inputs = layers.Input(shape=[128, 128, 3])

    # Encoder (Downsampling)
    d1 = downsample(64, 3, apply_batchnorm=False)(inputs) # [cite: 134, 145]
    d2 = downsample(128, 3)(d1) # [cite: 133, 163]
    d3 = downsample(256, 3)(d2)
    
    # Decoder (Upsampling) with Skip Connections [cite: 90, 146]
    u1 = upsample(128, 3)(d3)
    u1 = layers.Concatenate()([u1, d2]) # Skip connection [cite: 146]
    
    u2 = upsample(64, 3)(u1)
    u2 = layers.Concatenate()([u2, d1])
    
    # Final Output Layer [cite: 165]
    last = layers.Conv2DTranspose(3, 3, strides=2, padding='same', activation='tanh')
    return tf.keras.Model(inputs=inputs, outputs=last(u2))

def Discriminator(is_dense=False):
    """Dual Discriminator setup (PatchGAN or Dense Layer) [cite: 139, 140, 169]"""
    inputs = layers.Input(shape=[128, 128, 3])
    
    # Common Convolutional Base
    # Discriminators use 4x4 filters and stride 2 
    x = layers.Conv2D(64, 4, strides=2, padding='same')(inputs) # [cite: 154, 175]
    x = layers.LeakyReLU()(x) # [cite: 142]
    x = layers.Dropout(0.001)(x) # 
    
    x = layers.Conv2D(128, 4, strides=2, padding='same')(x) # [cite: 155, 175]
    x = layers.BatchNormalization()(x) # [cite: 175]
    x = layers.LeakyReLU()(x)

    if is_dense:
        # D2: Added Dense Layer blocks for accuracy [cite: 162, 171, 173]
        x = layers.Flatten()(x)
        x = layers.Dense(512)(x) # [cite: 157, 172]
        x = layers.Softmax()(x) # 
        x = layers.Dense(1)(x)
    else:
        # D1: PatchGAN outputting a dimensional array [cite: 166, 168]
        x = layers.Conv2D(1, 4, strides=1, padding='same')(x)
        
    return tf.keras.Model(inputs=inputs, outputs=x)