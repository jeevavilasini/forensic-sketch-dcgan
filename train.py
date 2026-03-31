import tensorflow as tf
import time
import os
from model import Generator, Discriminator

# --- Configuration ---
BINARY_CROSS_ENTROPY = tf.keras.losses.BinaryCrossentropy(from_logits=True)
LAMBDA = 100  # L1 loss weight per paper
LEARNING_RATE = 0.001 # Strict per paper simulation details
EPOCHS = 100
SWITCH_EPOCH = 30 # Logic for activating D2

# --- Initialize Models & Optimizers ---
generator = Generator()
d1_patch = Discriminator(is_dense=False) 
d2_dense = Discriminator(is_dense=True)

gen_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=0.5)
d1_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=0.5)
d2_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=0.5)

# --- Loss Functions ---
def generator_loss(disc_generated_output, gen_output, target, d2_output=None, epoch=0):
    # Standard GAN Loss
    gan_loss = BINARY_CROSS_ENTROPY(tf.ones_like(disc_generated_output), disc_generated_output)
    
    # D2 Contribution after Epoch 30
    if epoch > SWITCH_EPOCH and d2_output is not None:
        gan_loss += BINARY_CROSS_ENTROPY(tf.ones_like(d2_output), d2_output)
        
    # Mean Absolute Error (L1 Loss) for structural accuracy
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    return gan_loss + (LAMBDA * l1_loss)

def discriminator_loss(real_output, generated_output):
    real_loss = BINARY_CROSS_ENTROPY(tf.ones_like(real_output), real_output)
    generated_loss = BINARY_CROSS_ENTROPY(tf.zeros_like(generated_output), generated_output)
    return real_loss + generated_loss

# --- Training Step ---
@tf.function
def train_step(input_image, target, epoch):
    with tf.GradientTape(persistent=True) as gen_tape, \
         tf.GradientTape() as d1_tape, \
         tf.GradientTape() as d2_tape:
        
        gen_output = generator(input_image, training=True)

        # D1 (PatchGAN) Pass
        d1_real = d1_patch(target, training=True)
        d1_fake = d1_patch(gen_output, training=True)
        
        # D2 (Dense) Pass - Conditional Logic
        d2_real, d2_fake = None, None
        if epoch > SWITCH_EPOCH:
            d2_real = d2_dense(target, training=True)
            d2_fake = d2_dense(gen_output, training=True)

        # Calculate Losses
        gen_total_loss = generator_loss(d1_fake, gen_output, target, d2_fake, epoch)
        d1_loss = discriminator_loss(d1_real, d1_fake)
        
        if epoch > SWITCH_EPOCH:
            d2_loss = discriminator_loss(d2_real, d2_fake)

    # Backpropagation
    gen_grads = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    gen_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))

    d1_grads = d1_tape.gradient(d1_loss, d1_patch.trainable_variables)
    d1_optimizer.apply_gradients(zip(d1_grads, d1_patch.trainable_variables))

    if epoch > SWITCH_EPOCH:
        d2_grads = d2_tape.gradient(d2_loss, d2_dense.trainable_variables)
        d2_optimizer.apply_gradients(zip(d2_grads, d2_dense.trainable_variables))
        return gen_total_loss, d1_loss, d2_loss
    
    return gen_total_loss, d1_loss, 0

# --- Training Loop ---
def fit(train_ds, epochs):
    for epoch in range(epochs):
        start = time.time()
        print(f"Epoch {epoch+1}/{epochs}")
        
        for n, (input_image, target) in train_ds.enumerate():
            g_loss, d1_l, d2_l = train_step(input_image, target, epoch)
            
        if (epoch + 1) % 10 == 0:
            print(f'Time for epoch {epoch + 1} is {time.time()-start:.2f} sec')
            print(f'Gen Loss: {g_loss:.4f}, D1 Loss: {d1_l:.4f}, D2 Loss: {d2_l:.4f}')

print("Training script with 30-epoch D2 activation logic is ready.")