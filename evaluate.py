import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from model import Generator

# --- Configuration ---
TEST_SKETCH_DIR = '/content/dataset/test/sketches'
TEST_PHOTO_DIR = '/content/dataset/test/photos'
MODEL_WEIGHTS = 'generator_weights.h5'
SAVE_DIR = '/content/top_results'
IMG_SIZE = 128

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = (img / 127.5) - 1 # Normalize to [-1, 1]
    return img

def evaluate_and_visualize():
    # 1. Load trained model
    model = Generator()
    if os.path.exists(MODEL_WEIGHTS):
        model.load_weights(MODEL_WEIGHTS)
        print("Successfully loaded model weights.")
    else:
        print(f"Error: {MODEL_WEIGHTS} not found. Ensure training is finished.")
        return

    # 2. Process all test images
    test_files = [f for f in os.listdir(TEST_SKETCH_DIR) if f.endswith(('.jpg', '.png'))]
    all_results = []

    print(f"Evaluating {len(test_files)} test images...")

    for filename in test_files:
        sketch = load_img(os.path.join(TEST_SKETCH_DIR, filename))
        real = load_img(os.path.join(TEST_PHOTO_DIR, filename))

        # Generate Image
        prediction = model(tf.expand_dims(sketch, 0), training=False)
        
        # Convert to displayable format [0, 255]
        gen_np = ((prediction[0].numpy() + 1) * 127.5).astype(np.uint8)
        real_np = ((real.numpy() + 1) * 127.5).astype(np.uint8)
        sketch_np = ((sketch.numpy() + 1) * 127.5).astype(np.uint8)

        # Calculate SSIM (Structural Similarity)
        score = ssim(real_np, gen_np, channel_axis=2, data_range=255)
        
        all_results.append({
            'name': filename,
            'score': score,
            'sketch': sketch_np,
            'gen': gen_np,
            'real': real_np
        })

    # 3. Sort by SSIM score (Highest First)
    all_results.sort(key=lambda x: x['score'], reverse=True)
    top_5 = all_results[:5]

    # 4. Save and Display
    plt.figure(figsize=(18, 20))
    for i, item in enumerate(top_5):
        # Save to local folder
        comparison = np.hstack((item['sketch'], item['gen'], item['real']))
        cv2.imwrite(os.path.join(SAVE_DIR, f"rank_{i+1}_{item['name']}"), 
                    cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

        # Display in Colab
        plt.subplot(5, 3, i*3 + 1)
        plt.imshow(item['sketch'])
        plt.title(f"Rank {i+1}: Input Sketch")
        plt.axis('off')

        plt.subplot(5, 3, i*3 + 2)
        plt.imshow(item['gen'])
        plt.title(f"Generated (SSIM: {item['score']:.4f})")
        plt.axis('off')

        plt.subplot(5, 3, i*3 + 3)
        plt.imshow(item['real'])
        plt.title("Original Ground Truth")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    avg_ssim = np.mean([x['score'] for x in all_results])
    print(f"\nEvaluation Complete.")
    print(f"Overall Average SSIM: {avg_ssim:.4f}")
    print(f"Top 5 images saved to: {SAVE_DIR}")

if __name__ == "__main__":
    evaluate_and_visualize()