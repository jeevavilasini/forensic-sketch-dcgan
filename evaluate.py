import tensorflow as tf
import numpy as np
import os
import cv2
import re
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from model import Generator

# --- Configuration ---
# Update these to where your dataset actually unzipped
TEST_SKETCH_DIR = '/content/dataset/sketches' 
TEST_PHOTO_DIR = '/content/dataset/photos'
MODEL_WEIGHTS = 'generator_weights.weights.h5' # Updated to match train.py
SAVE_DIR = '/content/top_results'
IMG_SIZE = 128

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def extract_id(filename):
    """Extracts '005-01' from filenames to handle f- vs F2- naming."""
    match = re.search(r'(\d+-\d+)', filename)
    return match.group(1) if match else None

def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = (img / 127.5) - 1 
    return img

def evaluate_and_visualize():
    model = Generator()
    if os.path.exists(MODEL_WEIGHTS):
        # We must build the model's shape before loading weights
        model(tf.random.normal([1, 128, 128, 3])) 
        model.load_weights(MODEL_WEIGHTS)
        print("✅ Successfully loaded model weights.")
    else:
        print(f"❌ Error: {MODEL_WEIGHTS} not found. Check your path!")
        return

    # Create a mapping of photo IDs to full paths
    photo_files = os.listdir(TEST_PHOTO_DIR)
    photo_map = {extract_id(f): os.path.join(TEST_PHOTO_DIR, f) for f in photo_files if extract_id(f)}

    sketch_files = [f for f in os.listdir(TEST_SKETCH_DIR) if f.endswith(('.jpg', '.png'))]
    all_results = []

    print(f"Evaluating {len(sketch_files)} test images...")

    for s_file in sketch_files:
        s_id = extract_id(s_file)
        if s_id in photo_map:
            sketch = load_img(os.path.join(TEST_SKETCH_DIR, s_file))
            real = load_img(photo_map[s_id])

            # Generate Image
            prediction = model(tf.expand_dims(sketch, 0), training=False)
            
            gen_np = ((prediction[0].numpy() + 1) * 127.5).astype(np.uint8)
            real_np = ((real.numpy() + 1) * 127.5).astype(np.uint8)
            sketch_np = ((sketch.numpy() + 1) * 127.5).astype(np.uint8)

            # Calculate SSIM
            score = ssim(real_np, gen_np, channel_axis=2, data_range=255)
            
            all_results.append({
                'name': s_file,
                'score': score,
                'sketch': sketch_np,
                'gen': gen_np,
                'real': real_np
            })

    # Sort and take Top 5
    all_results.sort(key=lambda x: x['score'], reverse=True)
    top_5 = all_results[:5]

    # Visualization
    plt.figure(figsize=(15, 12))
    for i, item in enumerate(top_5):
        # Save comparison
        comparison = np.hstack((item['sketch'], item['gen'], item['real']))
        cv2.imwrite(os.path.join(SAVE_DIR, f"rank_{i+1}_{item['name']}"), 
                    cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

        # Show in Colab
        for j, (img, title) in enumerate([(item['sketch'], "Sketch"), (item['gen'], f"Gen (SSIM:{item['score']:.3f})"), (item['real'], "Real")]):
            plt.subplot(5, 3, i*3 + j + 1)
            plt.imshow(img)
            plt.title(title)
            plt.axis('off')

    plt.tight_layout()
    plt.show()

    avg_ssim = np.mean([x['score'] for x in all_results])
    print(f"\nOverall Average SSIM: {avg_ssim:.4f}")

if __name__ == "__main__":
    evaluate_and_visualize()