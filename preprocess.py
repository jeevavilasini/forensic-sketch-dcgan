import cv2
import numpy as np
import os

# --- PATH CONFIGURATION ---
# Based on your Colab direct download
RAW_PHOTO_DIR = '/content/dataset/photos'
RAW_SKETCH_DIR = '/content/dataset/sketches'
SAVE_DIR = '/content/processed_data'

def prepare_cufsf_data(photo_dir, sketch_dir, save_dir):
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)
    
    filenames = os.listdir(photo_dir)
    print(f"Processing {len(filenames)} pairs for forensic DCGAN...")

    for filename in filenames:
        photo_path = os.path.join(photo_dir, filename)
        sketch_path = os.path.join(sketch_dir, filename)
        
        real_img = cv2.imread(photo_path)
        sketch_img = cv2.imread(sketch_path)
        
        if real_img is None or sketch_img is None: continue
        
        # 1. Resize to 64x64 [Source: Page 6, Section 4.1]
        real_img = cv2.resize(real_img, (64, 64))
        sketch_img = cv2.resize(sketch_img, (64, 64))
        
        # 2. Convert to Grayscale
        real_gray = cv2.cvtColor(real_img, cv2.COLOR_BGR2GRAY)
        sketch_gray = cv2.cvtColor(sketch_img, cv2.COLOR_BGR2GRAY)
        
        # 3. Contrast Enhancement [Source: Page 6, Section 4.1]
        # Using CLAHE for better forensic detail preservation
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_sketch = clahe.apply(sketch_gray)
        
        # 4. Sobel Edge Detection [Source: Page 6, Section 4.1]
        sobelx = cv2.Sobel(enhanced_sketch, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(enhanced_sketch, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = cv2.magnitude(sobelx, sobely)
        
        # Normalize and Invert to look like a pencil sketch
        sobel_sketch = 255 - cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 5. Create Paired Image (64x128)
        # Left side: Processed Forensic Sketch | Right side: Ground Truth Photo
        combined = np.hstack((sobel_sketch, real_gray))
        cv2.imwrite(os.path.join(save_dir, filename), combined)

if __name__ == "__main__":
    prepare_cufsf_data(RAW_PHOTO_DIR, RAW_SKETCH_DIR, SAVE_DIR)
    print(f"Success! Data saved to {SAVE_DIR}")