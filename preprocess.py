import cv2
import numpy as np
import os
import re

# --- PATH CONFIGURATION ---
RAW_PHOTO_DIR = '/content/dataset/photos'
RAW_SKETCH_DIR = '/content/dataset/sketches'
SAVE_DIR = '/content/processed_data'

def extract_id(filename):
    """Finds the '005-01' part of a filename."""
    match = re.search(r'(\d+-\d+)', filename)
    return match.group(1) if match else None

def prepare_data_final():
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    
    photo_files = os.listdir(RAW_PHOTO_DIR)
    sketch_files = os.listdir(RAW_SKETCH_DIR)
    
    # Map the unique ID to the actual sketch filename
    sketch_map = {extract_id(f): f for f in sketch_files if extract_id(f)}

    count = 0
    for p_file in photo_files:
        p_id = extract_id(p_file)
        
        if p_id in sketch_map:
            p_path = os.path.join(RAW_PHOTO_DIR, p_file)
            s_path = os.path.join(RAW_SKETCH_DIR, sketch_map[p_id])
            
            real_img = cv2.imread(p_path)
            sketch_img = cv2.imread(s_path)
            
            if real_img is None or sketch_img is None: continue
            
            # Resize for the paper's model (128x128)
            real_img = cv2.resize(real_img, (128, 128))
            sketch_img = cv2.resize(sketch_img, (128, 128))
            
            real_gray = cv2.cvtColor(real_img, cv2.COLOR_BGR2GRAY)
            sketch_gray = cv2.cvtColor(sketch_img, cv2.COLOR_BGR2GRAY)
            
            # Sobel Filter (Forensic Detail)
            sobelx = cv2.Sobel(sketch_gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(sketch_gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_combined = cv2.magnitude(sobelx, sobely)
            sobel_sketch = 255 - cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Stack into 128x256 paired image
            combined = np.hstack((sobel_sketch, real_gray))
            cv2.imwrite(os.path.join(SAVE_DIR, p_file), combined)
            count += 1

    print(f"✅ Success! Created {count} paired images in {SAVE_DIR}")

if __name__ == "__main__":
    prepare_data_final()