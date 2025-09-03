# src/features.py
import os
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm
import glob

# --- Configuration ---
DATA_DIR = 'data/EuroSAT/2750'
IMG_SIZE = (64, 64)
BANDS = ('R', 'G', 'B')
PROPERTIES = ('contrast', 'homogeneity', 'energy', 'correlation')


def extract_features(img_path):
    """Extracts band statistics and GLCM features from a single image."""
    try:
        img = Image.open(img_path).resize(IMG_SIZE)
        img_array = np.array(img)

        features = []

        # 1. Band Statistics (mean + std per channel)
        for i in range(3):  # For R, G, B channels
            band = img_array[:, :, i]
            features.extend([np.mean(band), np.std(band)])

        # 2. GLCM Textures
        gray_img = np.array(img.convert('L'))

        distances = [1, 2, 3]  # pixel pair distances
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # angles in radians

        glcm = graycomatrix(
            gray_img,
            distances=distances,
            angles=angles,
            levels=256,
            symmetric=True,
            normed=True
        )

        for prop in PROPERTIES:
            values = graycoprops(glcm, prop)
            features.extend(values.flatten())  # flatten into 1D

        return features
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None


if __name__ == '__main__':
    print("Starting feature extraction...")

    class_folders = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))]

    all_features = []
    all_labels = []

    for label_idx, class_name in enumerate(tqdm(class_folders, desc="Processing Classes")):
        class_path = os.path.join(DATA_DIR, class_name)
        img_files = glob.glob(os.path.join(class_path, '*.jpg'))

        for img_path in tqdm(img_files, desc=f"Class {class_name}", leave=False):
            features = extract_features(img_path)
            if features:
                all_features.append(features)
                all_labels.append(label_idx)

    X = np.array(all_features)
    y = np.array(all_labels)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    # Preview the first feature vector
    if len(X) > 0:
        print("First feature vector example:", X[0][:20], "...")  # print first 20 values

    # Save to disk for quick loading later
    os.makedirs("data", exist_ok=True)
    np.save('data/features_X.npy', X)
    np.save('data/labels_y.npy', y)
    print("Features and labels saved to data/")
