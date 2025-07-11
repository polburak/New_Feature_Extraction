import os, pickle, shutil
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
from skimage import color

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

# --- Model init ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model = nn.Sequential(*list(model.children())[:-1]).eval().to(device)
preprocess = weights.transforms()

# --- Feature functions ---
def extract_features(image_path):
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(image).reshape(-1).cpu().numpy()

def get_average_lab_color(image_path):
    image = Image.open(image_path).convert('RGB').resize((224, 224))
    lab_image = color.rgb2lab(np.array(image))
    return lab_image.reshape(-1, 3).mean(axis=0)

# --- Similarity functions ---
def cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return dot / norm if norm > 0 else 0

def lab_distance(l1, l2):
    return np.linalg.norm(l1 - l2)

def combined_score(resnet_score, lab_dist, lab_max=100):
    lab_score = 1 - (lab_dist / lab_max)
    return 0.7 * resnet_score + 0.3 * max(min(lab_score, 1), 0)

def get_unique_filename(directory, filename):
    base, ext = os.path.splitext(filename)
    counter = 1
    while os.path.exists(os.path.join(directory, filename)):
        filename = f"{base}_{counter}{ext}"
        counter += 1
    return filename

def get_all_images(root_dir):
    return [os.path.join(dp, f)
            for dp, _, files in os.walk(root_dir)
            for f in files if f.lower().endswith('.jpg')]

# --- Precompute or load dataset ---
def load_or_create_features(dataset_root, pkl_path):
    if os.path.exists(pkl_path):
        print(f"[INFO] Loading cached features from {pkl_path}")
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)

    print(f"[INFO] Creating features for dataset...")
    features = []
    for path in tqdm(get_all_images(dataset_root), desc="Extracting"):
        try:
            feat = extract_features(path)
            lab = get_average_lab_color(path)
            features.append({"path": path, "feature": feat, "avg_lab": lab})
        except Exception as e:
            print(f"Error: {path} - {e}")

    with open(pkl_path, 'wb') as f:
        pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)
    return features

# --- Main matcher ---
def find_similar_images(input_paths, dataset_root, output_dir, threshold=0.55):
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "results.txt")
    pkl_path = os.path.join(dataset_root, "resnet50_dataset_features.pkl")
    dataset_features = load_or_create_features(dataset_root, pkl_path)

    inputs = []
    for path in input_paths:
        try:
            feat = extract_features(path)
            lab = get_average_lab_color(path)
            inputs.append((path, feat, lab))
        except Exception as e:
            print(f"Failed to process input: {path} - {e}")

    with open(results_file, 'w', encoding='utf-8') as f:
        for data in tqdm(dataset_features, desc="Comparing"):
            for ipath, ifeat, ilab in inputs:
                resnet_score = cosine_similarity(ifeat, data['feature'])
                lab_dist = lab_distance(ilab, data['avg_lab'])
                score = combined_score(resnet_score, lab_dist)

                if score >= threshold:
                    name = f"match_{Path(ipath).stem}_{Path(data['path']).stem}{Path(data['path']).suffix}"
                    name = get_unique_filename(output_dir, name)
                    shutil.copy(data['path'], os.path.join(output_dir, name))
                    f.write(f"{name} - %{score*100:.2f} (from {os.path.basename(ipath)})\n")
                    break

# --- Usage ---
if __name__ == "__main__":
    input_images = [
         r"D:\PycharmProjects\feature_extraction_project\input\ALGIDA_MAX_TWISTER_ISLAND_MINI\0.jpg",
      #  r"D:\PycharmProjects\feature_extraction_project\input\meyve1.jpg"
    ]
    dataset_root = r"D:\PycharmProjects\feature_extraction_project\impulse-inhome-database"
    output_dir = r"D:\PycharmProjects\feature_extraction_project\results"

    find_similar_images(input_images, dataset_root, output_dir, threshold=0.80)
