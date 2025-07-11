import os
import pickle
import shutil
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from skimage import color
import numpy as np

import torch
import clip

# --- Device configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load CLIP model ---
print("[INFO] Loading CLIP model...")
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()
print("[INFO] CLIP loaded.")

# --- Feature extraction (CLIP) ---
def extract_features(image_path):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image).squeeze().cpu().numpy()
        features = features / np.linalg.norm(features)  # Normalize
    return features

# --- LAB color extraction ---
def get_average_lab_color(image_path):
    image = Image.open(image_path).convert('RGB').resize((224, 224))
    lab_image = color.rgb2lab(np.array(image))
    avg_lab = lab_image.reshape(-1, 3).mean(axis=0)
    return avg_lab

# --- Utility functions ---
def get_all_images(root_dir):
    return [os.path.join(dp, f)
            for dp, dn, filenames in os.walk(root_dir)
            for f in filenames if f.lower().endswith('.jpg')]

def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot / norm if norm > 0 else 0

def lab_distance(lab1, lab2):
    return np.linalg.norm(lab1 - lab2)

def combined_similarity_score(clip_score, lab_dist, lab_max=100):
    lab_score = 1 - (lab_dist / lab_max)
    lab_score = max(min(lab_score, 1), 0)
    return 0.7 * clip_score + 0.3 * lab_score

def get_unique_filename(directory, filename):
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join(directory, new_filename)):
        new_filename = f"{base}_{counter}{ext}"
        counter += 1
    return new_filename

# --- Precompute and cache dataset features ---
def load_or_create_dataset_features(dataset_root, cache_path):
    if os.path.exists(cache_path):
        print(f"[INFO] Loading dataset features from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"[INFO] No cache found. Computing features for dataset: {dataset_root}")
    all_images = get_all_images(dataset_root)
    features_data = []

    for img_path in tqdm(all_images, desc="Precomputing dataset features"):
        try:
            feature = extract_features(img_path)
            avg_lab = get_average_lab_color(img_path)
            features_data.append({
                "path": img_path,
                "feature": feature,
                "avg_lab": avg_lab
            })
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    with open(cache_path, "wb") as f:
        pickle.dump(features_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[INFO] Dataset features cached at: {cache_path}")
    return features_data

# --- Main comparison function ---
def find_similar_images(input_image_paths, dataset_root, output_dir, threshold=0.80):
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "results.txt")

    # Load or compute dataset features
    pkl_cache = os.path.join(dataset_root, "clip_dataset_features.pkl")
    dataset_features = load_or_create_dataset_features(dataset_root, pkl_cache)

    # Process input images
    input_features = []
    for path in input_image_paths:
        try:
            feat = extract_features(path)
            avg_lab = get_average_lab_color(path)
            input_features.append((path, feat, avg_lab))
        except Exception as e:
            print(f"Failed to read input image: {path} - {e}")

    with open(results_file, 'w', encoding='utf-8') as f:
        for item in tqdm(dataset_features, desc="Comparing with inputs"):
            try:
                target_feat = item["feature"]
                target_lab = item["avg_lab"]
                img_path = item["path"]

                for input_path, input_feat, input_lab in input_features:
                    clip_score = cosine_similarity(input_feat, target_feat)
                    lab_dist = lab_distance(input_lab, target_lab)
                    final_score = combined_similarity_score(clip_score, lab_dist)

                    if final_score >= threshold:
                        input_name = Path(input_path).stem
                        target_name = Path(img_path).stem
                        target_ext = Path(img_path).suffix
                        new_filename = f"match_{input_name}_{target_name}{target_ext}"

                        new_filename = get_unique_filename(output_dir, new_filename)
                        dst_path = os.path.join(output_dir, new_filename)
                        shutil.copy(img_path, dst_path)

                        f.write(f"{new_filename} - %{final_score * 100:.2f} (match: {os.path.basename(input_path)})\n")
                        break
            except Exception as e:
                print(f"Error comparing with {item['path']}: {e}")

# --- Example usage ---
if __name__ == "__main__":
    input_images = [
        r"D:\PycharmProjects\feature_extraction_project\input\ALGIDA_MAX_TWISTER_ISLAND_MINI\0.jpg",
        #  r"D:\PycharmProjects\feature_extraction_project\input\meyve1.jpg"
    ]
    dataset_path = r"D:\PycharmProjects\feature_extraction_project\impulse-inhome-database"
    output_path = r"D:\PycharmProjects\feature_extraction_project\results"

    find_similar_images(
        input_image_paths=input_images,
        dataset_root=dataset_path,
        output_dir=output_path,
        threshold=0.80
    )
