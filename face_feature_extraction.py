from deepface import DeepFace
import numpy as np
from scipy.spatial.distance import cosine
import glob
import os

def extract_face_embeddings(img_path):
    try:
        # Use DeepFace to get face embeddings
        result = DeepFace.represent(img_path, model_name='Facenet', detector_backend="opencv", enforce_detection=False)
        embedding = result[0]['embedding']
        return embedding
    except ValueError as e:
        print(f"Face could not be detected in {img_path}. Skipping this image.")
        return None

def get_images_from_folders(folder_paths, valid_extensions=('.jpg', '.png', '.jpeg')):
    image_paths = []
    for folder_path in folder_paths:
        for ext in valid_extensions:
            image_paths.extend(glob.glob(os.path.join(folder_path, '**', f'*{ext}'), recursive=True))
    return image_paths

def is_match(known_embedding, new_embedding, threshold=0.2):
    # Calculate cosine distance between two embeddings
    distance = cosine(known_embedding, new_embedding)
    return distance < threshold

# Load images
train_folders = ['data/amitav', 'data/muni']
test_folders = ['test']
image_paths = get_images_from_folders(train_folders)
from sklearn.model_selection import train_test_split
train_image_paths = image_paths
test_image_paths = get_images_from_folders(test_folders)

# Store embeddings of known faces
known_embeddings = []
# Assuming you've extracted embeddings from the known images
for img_path in train_image_paths:
    embedding = extract_face_embeddings(img_path)
    print(img_path)
    if embedding is not None:  # Only append valid embeddings
        known_embeddings.append(embedding)

# Compare test images
for test_img_path in test_image_paths:
    new_img_embedding = extract_face_embeddings(test_img_path)

    if new_img_embedding is not None:
        match_found = False
        for known_embedding in known_embeddings:
            if is_match(known_embedding, new_img_embedding):
                print("Match found!")
                print(test_img_path)
                match_found = True
                break
        if not match_found:
            print("No match.")
            print(test_img_path)
