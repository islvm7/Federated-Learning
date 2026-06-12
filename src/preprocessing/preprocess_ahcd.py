import os
import numpy as np
from PIL import Image
import re


# Parametres

IMG_SIZE = (32, 32)
DATASET_PATH = "../../data/raw/AHCD"
OUTPUT_PATH = "../../data/processed/ahcd_processed.npz"

TRAIN_DIR = "Train Images 13440x32x32/train"
TEST_DIR = "Test Images 3360x32x32/test"

LABEL_REGEX = re.compile(r"label_(\d+)")

#chargement des images et labels
def load_ahcd_from_filenames(folder_path):
    images = []
    labels = []

    files = [f for f in os.listdir(folder_path)
             if f.lower().endswith((".png", ".jpg", ".bmp"))]

    print(f" {len(files)} images détectées dans {folder_path}")

    for file in files:
        match = LABEL_REGEX.search(file)
        if match is None:
            print(f" Label introuvable dans {file}, ignorée")
            continue

        label = int(match.group(1))  
        label -= 1                  

        try:
            img_path = os.path.join(folder_path, file)
            img = Image.open(img_path).convert("L")
            img = img.resize(IMG_SIZE)

            img_array = np.array(img, dtype=np.float32) / 255.0

            images.append(img_array)
            labels.append(label)

        except Exception as e:
            print(f" Erreur avec {file} : {e}")

    return np.array(images), np.array(labels)


print(" Chargement AHCD TRAIN...")
X_train, y_train = load_ahcd_from_filenames(
    os.path.join(DATASET_PATH, TRAIN_DIR)
)

print(" Chargement AHCD TEST...")
X_test, y_test = load_ahcd_from_filenames(
    os.path.join(DATASET_PATH, TEST_DIR)
)

print(f" Train : {X_train.shape}")
print(f" Test  : {X_test.shape}")

#vérifications
assert X_train.shape[0] > 0, "Train vide"
assert X_test.shape[0] > 0, "Test vide"

print(f" Labels train min/max : {y_train.min()} → {y_train.max()}")
print(f" Labels test  min/max : {y_test.min()} → {y_test.max()}")

assert y_train.max() <= 27 and y_train.min() >= 0
assert y_test.max() <= 27 and y_test.min() >= 0


# AJOUT CANAL CNN
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]


# Sauvegarde des données prétraitées

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

np.savez(
    OUTPUT_PATH,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    classes=np.arange(28)
)

print(" Dataset AHCD correctement prétraité")
print(f" Sauvegardé dans : {OUTPUT_PATH}")
