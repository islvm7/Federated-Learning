import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


RAW_PATH = "../../data/raw/OIHACDB/OIHACDB.csv"
OUTPUT_PATH = "../../data/processed/oihacdb_processed.npz"

print(" Chargement du dataset OIHACDB...")
df = pd.read_csv(RAW_PATH)

print(f" Shape initiale : {df.shape}")


X = df.iloc[:, :-1].values    
y = df.iloc[:, -1].values    

print(f"[INFO] Features : {X.shape}")
print(f"[INFO] Labels : {y.shape}")


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

classes = label_encoder.classes_

print(" Encodage des labels terminé")
print(f" Classes : {classes}")


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(" Normalisation des features terminée")

# Sauvegarde des données prétraitées
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

np.savez(
    OUTPUT_PATH,
    X=X_scaled,
    y=y_encoded,
    classes=classes
)

print(" Données OIHACDB prétraitées sauvegardées")
print(f" Fichier : {OUTPUT_PATH}")
