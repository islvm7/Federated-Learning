import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



DATA_PATH = "../../data/processed/oihacdb_processed.npz"

data = np.load(DATA_PATH)

X = data["X"]
y = data["y"]

print("[INFO] Dataset OIHACDB chargé")
print("Shape X:", X.shape, "Shape y:", y.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Modèle MLP centralisé

model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    max_iter=200,
    random_state=42
)


print(" Entraînement centralisé...")
model.fit(X_train, y_train)

#Evaluation
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print(f" Accuracy centralisée = {acc:.4f}")

