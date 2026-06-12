import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# Paramètres
DATA_PATH = "../../data/federated/oihacdb_clients/non_iid"
NUM_ROUNDS = 20
LOCAL_EPOCHS = 1
NUM_CLASSES = 28

# Chargement des clients
clients = []
for file in os.listdir(DATA_PATH):
    data = np.load(os.path.join(DATA_PATH, file))
    clients.append({
        "X": data["X"],
        "y": data["y"]
    })

print(f" {len(clients)} clients chargés")

# Modèle
def create_model():
    return MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=1,
        random_state=42
    )


# FedAvg utils
def get_weights(model):
    return [w.copy() for w in model.coefs_] + [b.copy() for b in model.intercepts_]

def set_weights(model, weights):
    n = len(model.coefs_)
    model.coefs_ = weights[:n]
    model.intercepts_ = weights[n:]

def fedavg(weights_list):
    return [np.mean(w, axis=0) for w in zip(*weights_list)]


# Initialisation globale
input_dim = clients[0]["X"].shape[1]
all_classes = np.arange(NUM_CLASSES)

global_model = create_model()
X_dummy = np.zeros((NUM_CLASSES, input_dim))
y_dummy = all_classes
global_model.partial_fit(X_dummy, y_dummy, classes=all_classes)


# Entraînement fédéré
print(" Démarrage de l'entraînement fédéré")

accuracy_history = []

for r in range(1, NUM_ROUNDS + 1):
    local_weights = []

    for client in clients:
        local_model = create_model()
        local_model.partial_fit(X_dummy, y_dummy, classes=all_classes)

        set_weights(local_model, get_weights(global_model))

        for _ in range(LOCAL_EPOCHS):
            local_model.partial_fit(client["X"], client["y"])

        local_weights.append(get_weights(local_model))

    # Agrégation
    global_weights = fedavg(local_weights)
    set_weights(global_model, global_weights)

    # Évaluation globale
    X_all = np.vstack([c["X"] for c in clients])
    y_all = np.hstack([c["y"] for c in clients])
    acc = accuracy_score(y_all, global_model.predict(X_all))
    accuracy_history.append(acc)

    print(f"[ROUND {r:02d}] Accuracy globale = {acc:.4f}")

print(" Entraînement fédéré terminé")

# Courbe de convergence
plt.figure()
plt.plot(range(1, NUM_ROUNDS + 1), accuracy_history, marker='o')
plt.xlabel("Rounds fédérés")
plt.ylabel("Accuracy globale")
plt.title("Convergence de l'apprentissage fédéré (OIHACDB)")
plt.grid(True)
plt.show()

