import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


BASE_PATH = "../../data/federated/oihacdb_clients"
NUM_ROUNDS = 20
LOCAL_EPOCHS = 5
NUM_CLASSES = 28


def create_model():
    return MLPClassifier(
        hidden_layer_sizes=(256, 128),
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

# Chargement des clients
def load_clients(path):
    clients = []
    for file in os.listdir(path):
        data = np.load(os.path.join(path, file))
        clients.append({"X": data["X"], "y": data["y"]})
    return clients

# Entraînement fédéré
def run_fl(clients):
    X_all = np.vstack([c["X"] for c in clients])
    y_all = np.hstack([c["y"] for c in clients])

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
    )

    input_dim = X_train.shape[1]
    all_classes = np.arange(NUM_CLASSES)

    global_model = create_model()
    global_model.partial_fit(
        np.zeros((NUM_CLASSES, input_dim)),
        all_classes,
        classes=all_classes
    )

    history = []

    for _ in range(NUM_ROUNDS):
        local_weights = []

        for client in clients:
            local_model = create_model()
            local_model.partial_fit(
                np.zeros((NUM_CLASSES, input_dim)),
                all_classes,
                classes=all_classes
            )

            set_weights(local_model, get_weights(global_model))

            for _ in range(LOCAL_EPOCHS):
                local_model.partial_fit(client["X"], client["y"])

            local_weights.append(get_weights(local_model))

        set_weights(global_model, fedavg(local_weights))
        acc = accuracy_score(y_test, global_model.predict(X_test))
        history.append(acc)

    return history

# Chargement des clients IID et Non-IID
clients_iid = load_clients(os.path.join(BASE_PATH, "iid"))
clients_non_iid = load_clients(os.path.join(BASE_PATH, "non_iid"))

acc_iid = run_fl(clients_iid)
acc_non_iid = run_fl(clients_non_iid)

#plotting results
plt.figure()
plt.plot(acc_iid, label="IID", marker="o")
plt.plot(acc_non_iid, label="Non-IID", marker="s")
plt.xlabel("Rounds fédérés")
plt.ylabel("Accuracy (jeu de test)")
plt.title("FL IID vs Non-IID – OIHACDB")
plt.legend()
plt.grid(True)
plt.savefig("../../results/compare_iid_non_iid_oihacdb.png", dpi=300, bbox_inches='tight')
print(f"\n[SUCCESS] Graphique sauvegardé : ../../results/compare_iid_non_iid_oihacdb.png")
plt.show()

# Résultats finaux
print("\n" + "="*60)
print("RÉSULTATS FINAUX")
print("="*60)
print(f"FL IID     - Accuracy finale : {acc_iid[-1]:.4f}")
print(f"FL Non-IID - Accuracy finale : {acc_non_iid[-1]:.4f}")
print(f"Dégradation : {(acc_iid[-1] - acc_non_iid[-1]):.4f}")