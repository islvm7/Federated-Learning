import os
import numpy as np
from collections import defaultdict


SEED = 42
np.random.seed(SEED)

DATA_PATH = "../../data/processed/oihacdb_processed.npz"
OUTPUT_BASE = "../../data/federated/oihacdb_clients1"

NUM_CLIENTS = 10
IID_SAMPLES_PER_CLIENT = 400

NON_IID_CLASSES_PER_CLIENT = 3
SAMPLES_PER_CLASS = 50


data = np.load(DATA_PATH)

X = data["X"]   
y = data["y"]   

num_samples = len(X)
num_classes = len(np.unique(y))

print(f" OIHACDB chargé : {num_samples} exemples, {num_classes} classes")


# Création des clients IID
def create_iid_clients(X, y, num_clients, samples_per_client):
    assert num_clients * samples_per_client <= len(X), \
        "Pas assez d'exemples pour créer les clients IID demandés."

    indices = np.random.permutation(len(X))
    clients = []

    for i in range(num_clients):
        idx = indices[i * samples_per_client:(i + 1) * samples_per_client]

        clients.append({
            "X": X[idx],
            "y": y[idx],
            "client_id": i + 1,
            "type": "IID"
        })

    return clients



# Création des clients Non-IID
def create_non_iid_clients(X, y, num_clients, classes_per_client):
    class_indices = defaultdict(list)

    for idx, label in enumerate(y):
        class_indices[label].append(idx)

   
    for cls in class_indices:
        np.random.shuffle(class_indices[cls])

    clients = []
    used_indices = set()

    for i in range(num_clients):
        chosen_classes = np.random.choice(
            list(class_indices.keys()),
            classes_per_client,
            replace=False
        )

        client_indices = []

        for cls in chosen_classes:
            available = [
                idx for idx in class_indices[cls]
                if idx not in used_indices
            ]

            take = min(SAMPLES_PER_CLASS, len(available))
            selected = available[:take]

            client_indices.extend(selected)
            used_indices.update(selected)

        client_indices = np.array(client_indices)

        clients.append({
            "X": X[client_indices],
            "y": y[client_indices],
            "client_id": i + 1,
            "type": "Non-IID",
            "classes": np.unique(y[client_indices])
        })

    return clients



# Sauvegarde des clients
def save_clients(clients, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for client in clients:
        np.savez(
            os.path.join(output_dir, f"client_{client['client_id']}.npz"),
            X=client["X"],
            y=client["y"],
            client_id=client["client_id"],
            type=client["type"],
            classes=np.unique(client["y"])
        )



if __name__ == "__main__":

    print("[INFO] Création des clients IID (OIHACDB)...")
    iid_clients = create_iid_clients(
        X, y, NUM_CLIENTS, IID_SAMPLES_PER_CLIENT
    )
    save_clients(iid_clients, os.path.join(OUTPUT_BASE, "iid"))

    print("[INFO] Création des clients Non-IID (OIHACDB)...")
    non_iid_clients = create_non_iid_clients(
        X, y, NUM_CLIENTS, NON_IID_CLASSES_PER_CLIENT
    )
    save_clients(non_iid_clients, os.path.join(OUTPUT_BASE, "non_iid"))

    print("[SUCCESS] Clients fédérés OIHACDB créés")

client = np.load(
    "../../data/federated/oihacdb_clients/non_iid/client_1.npz"
)

print(client["X"].shape)
print(np.unique(client["y"]))
print(client["type"])

