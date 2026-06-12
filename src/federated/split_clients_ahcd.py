import os
import numpy as np
from collections import defaultdict


SEED = 42
np.random.seed(SEED)

DATA_PATH = "../../data/processed/ahcd_processed.npz"
OUTPUT_BASE = "../../data/federated/ahcd_clients"
NUM_CLIENTS = 10

# IID
IID_SAMPLES_PER_CLIENT = 1344 
# 13440 / 10 = utilise TOUT le train set

# NON-IID (Dirichlet)
DIRICHLET_ALPHA = 0.1  


print("="*70)
print("CRÉATION CLIENTS FÉDÉRÉS - AHCD")
print("="*70)

data = np.load(DATA_PATH)
X_train = data["X_train"]
y_train = data["y_train"]
X_test = data["X_test"]
y_test = data["y_test"]

print(f"\n Train set : {X_train.shape}")
print(f"Test set  : {X_test.shape}")
print(f"Classes   : {len(np.unique(y_train))}")


X = X_train
y = y_train

num_samples = len(X)
num_classes = len(np.unique(y))

print(f"\n[INFO] Total pour clients : {num_samples} exemples, {num_classes} classes")


def create_iid_clients(X, y, num_clients):
    
    print(f"\n{'='*70}")
    print("CRÉATION CLIENTS IID")
    print(f"{'='*70}")
    
    
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    
    samples_per_client = len(X) // num_clients
    
    clients = []
    for i in range(num_clients):
        start = i * samples_per_client
        end = start + samples_per_client if i < num_clients - 1 else len(X)
        
        X_client = X_shuffled[start:end]
        y_client = y_shuffled[start:end]
        
        # Statistiques
        unique, counts = np.unique(y_client, return_counts=True)
        
        clients.append({
            "X": X_client,
            "y": y_client
        })
        
        print(f"Client {i}: {len(y_client):5d} samples, {len(unique):2d}/28 classes")
    
    return clients

# Création Non-IID (Dirichlet)
def create_non_iid_dirichlet(X, y, num_clients, alpha):
    
    print(f"\n{'='*70}")
    print(f"CRÉATION CLIENTS NON-IID (Dirichlet alpha={alpha})")
    print(f"{'='*70}")
    
    class_indices = defaultdict(list)
    for idx, label in enumerate(y):
        class_indices[label].append(idx)
    
    client_data_indices = [[] for _ in range(num_clients)]
    
    for class_id in range(num_classes):
        
        proportions = np.random.dirichlet([alpha] * num_clients)
        
        
        indices_class = np.array(class_indices[class_id])
        np.random.shuffle(indices_class)
        
        
        splits = (np.cumsum(proportions) * len(indices_class)).astype(int)[:-1]
        client_splits = np.split(indices_class, splits)
        
        
        for client_id, split_indices in enumerate(client_splits):
            client_data_indices[client_id].extend(split_indices)
    

    clients = []
    all_classes_covered = []
    
    for i in range(num_clients):
        indices_client = client_data_indices[i]
        
        np.random.shuffle(indices_client)
        
        X_client = X[indices_client]
        y_client = y[indices_client]
        
        unique, counts = np.unique(y_client, return_counts=True)
        all_classes_covered.append(set(unique))
        
        distribution = dict(zip(unique, counts))
        top5 = sorted(distribution.items(), key=lambda x: x[1], reverse=True)[:5]
        
        clients.append({
            "X": X_client,
            "y": y_client
        })
        
        print(f"\nClient {i}:")
        print(f"  Total     : {len(y_client):5d} samples")
        print(f"  Classes   : {len(unique):2d}/28")
        print(f"  Top-5     : {[f'c{c}:{n}' for c, n in top5]}")
        
        if len(unique) < 10:
            print(f"  ⚠️ WARNING: Seulement {len(unique)} classes (risque de sous-apprentissage)")
    
    print(f"\n{'='*70}")
    print("ANALYSE GLOBALE NON-IID")
    print(f"{'='*70}")
    
    covered_classes = set.union(*all_classes_covered)
    print(f"Classes couvertes par ≥1 client : {len(covered_classes)}/28")
    
    if len(covered_classes) < 28:
        missing = set(range(28)) - covered_classes
        print(f"⚠️⚠️⚠️ CLASSES MANQUANTES : {sorted(missing)}")
        print("⚠️⚠️⚠️ CES CLASSES NE SERONT JAMAIS APPRISES !")
    else:
        print("✅ Toutes les 28 classes sont couvertes")
    
    avg_classes = np.mean([len(s) for s in all_classes_covered])
    print(f"Moyenne classes/client         : {avg_classes:.1f}")
    
    if avg_classes >= 20:
        print("📊 Niveau : Faiblement Non-IID")
    elif avg_classes >= 15:
        print("📊 Niveau : Modérément Non-IID ✅")
    elif avg_classes >= 10:
        print("📊 Niveau : Fortement Non-IID")
    else:
        print("📊 Niveau : Extrêmement Non-IID ⚠️")
    
    return clients

#sauvegarde des clients
def save_clients(clients, output_dir, client_type):
    os.makedirs(output_dir, exist_ok=True)
    
    for i, client in enumerate(clients):
        output_path = os.path.join(output_dir, f"client_{i}.npz")
        np.savez(
            output_path,
            X=client["X"],
            y=client["y"]
        )
    
    print(f"\n[SUCCESS] {len(clients)} clients {client_type} sauvegardés dans {output_dir}")

if __name__ == "__main__":
    
    iid_clients = create_iid_clients(X, y, NUM_CLIENTS)
    save_clients(iid_clients, os.path.join(OUTPUT_BASE, "iid"), "IID")
    
    non_iid_clients = create_non_iid_dirichlet(X, y, NUM_CLIENTS, DIRICHLET_ALPHA)
    save_clients(non_iid_clients, os.path.join(OUTPUT_BASE, "non_iid"), "Non-IID")
    
    print("\n" + "="*70)
    print("✅ CRÉATION TERMINÉE AVEC SUCCÈS")


