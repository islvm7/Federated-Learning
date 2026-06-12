import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Paramètres
DATASET_PATH = "../../data/federated/ahcd_clients"
TEST_DATA_PATH = "../../data/processed/ahcd_processed.npz"
CLIENT_TYPE = "non_iid"   
NUM_ROUNDS = 20
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
NUM_CLASSES = 28
INPUT_SHAPE = (32, 32, 1)
LEARNING_RATE = 0.001
SEED = 42

# Reproductibilité
np.random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)

# Modèle CNN
def create_cnn_model():
    
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=INPUT_SHAPE),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation="softmax")
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

# Chargement des clients
def load_clients(client_type):
    path = os.path.join(DATASET_PATH, client_type)
    clients = []
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dossier introuvable : {path}")
    
    files = sorted([f for f in os.listdir(path) if f.endswith(".npz")])
    
    if len(files) == 0:
        raise ValueError(f"Aucun fichier .npz trouvé dans {path}")
    
    print(f"\n[INFO] Chargement {len(files)} clients {client_type.upper()}...")
    
    for file in files:
        data = np.load(os.path.join(path, file))
        X = data["X"]
        y = data["y"]
        
        #  VÉRIFICATION CRITIQUE
        assert X.min() >= 0 and X.max() <= 1, f" {file} : normalisation incorrecte"
        assert y.min() >= 0 and y.max() < NUM_CLASSES, f" {file} : labels incorrects"
        
        clients.append({"X": X, "y": y})
        print(f"  ✓ {file:15s} : {len(X):5d} samples, classes {np.unique(y)[:3]}...{np.unique(y)[-3:]}")
    
    return clients

# Chargement test set
def load_test_set():
    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"Test set introuvable : {TEST_DATA_PATH}")
    
    data = np.load(TEST_DATA_PATH)
    X_test = data["X_test"]
    y_test = data["y_test"]
    
    #  VÉRIFICATION CRITIQUE
    assert X_test.min() >= 0 and X_test.max() <= 1, " Test set : normalisation incorrecte"
    assert y_test.min() >= 0 and y_test.max() < NUM_CLASSES, " Test set : labels incorrects"
    
    print(f"\n[INFO] Test set chargé : {X_test.shape}")
    print(f"       Normalisation : [{X_test.min():.4f}, {X_test.max():.4f}]")
    print(f"       Labels : {y_test.min()} → {y_test.max()}")
    
    return X_test, y_test

# FedAvg (moyenne pondérée)
def federated_average(weights_list, client_sizes):

    total_size = sum(client_sizes)
    avg_weights = []
    
    for layer_weights in zip(*weights_list):
        weighted_sum = np.zeros_like(layer_weights[0])
        for client_idx, w in enumerate(layer_weights):
            weight = client_sizes[client_idx] / total_size
            weighted_sum += w * weight
        avg_weights.append(weighted_sum)
    
    return avg_weights


# Évaluation 
def evaluate_model(model, X, y, description=""):


    predictions = model.predict(X, verbose=0, batch_size=128)
    y_pred = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(y, y_pred)
    
    if description:
        print(f"  {description:20s} : {accuracy:.4f}")
    
    return accuracy


# Entraînement fédéré
def federated_training(clients, X_test, y_test):
    global_model = create_cnn_model()
    
    # Historique
    history = {
        "train_accuracy": [],
        "test_accuracy": []
    }
    
    print("\n" + "="*70)
    print("DÉMARRAGE ENTRAÎNEMENT FÉDÉRÉ")
    print("="*70)
    
    for round_idx in range(NUM_ROUNDS):
        print(f"\n{'='*70}")
        print(f"ROUND {round_idx + 1}/{NUM_ROUNDS}")
        print(f"{'='*70}")
        
        local_weights = []
        client_sizes = []
        
        # Entraînement local de chaque client
        for client_id, client in enumerate(clients):
            # Modèle local = copie du global
            local_model = create_cnn_model()
            local_model.set_weights(global_model.get_weights())
            
            # Entraînement local
            history_local = local_model.fit(
                client["X"],
                client["y"],
                epochs=LOCAL_EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=0,
                validation_split=0.0  
            )
            
            # Récupération des poids
            local_weights.append(local_model.get_weights())
            client_sizes.append(len(client["X"]))
            
            # Affichage progression
            final_loss = history_local.history['loss'][-1]
            final_acc = history_local.history['accuracy'][-1]
            print(f"  Client {client_id+1:2d} | {len(client['X']):4d} samples | Loss: {final_loss:.4f} | Acc: {final_acc:.4f}")
        
        # Agrégation FedAvg
        print(f"\n   Moyenne pondérée de {len(clients)} clients...")
        new_weights = federated_average(local_weights, client_sizes)
        global_model.set_weights(new_weights)
        
        # Évaluation globale
        print(f"\n  [EVALUATION]")
        
        # Train accuracy (sur tous les clients)
        X_train_all = np.concatenate([c["X"] for c in clients], axis=0)
        y_train_all = np.concatenate([c["y"] for c in clients], axis=0)
        
        train_acc = evaluate_model(global_model, X_train_all, y_train_all, "Train Accuracy")
        test_acc = evaluate_model(global_model, X_test, y_test, "Test Accuracy")
        
        # Sauvegarde historique
        history["train_accuracy"].append(train_acc)
        history["test_accuracy"].append(test_acc)
        
        
        if round_idx == 0 or round_idx == NUM_ROUNDS - 1:
            sample_preds = np.argmax(global_model.predict(X_test[:10], verbose=0), axis=1)
            print(f"\n   Prédictions échantillon (10 premiers) :")
            print(f"    Vrais labels : {y_test[:10]}")
            print(f"    Prédictions  : {sample_preds}")
    
    return global_model, history

# Sauvegarde résultats
def save_results(history, client_type):
    output_dir = "../../results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarde historique
    output_file = os.path.join(output_dir, f"fl_{client_type}_ahcd_history.npz")
    np.savez(
        output_file,
        train_accuracy=np.array(history["train_accuracy"]),
        test_accuracy=np.array(history["test_accuracy"])
    )
    print(f"\n[SUCCESS] Historique sauvegardé : {output_file}")
    
    # Graphique
    plt.figure(figsize=(12, 6))
    plt.plot(history["train_accuracy"], marker='o', label='Train Accuracy', linewidth=2, markersize=8)
    plt.plot(history["test_accuracy"], marker='s', label='Test Accuracy', linewidth=2, markersize=8)
    plt.title(f"Federated Learning {client_type.upper()} - AHCD", fontsize=14, fontweight='bold')
    plt.xlabel("Round", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    
    graph_file = os.path.join(output_dir, f"fl_{client_type}_ahcd_convergence.png")
    plt.savefig(graph_file, dpi=150, bbox_inches='tight')
    print(f"[SUCCESS] Graphique sauvegardé : {graph_file}")
    
    plt.show()

# Affichage résultats finaux
def print_final_results(history, client_type):
    print("\n" + "="*70)
    print(f"RÉSULTATS FINAUX - FL {client_type.upper()} (AHCD)")
    print("="*70)
    print(f"Train Accuracy finale : {history['train_accuracy'][-1]:.4f}")
    print(f"Test Accuracy finale  : {history['test_accuracy'][-1]:.4f}")
    print(f"Écart Train-Test      : {(history['train_accuracy'][-1] - history['test_accuracy'][-1]):.4f}")
    print(f"Nombre de rounds      : {len(history['test_accuracy'])}")
    print("="*70)
    
    # Analyse convergence
    if len(history['test_accuracy']) >= 5:
        last_5_test = history['test_accuracy'][-5:]
        std_last_5 = np.std(last_5_test)
        print(f"\n[ANALYSE CONVERGENCE]")
        print(f"Écart-type des 5 derniers rounds : {std_last_5:.4f}")
        if std_last_5 < 0.01:
            print(" Convergence stable")
        else:
            print(" Convergence encore en cours (envisager plus de rounds)")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("FEDERATED LEARNING - AHCD")
    print("="*70)
    print(f"Type de client : {CLIENT_TYPE.upper()}")
    print(f"Rounds         : {NUM_ROUNDS}")
    print(f"Epochs locaux  : {LOCAL_EPOCHS}")
    print(f"Batch size     : {BATCH_SIZE}")
    print(f"Learning rate  : {LEARNING_RATE}")
    print("="*70)
    
    # Chargement
    print("\n Chargement des données...")
    clients = load_clients(CLIENT_TYPE)
    X_test, y_test = load_test_set()
    
    # Entraînement
    print("\n Entraînement fédéré...")
    global_model, history = federated_training(clients, X_test, y_test)
    
    # Résultats
    print("\n Sauvegarde des résultats...")
    print_final_results(history, CLIENT_TYPE)
    save_results(history, CLIENT_TYPE)
    
    print("\n" + "="*70)
    print(" Entraînement terminé avec succès !")
    print("="*70)