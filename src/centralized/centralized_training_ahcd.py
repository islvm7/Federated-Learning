import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D,
    Flatten, Dense, Dropout
)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#Parametre model
DATA_PATH = "../../data/processed/ahcd_processed.npz"
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001


data = np.load(DATA_PATH)

X_train = data["X_train"]
y_train = data["y_train"]
X_test = data["X_test"]
y_test = data["y_test"]

num_classes = len(np.unique(y_train))

print("[INFO] Dataset AHCD chargé")
print(f"Train : {X_train.shape}, Test : {X_test.shape}")
print(f"Classes : {num_classes}")

#CNN model
def build_cnn(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),

        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


#entrainement centralisé
print(" Démarrage entraînement centralisé AHCD")

model = build_cnn(X_train.shape[1:], num_classes)

model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

#Evaluation
y_pred = np.argmax(model.predict(X_test), axis=1)
acc = accuracy_score(y_test, y_pred)

print(f"\n Accuracy CENTRALISÉE (AHCD) : {acc:.4f}")

