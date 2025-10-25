"""
Script d'entraînement d'un modèle CNN pour la reconnaissance de chiffres (0–9)
Basé sur les images générées avec la police Rockwell.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ===============================
# ⚙️ CONFIGURATION
# ===============================
DATASET_DIR = "training/digit_dataset"
MODEL_PATH = "app/models/digit_cnn.keras"
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 10  # Ajuste selon ton PC (10 à 30 donne déjà de bons résultats)

# ===============================
# 📦 CHARGEMENT DU DATASET
# ===============================
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2,  # 20% pour validation
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    subset="training",
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    subset="validation"
)

# ===============================
# 🧠 DÉFINITION DU MODÈLE CNN
# ===============================
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')  # 10 chiffres (0–9)
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ===============================
# 🚀 ENTRAÎNEMENT
# ===============================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# ===============================
# 💾 SAUVEGARDE DU MODÈLE
# ===============================
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)
print(f"✅ Modèle sauvegardé : {MODEL_PATH}")

# ===============================
# 📊 ÉVALUATION FINALE
# ===============================
loss, acc = model.evaluate(val_gen)
print(f"📈 Validation accuracy : {acc*100:.2f}%")
