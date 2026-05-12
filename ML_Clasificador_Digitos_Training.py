# Databricks notebook source
# DBTITLE 1,Install required packages
# MAGIC %pip install --upgrade tensorflow scikit-learn seaborn matplotlib
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Librerías
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# COMMAND ----------

# Cargar el dataset MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# COMMAND ----------

# Normalizar los valores de píxeles (0–255 → 0–1)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# COMMAND ----------

# Añadir dimensión de canal (para imágenes 2D)
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

print("Forma del conjunto de entrenamiento:", x_train.shape)
print("Forma del conjunto de prueba:", x_test.shape)

# COMMAND ----------

# Visualizar ejemplos
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i].squeeze(), cmap="gray")
    plt.title(f"Etiqueta: {y_train[i]}")
    plt.axis("off")
plt.show()

# COMMAND ----------

# Definir el modelo
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# COMMAND ----------

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# COMMAND ----------

model.summary()

# COMMAND ----------

history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# COMMAND ----------

model.save("mnist_model.keras")
print("Modelo guardado correctamente")

# COMMAND ----------

import os
print(os.listdir())