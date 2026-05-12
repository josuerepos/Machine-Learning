# Databricks notebook source
# MAGIC %pip install tensorflow

# COMMAND ----------

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import os

# COMMAND ----------

# Ver qué archivos hay en la carpeta actual
print(os.listdir())

# COMMAND ----------

# Cargar modelo entrenado
modelo = load_model("mnist_model.keras")

print("Modelo cargado correctamente")

# COMMAND ----------

# Cargar imagen
img = Image.open("Num6.png")

# Mostrar imagen
plt.imshow(img, cmap='gray')
plt.axis('off')

# COMMAND ----------

# Convertir a escala de grises y redimensionar
img = img.convert("L")
img = img.resize((28, 28))

# Convertir a array
img_array = np.array(img)

# Invertir colores (fondo blanco → negro)
img_array = 255 - img_array

# Normalizar
img_array = img_array / 255.0

# Ajustar forma
img_array = img_array.reshape(1, 28, 28)

# COMMAND ----------

# Predecir
pred = modelo.predict(img_array)

# Resultado
print("Predicción:", np.argmax(pred))
print("Probabilidades:", pred)

# COMMAND ----------

digito = np.argmax(pred)
plt.imshow(img_array.reshape(28,28), cmap='gray')
plt.title(f"Predicción: {digito}")
plt.axis('off')

# COMMAND ----------

digito = np.argmax(pred)
print("Dígito predicho:", digito)