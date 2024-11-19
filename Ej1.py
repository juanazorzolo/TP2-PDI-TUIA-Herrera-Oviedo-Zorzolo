# Importamos librerías útiles para el tp
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont


# Carga de la imagen 
img = cv2.imread('monedas.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Aplicar un filtro Gaussiano para reducir el ruido
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Aplicar umbral para binarizar la imagen
_, binary = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Realizar operaciones morfológicas para cerrar pequeños huecos
kernel_close = np.ones((5, 5), np.uint8)
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)

# Aplicar apertura para eliminar ruido
kernel_open = np.ones((3, 3), np.uint8)
cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open, iterations=1)

# Detección de bordes con el operador Canny
edges = cv2.Canny(cleaned, 100, 300)

# Encontrar los contornos
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtrar contornos pequeños por área mínima
min_area = 100  # Ajusta este valor según sea necesario
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

# Dibujar los contornos filtrados en la imagen original
output = img.copy()
cv2.drawContours(output, filtered_contours, -1, (0, 255, 0), 2)  # Dibujar en color verde

# Mostrar la imagen final con contornos
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title('Contornos detectados')
plt.axis('off')
plt.show()




"""-------------------- OTRA PRUEBA ----------------------------"""

img = cv2.imread('monedas.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Aplicar un filtro Gaussiano con un kernel diferente
blurred = cv2.GaussianBlur(gray, (7, 7), 0)  # Aumentado a (7, 7)

# Probar binarización adaptativa en lugar de Otsu
binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY_INV, 11, 2)

# Realizar operaciones morfológicas para cerrar huecos
kernel = np.ones((5, 5), np.uint8)  # Kernel más grande para un cierre mejor
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)  # Más iteraciones

# Detección de bordes con Canny ajustado
edges = cv2.Canny(closed, 30, 100)  # Reducido a 30 y 100 para mayor sensibilidad

# Encontrar los contornos
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dibujar los contornos detectados en la imagen original
output = img.copy()
cv2.drawContours(output, contours, -1, (0, 255, 0), 2)  # Dibujar en color verde

# Mostrar la imagen final con contornos
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title('Contornos detectados')
plt.axis('off')
plt.show()