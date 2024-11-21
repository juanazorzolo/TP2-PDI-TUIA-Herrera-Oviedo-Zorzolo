""" ------------------ EJERCICIO 2 ----------------------- """
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans

# Defininimos función para mostrar imágenes
def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:        
        plt.show(block=blocking)

# Las patentes argentinas miden 29,4 cm de largo por 12,9 cm de alto
# Relación de aspecto = largo/alto = 2.28

# Crear carpeta para guardar recortes
output_folder = "placas_detectadas"
os.makedirs(output_folder, exist_ok=True)

# Cargar la imagen
img = cv2.imread('patentes/img12.png')

# Mostrar imagen original
#imshow(img, title="Imagen Original", color_img=True)

# Convertir a escala de grises
img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#imshow(img_gris, title="Escala de Grises")

# Reducir ruido con un desenfoque Gaussiano
img_desenfoque = cv2.GaussianBlur(img_gris, (5, 5), 0)
#imshow(img_desenfoque, title="Desenfoque Gaussiano")

# Detectar bordes con Canny
bordes = cv2.Canny(img_desenfoque, 80, 160)
imshow(bordes, title="Bordes Detectados")

"""VER COMPONENTES CONECTADAS PARA LAS LETRAS???"""

# Aplicar operaciones morfológicas para resaltar formas rectangulares
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
morph = cv2.morphologyEx(bordes, cv2.MORPH_CLOSE, kernel, iterations=2)
imshow(morph, title="Morfología Aplicada")

# Encontrar contornos
contornos, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

img_contornos = img.copy()
cv2.drawContours(img_contornos, contornos, -1, (0, 255, 0), 2)
imshow(img_contornos, title="Contornos Dibujados")

# Filtrar contornos para identificar la placa
relacion_aspecto_esperada = 2.28
tolerancia_aspecto = 0.7  # Tolerancia del 30% alrededor de la relación esperada
patente_detectada = False  # Bandera para verificar si se detectó alguna placa

for idx, contorno in enumerate(contornos):
    # Calcular área y perímetro
    area = cv2.contourArea(contorno)
    perimetro = cv2.arcLength(contorno, True)
    
    # Aproximar el contorno a un polígono
    epsilon = 0.02 * perimetro
    aproximado = cv2.approxPolyDP(contorno, epsilon, True)
    
    # Verificar si es un rectángulo (4 vértices)
    if len(aproximado) == 4:
        # Calcular bounding box del contorno
        x, y, w, h = cv2.boundingRect(aproximado)
        
        # Calcular relación de aspecto
        relacion_aspecto = w / h
        
        # Validar la relación de aspecto y el área. 1.58 < 1.89 < 2.98
        if (relacion_aspecto_esperada - tolerancia_aspecto <= relacion_aspecto <= relacion_aspecto_esperada + tolerancia_aspecto) and area < 5000:
            # Dibujar el contorno en la imagen original
            cv2.drawContours(img, [aproximado], -1, (0, 255, 0), 3)
            imshow(img, title=f"Placa Detectada #{idx+1}", color_img=True)
            
            # Recortar la región de interés (ROI)
            roi_patente = img[y:y+h, x:x+w]
            imshow(roi_patente, title=f"Placa Recortada #{idx+1}", color_img=True)
            
            # Guardar el recorte en la carpeta
            output_path = os.path.join(output_folder, f"placa_{idx+1}.png")
            cv2.imwrite(output_path, roi_patente)
            print(f"Placa #{idx+1} guardada en: {output_path}")
            
            patente_detectada = True

if not patente_detectada:
    print("No se detectó ninguna placa en la imagen.")