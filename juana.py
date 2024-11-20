import cv2
import numpy as np
from matplotlib import pyplot as plt

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

"""morfologia nates de detectar contornos"""
# Cargar la imagen
img = cv2.imread('monedas.jpg')
img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Aplicar desenfoque gaussiano para reducir ruido
img_desenfoque = cv2.GaussianBlur(img_gris, (5,5), 0)

# Detectar bordes con Canny
img_bordes = cv2.Canny(img_desenfoque, 80, 160)
imshow(img_bordes)

# Realizar dilatación y erosión para eliminar detalles pequeños
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# Otra alternativa: primero dilatar y luego aplicar erosión
img_dilatada = cv2.dilate(img_bordes, k, iterations=5)  # Aumenta las áreas blancas
img_limpia = cv2.erode(img_dilatada, k, iterations=5)  # Reduce detalles internos pequeños
imshow(img_limpia)


#Realizar apertura morfológica para eliminar detalles pequeños
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
morph_open = cv2.morphologyEx(img_limpia, cv2.MORPH_OPEN, kernel)
imshow(morph_open)

# Umbralización
_, thresh_otsu = cv2.threshold(morph_open, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
contours_otsu, _ = cv2.findContours(thresh_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours_otsu, contourIdx=-1, color=(0, 0, 255), thickness=2) 
imshow(img)



""" ------------------- PARA GUARDAR LAS IMAGENES ---------------- """
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

# Función para mostrar imágenes
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

# Crear carpetas para guardar imágenes
os.makedirs("monedas", exist_ok=True)
os.makedirs("dados", exist_ok=True)

# Cargar la imagen
img = cv2.imread('monedas.jpg')
img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Aplicar desenfoque gaussiano para reducir ruido
img_desenfoque = cv2.GaussianBlur(img_gris, (5, 5), 0)

# Detectar bordes con Canny
img_bordes = cv2.Canny(img_desenfoque, 80, 160)

# Realizar operaciones morfológicas para limpiar bordes
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
img_dilatada = cv2.dilate(img_bordes, k, iterations=5)
img_limpia = cv2.erode(img_dilatada, k, iterations=5)

# Umbralización y detección de contornos
_, thresh_otsu = cv2.threshold(img_limpia, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
contours_otsu, _ = cv2.findContours(thresh_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Variables para conteo
conteo_monedas = 0
conteo_dados = 0

# Procesar cada contorno detectado
for i, contour in enumerate(contours_otsu):
    # Calcular área y perímetro
    area = cv2.contourArea(contour)
    perimetro = cv2.arcLength(contour, True)
    
    # Ignorar objetos pequeños (ruido)
    if area < 500:
        continue
    
    # Calcular el factor de forma
    factor_forma = (4 * np.pi * area) / (perimetro ** 2) if perimetro > 0 else 0
    
    # Obtener bounding box y recortar ROI
    x, y, w, h = cv2.boundingRect(contour)
    roi = img[y:y+h, x:x+w]
    
    # Identificar si es moneda o dado usando el factor de forma
    if factor_forma > 0.7:  # Umbral para formas redondeadas
        tipo = "moneda"
        conteo_monedas += 1
        carpeta = "monedas"
    else:
        tipo = "dado"
        conteo_dados += 1
        carpeta = "dados"
    
    # Guardar la imagen recortada en la carpeta correspondiente
    ruta_archivo = os.path.join(carpeta, f"{tipo}_{i}.jpg")
    cv2.imwrite(ruta_archivo, roi)
    print(f"Guardado: {ruta_archivo}, Área={area}, Factor de Forma={factor_forma:.2f}")
    
    # Dibujar el contorno y mostrar clasificación en la imagen original
    color = (0, 255, 0) if tipo == "moneda" else (255, 0, 0)
    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
    cv2.putText(img, tipo, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# Mostrar resultados
imshow(img, title="Clasificación de objetos")
print(f"Total monedas: {conteo_monedas}")
print(f"Total dados: {conteo_dados}")
