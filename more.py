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
f = cv2.imread('monedas.jpg')
gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

# Aplicar desenfoque gaussiano para reducir ruido
blurred = cv2.GaussianBlur(gray, (9, 9), 0)

# Detectar bordes con Canny
edges = cv2.Canny(blurred, 40, 80)
imshow(edges)

# Realizar dilatación y erosión para eliminar detalles pequeños
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# Otra alternativa: primero dilatar y luego aplicar erosión
morph_dilate = cv2.dilate(edges, k, iterations=2)  # Aumenta las áreas blancas
morph_cleaned = cv2.erode(morph_dilate, k, iterations=2)  # Reduce detalles internos pequeños
imshow(morph_cleaned)

#Realizar apertura morfológica para eliminar detalles pequeños
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
morph_open = cv2.morphologyEx(morph_cleaned, cv2.MORPH_OPEN, kernel)
imshow(morph_open)

# Umbralización
_, thresh_otsu = cv2.threshold(morph_open, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
contours_otsu, _ = cv2.findContours(thresh_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(f, contours_otsu, contourIdx=-1, color=(0, 0, 255), thickness=2)  # https://docs.opencv.org/3.4/d6/d6e/group__imgproc__draw.html#ga746c0625f1781f1ffc9056259103edbc
imshow(f)





"""morfologia despues"""
# Cargar la imagen
f = cv2.imread('monedas.jpg')
gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

# Aplicar desenfoque gaussiano para reducir ruido
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# Detectar bordes con Canny
edges = cv2.Canny(blurred, 50, 90)
imshow(edges)

# Umbralización
thresh_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
contours_adaptive, _ = cv2.findContours(thresh_adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(f, contours_adaptive, contourIdx=-1, color=(0, 0, 255), thickness=2)
imshow(f)

# Realizar dilatación y erosión para eliminar detalles pequeños
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))

# Otra alternativa: primero dilatar y luego aplicar erosión
morph_dilate = cv2.dilate(thresh_otsu, k, iterations=2)  # Aumenta las áreas blancas
morph_cleaned = cv2.erode(morph_dilate, k, iterations=2)  # Reduce detalles internos pequeños
imshow(morph_cleaned)

"""#Realizar apertura morfológica para eliminar detalles pequeños
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
morph_open = cv2.morphologyEx(morph_cleaned, cv2.MORPH_OPEN, kernel)
imshow(morph_open)"""

# Mostrar los resultados
imshow(f)
imshow(morph_open)