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
