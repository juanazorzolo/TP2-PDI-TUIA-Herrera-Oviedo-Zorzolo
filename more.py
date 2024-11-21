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

""" PROCESO ITERATIVO PARA CANNY """
def ajustar_canny(imagen_gray, rango_threshold1, rango_threshold2, area_min=1000, area_max=10000):
    mejor_threshold1 = 0
    mejor_threshold2 = 0
    mejor_puntaje = 0
    mejor_bordes = None

    for t1 in rango_threshold1:
        for t2 in rango_threshold2:
            # Aplicar Canny con los parámetros actuales
            bordes = cv2.Canny(imagen_gray, t1, t2)
            
            # Operaciones morfológicas para cerrar bordes pequeños
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            bordes_dilatados = cv2.dilate(bordes, kernel, iterations=2)
            bordes_erosionados = cv2.erode(bordes_dilatados, kernel, iterations=1)
            
            # Encontrar contornos
            contornos, _ = cv2.findContours(bordes_erosionados, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filtrar contornos según el área
            puntaje_actual = 0
            for contorno in contornos:
                area = cv2.contourArea(contorno)
                if area_min < area < area_max:
                    puntaje_actual += 1  # Sumar puntos por cada contorno válido
            
            # Actualizar los mejores parámetros si el puntaje es mayor
            if puntaje_actual > mejor_puntaje:
                mejor_puntaje = puntaje_actual
                mejor_threshold1 = t1
                mejor_threshold2 = t2
                mejor_bordes = bordes

    return mejor_threshold1, mejor_threshold2, mejor_bordes


""" PROCESAR LA PATENTE Y FILTRAR CONTORNOS POR ÁREA Y RELACIÓN DE ASPECTO """
def procesar_patente(imagen, bordes, area_min=500, area_max=5000, aspect_ratio=2.28):
    # Encontrar contornos
    contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Lista para almacenar los contornos relevantes
    contornos_filtrados = []

    for contorno in contornos:
        # Calcular el área del contorno
        area = cv2.contourArea(contorno)

        # Filtrar por área (evitar contornos pequeños y grandes)
        if area_min < area < area_max:
            # Calcular la relación de aspecto
            x, y, w, h = cv2.boundingRect(contorno)
            ratio = w / h

            # Filtrar por relación de aspecto cercana a la de una patente
            if 1.8 < ratio < 2.5:  # Permitir un margen de error
                contornos_filtrados.append(contorno)

    # Dibujar los contornos filtrados sobre la imagen original
    cv2.drawContours(imagen, contornos_filtrados, -1, (0, 255, 0), 2)  # Contornos en verde

    return imagen

"""PROCESAR TODAS LAS IMÁGENES DE LA CARPETA PATENTES Y GUARDAR EN LA CARPETA RESULTADOS"""
carpeta_imagenes = "patentes/" 
rango_threshold1 = range(50, 200, 10)  # Para pasarle a la función ajustar_canny y que encuentre el mejor
rango_threshold2 = range(100, 400, 10)  # Lo mismo que arriba

# Crear carpeta de resultados si no existe
if not os.path.exists("resultados/"):
    os.makedirs("resultados/")

# Procesar cada imagen en la carpeta
for nombre_imagen in os.listdir(carpeta_imagenes):
    if nombre_imagen.endswith(".png") or nombre_imagen.endswith(".jpg"):
        ruta_imagen = os.path.join(carpeta_imagenes, nombre_imagen)
        imagen = cv2.imread(ruta_imagen)
        imagen_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        
        # Ajustar Canny
        mejor_t1, mejor_t2, bordes_optimizados = ajustar_canny(imagen_gray, rango_threshold1, rango_threshold2)
        
        # Procesar los bordes para extraer la patente y filtrar contornos
        imagen_resultado = procesar_patente(imagen, bordes_optimizados)

        # Guardar resultado
        ruta_guardado = os.path.join("resultados/", f"patente_{nombre_imagen}")
        cv2.imwrite(ruta_guardado, imagen_resultado)
        print(f"Procesada {nombre_imagen}: Mejor threshold1={mejor_t1}, Mejor threshold2={mejor_t2}")

print("Procesamiento completado. Revisa la carpeta 'resultados/'.")