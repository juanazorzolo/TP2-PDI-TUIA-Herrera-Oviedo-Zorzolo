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

""" PROCESO ITERATIVO PARA CANNY """
def ajustar_canny(imagen_gray, rango_threshold1, rango_threshold2, area_min=2000, area_max=15000):
    mejor_threshold1 = 0
    mejor_threshold2 = 0
    mejor_puntaje = 0
    mejor_bordes = None

    for t1 in rango_threshold1:
        for t2 in rango_threshold2:
            # Aplicar Canny con los parámetros actuales
            bordes = cv2.Canny(imagen_gray, t1, t2)
            
            # Operaciones morfológicas para cerrar bordes pequeños
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
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


"""PROCESAR TODAS LAS IMÁGENES DE LA CARPETA PATENTES Y GUARDAR EN LA CARPETA RESULTADOS"""
carpeta_imagenes = "patentes/" 
rango_threshold1 = range(50, 200, 10) # Para pasarle a la función ajustar_canny y que encuentre el mejor
rango_threshold2 = range(100, 400, 10) # Lo mismo que arriba

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
        
        # Guardar resultado
        ruta_guardado = os.path.join("resultados/", f"bordes_{nombre_imagen}")
        cv2.imwrite(ruta_guardado, bordes_optimizados)
        print(f"Procesada {nombre_imagen}: Mejor threshold1={mejor_t1}, Mejor threshold2={mejor_t2}")

print("Procesamiento completado. Revisa la carpeta 'resultados/'.")

# Añadimos las funciones después del ajuste de Canny
def detectar_patente(imagen_gray, bordes, rel_aspecto=2.28, tol_aspecto=0.2, area_min=2000, area_max=15000):
    contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    patente_roi = None

    for contorno in contornos:
        x, y, w, h = cv2.boundingRect(contorno)
        area = cv2.contourArea(contorno)
        relacion_aspecto = w / h

        if area_min < area < area_max and (rel_aspecto - tol_aspecto) < relacion_aspecto < (rel_aspecto + tol_aspecto):
            patente_roi = imagen_gray[y:y+h, x:x+w]
            break  # Suponemos una sola patente por imagen

    return patente_roi

def segmentar_caracteres(patente_roi):
    _, binaria = cv2.threshold(patente_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    caracteres = []

    for contorno in contornos:
        x, y, w, h = cv2.boundingRect(contorno)
        if h > 10:  # Filtrar ruido basado en altura mínima
            caracteres.append((x, y, w, h))

    caracteres = sorted(caracteres, key=lambda c: c[0])  # Ordenar por posición horizontal
    return caracteres

# Procesar cada imagen incluyendo la detección de la patente y segmentación de caracteres
for nombre_imagen in os.listdir(carpeta_imagenes):
    if nombre_imagen.endswith(".png") or nombre_imagen.endswith(".jpg"):
        ruta_imagen = os.path.join(carpeta_imagenes, nombre_imagen)
        imagen = cv2.imread(ruta_imagen)
        imagen_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        
        # Ajustar Canny
        mejor_t1, mejor_t2, bordes_optimizados = ajustar_canny(imagen_gray, rango_threshold1, rango_threshold2)
        
        # Detectar la patente
        patente_roi = detectar_patente(imagen_gray, bordes_optimizados)
        if patente_roi is not None:
            # Guardar la región de la patente
            ruta_patente = os.path.join("resultados/", f"patente_{nombre_imagen}")
            cv2.imwrite(ruta_patente, patente_roi)
            print(f"Patente detectada en {nombre_imagen}, guardada en resultados/")
            
            # Segmentar caracteres
            caracteres = segmentar_caracteres(patente_roi)
            for idx, (x, y, w, h) in enumerate(caracteres):
                caracter = patente_roi[y:y+h, x:x+w]
                ruta_caracter = os.path.join("resultados/", f"caracter_{nombre_imagen}_{idx}.png")
                cv2.imwrite(ruta_caracter, caracter)
                print(f"Caracter {idx} de {nombre_imagen} guardado.")
        else:
            print(f"No se detectó patente en {nombre_imagen}.")
