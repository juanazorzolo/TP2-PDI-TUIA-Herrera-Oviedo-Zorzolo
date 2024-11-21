import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans

"""MOSTRAR IMAGENES"""
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


#-----------------------------------------------------------------------
#-----------------------------------------------------------------------


"""PROCESAR LA IMAGEN""" #hay que pasarle los parametros de canny y kernel
def procesar_imagen(imagen, t1, t2, pk): #t1 y t2 son los parametros de canny. pk es del kernel
    img_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Aplicar desenfoque gaussiano para reducir ruido
    img_desenfoque = cv2.GaussianBlur(img_gris, (5, 5), 0)
    #imshow(img_desenfoque)

    # Detectar bordes con Canny
    img_bordes = cv2.Canny(img_desenfoque, t1, t2)
    #imshow(img_bordes)

    # Realizar operaciones morfológicas para limpiar bordes
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pk, pk))
    img_dilatada = cv2.dilate(img_bordes, k, iterations=7)
    img_limpia = cv2.erode(img_dilatada, k, iterations=3)

    # Umbralización y detección de contornos
    _, thresh_otsu = cv2.threshold(img_limpia, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours_otsu, _ = cv2.findContours(thresh_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours_otsu, imagen


#-----------------------------------------------------------------------
#-----------------------------------------------------------------------


"""IDENTIFICA DADOS Y MONEDAS"""
def clasificar_tipo_objeto(img):
    contornos, imagen_procesada = procesar_imagen(img, 80, 180, 7)

    # Variables para conteo y almacenamiento de información
    conteo_monedas = 0
    conteo_dados = 0
    total_dados = 0
    areas_monedas = []
    coordenadas_monedas = []

    for i, contour in enumerate(contornos):
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

        # Clasificar como dado o moneda
        if factor_forma < 0.8 and area < 85000.0:
            tipo = "dado"
            conteo_dados += 1
            carpeta = "dados"
            color = (255, 0, 0)  # Azul

        else:
            tipo = "moneda"
            conteo_monedas += 1
            carpeta = "monedas"
            areas_monedas.append(area)
            coordenadas_monedas.append((x, y, w, h))
            color = (0, 255, 0)  # Verde

        # Guardar la imagen recortada en la carpeta correspondiente
        ruta_archivo = os.path.join(carpeta, f"{tipo}_{i}.jpg")
        cv2.imwrite(ruta_archivo, roi)
        print(f"Guardado: {ruta_archivo}")

        # Dibujar el rectángulo y el texto en la imagen original
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, tipo, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    imshow(img)
    return conteo_monedas, conteo_dados, total_dados, areas_monedas, coordenadas_monedas


#-----------------------------------------------------------------------
#-----------------------------------------------------------------------


"""IDENTIFICA EL TIPO DE DADO""" #hay que pasarle los parametros de canny
def procesar_imagen_dado(imagen_dado, umbral_canny_1, umbral_canny_2):
    contornos, imagen_procesada = procesar_imagen(imagen_dado, umbral_canny_1, umbral_canny_2, 2)

    # Dibuja todos los contornos sobre la imagen original
    img_contornos = imagen_dado.copy()
    cv2.drawContours(img_contornos, contornos, -1, (0, 255, 0), 2)
    imshow(img_contornos, title="Contornos Dibujados")

    # Contar el número de contornos
    #print(f"Total de contornos detectados: {len(contornos)}")

    # Variables para conteo
    nro_dado = 0

    # Procesar cada contorno detectado
    for i, contour in enumerate(contornos):
        # Calcular área y perímetro
        area = cv2.contourArea(contour)
        perimetro = cv2.arcLength(contour, True)
        
        # Ignorar objetos pequeños (ruido)
        if area < 100:
            continue
        
        # Calcular el factor de forma
        factor_forma = (4 * np.pi * area) / (perimetro ** 2) if perimetro > 0 else 0
        
        if factor_forma > 0.5:  # Umbral para formas redondeadas
            nro_dado += 1

    print(f"Número del dado detectado: {nro_dado}")
    return nro_dado


#-----------------------------------------------------------------------
#-----------------------------------------------------------------------


"""IDENTIFICA EL TIPO DE MONEDA"""
def detectar_tipo_monedas(img, areas_monedas, coordenadas_monedas):
    if areas_monedas:
        areas_monedas = np.array(areas_monedas).reshape(-1, 1)
        kmeans = KMeans(n_clusters=3, random_state=0)  # Ajusta n_clusters según sea necesario
        labels = kmeans.fit_predict(areas_monedas)

        # Crear una nueva imagen para mostrar los clusters
        img_clusters = img.copy()
        total_dinero = 0

        for idx, (x, y, w, h) in enumerate(coordenadas_monedas):
            label = labels[idx]
            if label == 0:
                color = (0, 255, 0)  # Verde (1 peso)
                valor = 1.0
            elif label == 1:
                color = (0, 0, 255)  # Rojo (10 centavos)
                valor = 0.10
            else:
                color = (255, 255, 0)  # Azul (50 centavos)
                valor = 0.50

            total_dinero += valor
            # Dibujar el rectángulo y el texto en la imagen
            cv2.rectangle(img_clusters, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img_clusters, f"Cluster {label}, Valor: {valor:.2f} pesos", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Devolver la imagen y el total de dinero
        return img_clusters, total_dinero


#-----------------------------------------------------------------------
#-----------------------------------------------------------------------


"""PROGRAMA PRINCIPAL"""
# Crear carpetas para guardar imágenes
os.makedirs("monedas", exist_ok=True)
os.makedirs("dados", exist_ok=True)

# Cargar imagen
img = cv2.imread('monedas.jpg')

# Clasificar y guardar los objetos
conteo_monedas, conteo_dados, total_dados, areas_monedas, coordenadas_monedas = clasificar_tipo_objeto(img)
print("Cantidad de monedas: ", conteo_monedas)
print("Cantidad de dados: ", conteo_dados)

# Detectar nro del dado 1
img_dado1 = cv2.imread('dados/DADO_10.jpg')
nro_dado1 = procesar_imagen_dado(img_dado1, umbral_canny_1=100, umbral_canny_2=452)

# Detectar nro del dado 2
img_dado2 = cv2.imread('dados/DADO_18.jpg')
nro_dado2 = procesar_imagen_dado(img_dado2, umbral_canny_1=280, umbral_canny_2=460)

total = nro_dado1 + nro_dado2
print("Total de los dados: ", total)

# Detectar el tipo de monedas y devolver la imagen final
img_resultado, total_dinero = detectar_tipo_monedas(img, areas_monedas, coordenadas_monedas)
print(f"Total de dinero: {total_dinero} pesos")

# Mostrar la imagen final
plt.imshow(cv2.cvtColor(img_resultado, cv2.COLOR_BGR2RGB))
plt.title("Clasificación Final")
plt.axis("off")
plt.show()

print(f"Total monedas: {conteo_monedas}")
print(f"Total dados: {conteo_dados}")
print(f"Suma dados: {total_dados}")