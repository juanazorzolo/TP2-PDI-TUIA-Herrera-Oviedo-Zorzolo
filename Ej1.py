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

"""PARA PROCESAR LA IMAGEN SIN REPETIR SIEMPRE LO MISMO (se le tiene que pasar parametros de 
canny y kernel para cada dado, por eso no me convence, porque no estaria tan automatizado)"""
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

"""IDENTIFICA DADOS Y MONEDAS"""
# Crear carpetas para guardar imágenes
os.makedirs("monedas", exist_ok=True)
os.makedirs("dados", exist_ok=True)

# Cargar y procesar la imagen
img = cv2.imread('monedas.jpg')
contornos, imagen_procesada = procesar_imagen(img, 80, 180, 7)

# Variables para conteo
conteo_monedas = 0
conteo_dados = 0

# Procesar cada contorno detectado
for i, contour in enumerate(contornos):
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
    if factor_forma > 0.8:  # Umbral para formas redondeadas
        tipo = "MONEDA"
        conteo_monedas += 1
        carpeta = "monedas"
    else:
        tipo = "DADO"
        conteo_dados += 1
        carpeta = "dados"
    
    # Guardar la imagen recortada en la carpeta correspondiente
    ruta_archivo = os.path.join(carpeta, f"{tipo}_{i}.jpg")
    cv2.imwrite(ruta_archivo, roi)
    print(f"Guardado: {ruta_archivo}, Área={area}, Factor de Forma={factor_forma:.2f}")
    
    # Dibujar el contorno y mostrar clasificación en la imagen original
    color = (0, 255, 0) if tipo == "MONEDA" else (255, 0, 0)
    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
    cv2.putText(img, tipo, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# Mostrar resultados
imshow(img, title="Clasificación de objetos")
print(f"Total monedas: {conteo_monedas}")
print(f"Total dados: {conteo_dados}")



#-----------------------------------------------------------------------
#-----------------------------------------------------------------------



"""CODIGO OPTMIZADO PERO SE LE TIENE QUE PASAR LOS PARAMETROS DE CANNY (no esta tan bueno eso porque no estaría automatizado)"""
def procesar_imagen_dado(ruta_imagen, umbral_canny_1, umbral_canny_2):
    # Cargar la imagen
    img_dado = cv2.imread(ruta_imagen)
    imagen = cv2.cvtColor(img_dado, cv2.COLOR_BGR2GRAY)
    
    # Mostrar imagen original
    #imshow(imagen, title="Imagen Original")

    # Aplicar desenfoque gaussiano para reducir ruido
    img_desenfoque = cv2.GaussianBlur(imagen, (5, 5), 0)
    #imshow(img_desenfoque, title="Imagen Desenfocada")

    # Detectar bordes con Canny
    img_bordes = cv2.Canny(img_desenfoque, umbral_canny_1, umbral_canny_2)
    #imshow(img_bordes, title="Bordes Detectados con Canny")

    # Realizar operaciones morfológicas para limpiar bordes
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    img_dilatada = cv2.dilate(img_bordes, k, iterations=7)
    #imshow(img_dilatada, title="Imagen Dilatada")

    img_limpia = cv2.erode(img_dilatada, k, iterations=3)
    #imshow(img_limpia, title="Imagen Limpiada")

    # Umbralización y detección de contornos
    _, thresh_otsu = cv2.threshold(img_limpia, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours_otsu, _ = cv2.findContours(thresh_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibuja todos los contornos sobre la imagen original
    img_contornos = imagen.copy()
    cv2.drawContours(img_contornos, contours_otsu, -1, (0, 255, 0), 2)
    #imshow(img_contornos, title="Contornos Dibujados")

    # Contar el número de contornos
    print(f"Total de contornos detectados: {len(contours_otsu)}")

    # Variables para conteo
    nro_dado = 0

    # Procesar cada contorno detectado
    for i, contour in enumerate(contours_otsu):
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

# Llamada a la función para procesar la imagen del dado 10
nro_dado1 = procesar_imagen_dado('dados/DADO_10.jpg', umbral_canny_1=100, umbral_canny_2=452)

# Llamada a la función para procesar la imagen del dado 18
nro_dado2 = procesar_imagen_dado('dados/DADO_18.jpg', umbral_canny_1=280, umbral_canny_2=460)

total = nro_dado1 + nro_dado2
print("Total de los dados: ", total)


#-----------------------------------------------------------------------
#-----------------------------------------------------------------------


"""IDENTIFICA EL TIPO DE MONEDA"""
""" OJO QUE ESTA LO DE VICKY, ACOMODAR!!!! """

# Crear carpetas para guardar imágenes
os.makedirs("monedas", exist_ok=True)
os.makedirs("dados", exist_ok=True)

# Cargar la imagen
img = cv2.imread('monedas.jpg')
img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Aplicar desenfoque gaussiano para reducir ruido
img_desenfoque = cv2.GaussianBlur(img_gris, (5, 5), 0)

# Detectar bordes con Canny
img_bordes = cv2.Canny(img_desenfoque, 80, 180)

# Realizar operaciones morfológicas para limpiar bordes
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
img_dilatada = cv2.dilate(img_bordes, k, iterations=7)
img_limpia = cv2.erode(img_dilatada, k, iterations=3)

# Umbralización y detección de contornos
_, thresh_otsu = cv2.threshold(img_limpia, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
contours_otsu, _ = cv2.findContours(thresh_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Variables para conteo
conteo_monedas = 0
conteo_dados = 0
total_dados = 0

# Variables para almacenar áreas y coordenadas de monedas
areas_monedas = []
coordenadas_monedas = []

# Procesar cada contorno detectado y clasificar como dado o moneda
for i, contour in enumerate(contours_otsu):
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
    
    # Clasificar si es dado o moneda
    if factor_forma < 0.8 and area < 85000.0:  
        tipo = "dado"
        conteo_dados += 1
        carpeta = "dados"
        
        # Detectar círculos dentro del dado (caras del dado)
        gris_dado = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gris_dado = cv2.GaussianBlur(gris_dado, (9, 9), 2)
        círculos = cv2.HoughCircles(gris_dado, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=10, maxRadius=40)
        
        # Filtrar y contar círculos detectados
        if círculos is not None:
            círculos = np.uint16(np.around(círculos))
            num_círculos = len(círculos[0])
            print(f"Dado {i} - Número dado: {num_círculos}")
            cv2.putText(img, str(num_círculos), (x + w // 2, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        

            # Sumar el número de puntos detectados (valor del dado)
            total_dados += num_círculos

    else:
        tipo = "moneda"
        conteo_monedas += 1
        carpeta = "monedas"
        areas_monedas.append(area)  # Solo guardar áreas de las monedas
        coordenadas_monedas.append((x, y, w, h))  # Guardar coordenadas de las monedas
    
    # Guardar la imagen recortada en la carpeta correspondiente
    ruta_archivo = os.path.join(carpeta, f"{tipo}_{i}.jpg")
    cv2.imwrite(ruta_archivo, roi)
    print(f"Guardado: {ruta_archivo}, Área={area}, Factor de Forma={factor_forma:.2f}")
    
    # Dibujar el contorno y mostrar clasificación en la imagen original
    color = (0, 255, 0) if tipo == "moneda" else (255, 0, 0)
    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
    cv2.putText(img, tipo, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# Aplicar KMeans para clasificar las monedas en 3 grupos según el área
if areas_monedas:
    areas_monedas = np.array(areas_monedas).reshape(-1, 1)
    kmeans = KMeans(n_clusters=3, random_state=0)  # Ajusta n_clusters según sea necesario
    labels = kmeans.fit_predict(areas_monedas)

    # Crear una nueva imagen para mostrar los clusters
    img_clusters = img.copy()  # Hacer una copia para no sobreescribir la imagen original
    
    # Dibujar los clústeres en la imagen de los clusters
    total_dinero = 0  # Variable para almacenar el total de dinero
    for idx, (x, y, w, h) in enumerate(coordenadas_monedas):
        label = labels[idx]  # Asignar el cluster al que pertenece la moneda
        if label == 0:
            color = (0, 255, 0)  # Verde para el primer cluster (1 peso)
            valor = 1  # 1 peso
        elif label == 1:
            color = (0, 0, 255)  # Rojo para el segundo cluster (10 centavos)
            valor = 0.10 # 10 centavos
        else:
            color = (255, 255, 0)  # Azul para el tercer cluster (50 centavos)
            valor = 0.50  # 50 centavos
        
        # Imprimir el cluster al que pertenece la moneda
        print(f"Moneda {idx + 1} - Cluster {label} - Valor: {valor} pesos")
        
        # Sumar el valor de la moneda al total
        total_dinero += valor

        # Dibujar el rectángulo y el texto en la imagen de los clusters
        cv2.rectangle(img_clusters, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img_clusters, f"Cluster {label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Mostrar la imagen con los clusters
    imshow(img_clusters, title="Clusters de las Monedas")

    # Imprimir el total de dinero calculado
    print(f"Total de dinero: {total_dinero} pesos")

# Mostrar resultados en la imagen original
imshow(img, title="Clasificación de objetos")
print(f"Total monedas: {conteo_monedas}")
print(f"Total dados: {conteo_dados}")
print(f"suma dados: {total_dados}")
# Mantener abiertas las ventanas hasta que el usuario las cierre
plt.show(block=True)