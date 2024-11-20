import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont

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



"""IDENTIFICA NRO DEL DADO 10 (el que es 3)"""
# Cargar la imagen
img_dado = cv2.imread('dados/DADO_10.jpg')
contornos, imagen_procesada = procesar_imagen(img_dado, 100, 142, 2)

# Dibuja todos los contornos sobre la imagen original
img_contornos = img_dado.copy()
cv2.drawContours(img_contornos, contornos, -1, (0, 255, 0), 2)
imshow(img_contornos, title="Contornos Dibujados")

print(len(contornos))

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

print(nro_dado)



#-----------------------------------------------------------------------
#-----------------------------------------------------------------------



"""IDENTIFICA NRO DEL DADO 18 (el que es 5)"""
# Cargar la imagen
img_dado = cv2.imread('dados/DADO_18.jpg')
contornos, imagen_procesada = procesar_imagen(img, 280, 460, 2)

# Dibuja todos los contornos sobre la imagen original
img_contornos = img_dado.copy()
cv2.drawContours(img_contornos, contornos, -1, (0, 255, 0), 2)
imshow(img_contornos, title="Contornos Dibujados")

print(len(contornos))

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

print(nro_dado)


#-----------------------------------------------------------------------
#-----------------------------------------------------------------------


"""CODIGO OPTMIZADO PERO SE LE TIENE QUE PASAR LOS PARAMETROS DE CANNY (no esta tan bueno eso porque no estaría automatizado)"""
def procesar_imagen_dado(ruta_imagen, umbral_canny_1=100, umbral_canny_2=452, area_minima=100, factor_forma_min=0.5):
    # Cargar la imagen
    img_dado = cv2.imread(ruta_imagen)
    imagen = cv2.cvtColor(img_dado, cv2.COLOR_BGR2GRAY)
    
    # Mostrar imagen original
    imshow(imagen, title="Imagen Original")

    # Aplicar desenfoque gaussiano para reducir ruido
    img_desenfoque = cv2.GaussianBlur(imagen, (5, 5), 0)
    imshow(img_desenfoque, title="Imagen Desenfocada")

    # Detectar bordes con Canny
    img_bordes = cv2.Canny(img_desenfoque, umbral_canny_1, umbral_canny_2)
    imshow(img_bordes, title="Bordes Detectados con Canny")

    # Realizar operaciones morfológicas para limpiar bordes
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    img_dilatada = cv2.dilate(img_bordes, k, iterations=7)
    imshow(img_dilatada, title="Imagen Dilatada")

    img_limpia = cv2.erode(img_dilatada, k, iterations=3)
    imshow(img_limpia, title="Imagen Limpiada")

    # Umbralización y detección de contornos
    _, thresh_otsu = cv2.threshold(img_limpia, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours_otsu, _ = cv2.findContours(thresh_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibuja todos los contornos sobre la imagen original
    img_contornos = imagen.copy()
    cv2.drawContours(img_contornos, contours_otsu, -1, (0, 255, 0), 2)
    imshow(img_contornos, title="Contornos Dibujados")

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
        if area < area_minima:
            continue
        
        # Calcular el factor de forma
        factor_forma = (4 * np.pi * area) / (perimetro ** 2) if perimetro > 0 else 0
        
        if factor_forma > factor_forma_min:  # Umbral para formas redondeadas
            nro_dado += 1

    print(f"Número de dados detectados: {nro_dado}")

# Llamada a la función para procesar la imagen del dado 10
procesar_imagen_dado('dados/DADO_10.jpg', umbral_canny_1=100, umbral_canny_2=452)

# Llamada a la función para procesar la imagen del dado 18
procesar_imagen_dado('dados/DADO_18.jpg', umbral_canny_1=280, umbral_canny_2=460)



#-----------------------------------------------------------------------
#-----------------------------------------------------------------------


"""IDENTIFICA EL TIPO DE MONEDA"""