import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont
import shutil

# Mostrar imágenes
def imshow(img, new_fig=True, title=None, color_img=False, blocking=True, colorbar=True, ticks=False):
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
        if blocking:
            plt.show(block=True)  # Bloquea el flujo de ejecución hasta que se cierre la ventana

# Eliminar carpetas y su contenido para reiniciar la ejecución
def eliminar_carpetas_contenido(carpeta1, carpeta2, carpeta3, carpeta4, carpeta5, carpeta6):
    carpetas = [carpeta1, carpeta2, carpeta3, carpeta4, carpeta5, carpeta6]
    for carpeta in carpetas:
        if os.path.exists(carpeta):
            shutil.rmtree(carpeta)
            print(f"Carpeta '{carpeta}' y su contenido han sido eliminados.")
        else:
            print(f"La carpeta '{carpeta}' no existe.")

#------------------------------

# Encontrar los mejores parámetros de canny y procesar la imagen 
def ajustar_canny(imagen_gray, rango_threshold1, rango_threshold2, area_min=300, area_max=30000):
    mejor_threshold1 = 0
    mejor_threshold2 = 0
    mejor_puntaje = 0
    lista_cont = []

    for t1 in rango_threshold1:
        for t2 in rango_threshold2:
            # Procesamiento de la imagen
            img_canny = cv2.Canny(imagen_gray, t1, t2, apertureSize=3, L2gradient=True)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 4))
            img_dilatacion = cv2.dilate(img_canny, kernel, iterations=1)
            img_erosion = cv2.erode(img_dilatacion, kernel, iterations=1)
            img_cerrada = cv2.morphologyEx(img_erosion, cv2.MORPH_CLOSE, kernel)
            _, bordes_binarios = cv2.threshold(img_cerrada, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            contornos, _ = cv2.findContours(bordes_binarios, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            # Determinar mejor t1 y t2
            puntaje_actual = 0
            for contorno in contornos:
                area = cv2.contourArea(contorno)
                if area_min < area < area_max:
                    puntaje_actual += 1
                    lista_cont.append(contorno)

            if puntaje_actual > mejor_puntaje:
                mejor_puntaje = puntaje_actual
                mejor_threshold1 = t1
                mejor_threshold2 = t2

    return mejor_threshold1, mejor_threshold2, contornos, img_canny

# Filtrar los contornos de la imagen y recuadrar las primeras bounding boxes
def procesar_patente(imagen, todos_contornos):
    contornos_filtrados = []
    bounding_boxes = []
    finales = []

    # Filtrar contornos por su área
    for contorno in todos_contornos:
        area = cv2.contourArea(contorno)
        if 1500 < area < 35000:
            contornos_filtrados.append(contorno)

    # Dibujar contornos en la imagen original
    for contorno in contornos_filtrados:
        x, y, w, h = cv2.boundingRect(contorno)
        bounding_boxes.append((x, y, w, h))
        cv2.rectangle(imagen, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return imagen

# Detectar las bounding boxes de las letras con componentes conectadas
def detectar_caracteres(img, ecualizar=False):
    imagen_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Ecualizar si es True
    if ecualizar:
        imagen_gray = cv2.equalizeHist(imagen_gray)
    
    _, binary = cv2.threshold(imagen_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Detectar componentes conectadas
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    componentes_conectadas = []
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # Filtrar componentes por su área y ratio
        if 17 <= area <= 200 and 1.5 <= h / w <= 3.0:
            componentes_conectadas.append((x, y, w, h))
    return componentes_conectadas

# Combinar las componentes conectadas detectadas con y sin ecualización, eliminando duplicados
def combinar_bounding_boxes(boxes1, boxes2, iou_threshold=0.3):
    combined_boxes = boxes1.copy()

    for box2 in boxes2:
        x2, y2, w2, h2 = box2
        is_duplicate = False  # Identificador de duplicados

        # Comparar box2 con cada bounding box de combined_boxes
        for box1 in combined_boxes:
            x1, y1, w1, h1 = box1

            # Coordenadas de la intersección
            inter_x1 = max(x1, x2)  # Límite izquierdo
            inter_y1 = max(y1, y2)  # Límite superior
            inter_x2 = min(x1 + w1, x2 + w2)  # Límite derecho
            inter_y2 = min(y1 + h1, y2 + h2)  # Límite inferior

            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1) # Area de la intersección
            union_area = w1 * h1 + w2 * h2 - inter_area # Area de la unión

            # Calcular IoU
            iou = inter_area / union_area if union_area > 0 else 0

            # Filtro de duplicado
            if iou > iou_threshold:
                is_duplicate = True
                break

        # Agregar box2 a combined_boxes
        if not is_duplicate:
            combined_boxes.append(box2)
    return combined_boxes

# Filtrar cajas cercanas para quedarse con las bounding boxes de las letras
def seleccionar_cajas_cercanas(bboxes):
    valid_boxes = []

    for i, (x1, y1, w1, h1) in enumerate(bboxes):
        centro1 = (x1 + w1 // 2, y1 + h1 // 2)  # Centro de la bounding box actual
        is_valid = False  # Identificador de bounding box válida

        # Comparar bounding box actual con todas las demás
        for j, (x2, y2, w2, h2) in enumerate(bboxes):
            if i != j:
                centro2 = (x2 + w2 // 2, y2 + h2 // 2)  # Centro de la bounding box comparada

                # Calcular las distancias horizontal y vertical entre los centros
                dist_x = abs(centro1[0] - centro2[0])
                diff_y = abs(centro1[1] - centro2[1])

                # Filtrar bounding box válida
                if dist_x <= 15 and diff_y <= 10:
                    is_valid = True
                    break
        if is_valid:
            valid_boxes.append((x1, y1, w1, h1))

    # Devolver la lista de bounding boxes válidas
    return valid_boxes

# Extraer y guardar bounding box de cada letra
def extraer_y_mostrar_bboxes(img, cajas_cercanas):
    for i, (x, y, w, h) in enumerate(cajas_cercanas):
        
        # Extraer la región de la imagen correspondiente a la bounding box y aumentar su tamaño
        letra_recortada = img[y:y + h, x:x + w]
        letra_recortada_aumentada = cv2.resize(letra_recortada, (w * 5, h * 5))

        imagen_fondo = 255 * np.ones((h * 5, w * 5, 3), dtype=np.uint8)

        # Pegar imagen recortada en el centro de la imagen de fondo
        imagen_fondo[
            (imagen_fondo.shape[0] - letra_recortada_aumentada.shape[0]) // 2:
            (imagen_fondo.shape[0] - letra_recortada_aumentada.shape[0]) // 2 + letra_recortada_aumentada.shape[0],
            (imagen_fondo.shape[1] - letra_recortada_aumentada.shape[1]) // 2:
            (imagen_fondo.shape[1] - letra_recortada_aumentada.shape[1]) // 2 + letra_recortada_aumentada.shape[1]
        ] = letra_recortada_aumentada

        nombre_archivo = f"letra_{i+1}.jpg"
        ruta_guardado = os.path.join("LETRAS/", nombre_archivo)
        cv2.imwrite(ruta_guardado, imagen_fondo)

# Crear imagen con las bounding box de cada letra para mejor visualización
def pegar_y_guardar_patente(nombre_imagen="patente"):
    # Obtener imágenes en la carpeta LETRAS
    imagenes = [f for f in os.listdir("LETRAS") if f.endswith('.jpg')]

    if not imagenes:
        # Si la carpeta LETRAS está vacía, crear una imagen blanca
        imagen_final = 255 * np.ones((500, 500, 3), dtype=np.uint8)
        ruta_guardado = os.path.join("patente_caracteres/", f"caracteres_{nombre_imagen}.jpg")
        cv2.imwrite(ruta_guardado, imagen_final)
        print(f"La carpeta LETRAS está vacía. Imagen blanca guardada en {ruta_guardado}")
        return

    imagenes_letras = []
    ancho_total = 0
    altura_maxima = 0

    for imagen in imagenes:
        ruta_imagen = os.path.join("LETRAS", imagen)
        letra = cv2.imread(ruta_imagen)
        imagenes_letras.append(letra)

        # Calcular el ancho total y la altura máxima de la imagen final
        ancho_total += letra.shape[1] + 40 # 1cm en píxeles
        altura_maxima = max(altura_maxima, letra.shape[0])

 # Crear la imagen final
    espacio_titulo = 100  # Altura reservada para el texto
    imagen_final = 255 * np.ones((altura_maxima + espacio_titulo, ancho_total, 3), dtype=np.uint8)

    # Agregar el título centrado en la parte superior
    texto_titulo = f"Caracteres detectados"
    fuente = cv2.FONT_HERSHEY_DUPLEX
    (ancho_texto, alto_texto), _ = cv2.getTextSize(texto_titulo, fuente, 1, 2)
    x_texto = max((ancho_total - ancho_texto) // 2, 10)
    y_texto = 50  # Margen superior
    cv2.putText(imagen_final, texto_titulo, (x_texto, y_texto), fuente, 1, (0, 0, 0), 2)

    # Pegar cada letra en la imagen final
    posicion_h = 0
    for letra in imagenes_letras:
        imagen_final[espacio_titulo:espacio_titulo + letra.shape[0], posicion_h:posicion_h + letra.shape[1]] = letra
        posicion_h += letra.shape[1] + 40  # Mover el offset para la siguiente letra

    ruta_guardado = os.path.join("patente_caracteres/", f"caracteres_{nombre_imagen}.jpg")
    cv2.imwrite(ruta_guardado, imagen_final)
    print(f"Imagen final guardada en {ruta_guardado}")

# Encontrar patente completa en base a los carcateres detectados
def encontrar_rectangulo_patente(bounding_boxes):
    if not bounding_boxes:
        return None
    
    # Encontrar coordenadas mínimas y máximas
    x_min = min([x for x, y, w, h in bounding_boxes]) - 12
    y_min = min([y for x, y, w, h in bounding_boxes]) - 12
    x_max = max([x + w for x, y, w, h in bounding_boxes]) + 12
    y_max = max([y + h for x, y, w, h in bounding_boxes]) + 12
    
    p = max(0, x_min), max(0, y_min), x_max, y_max
    return p

eliminar_carpetas_contenido("resultados", "bb_detectadas", "FINALES", "LETRAS", "patente_caracteres", "PATENTES_DETECTADAS")
os.makedirs("resultados/", exist_ok=True)
os.makedirs("bb_detectadas/", exist_ok=True)
os.makedirs("FINALES/", exist_ok=True)
os.makedirs("PATENTES_DETECTADAS/", exist_ok=True)
os.makedirs("LETRAS/", exist_ok=True)
os.makedirs("patente_caracteres/", exist_ok=True)

for filename in os.listdir("patentes/"):
    img_path = os.path.join("patentes/", filename)
    img = cv2.imread(img_path)
    imagen_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detectar bordes con Canny
    mejor_t1, mejor_t2, todos_contornos, guardar = ajustar_canny(imagen_gray, range(50, 200, 10), range(100, 400, 10))
    imagen_resultado = procesar_patente(img.copy(), todos_contornos)
    
    # Guardar imagen con bordes detectados, es para corrección ya que no se utiliza posteriormente
    ruta_guardado = os.path.join("resultados/", f"boundingb_{filename}")
    cv2.imwrite(ruta_guardado, imagen_resultado)

    # Detectar caracteres con y sin ecualización
    boxes_no_eq = detectar_caracteres(imagen_resultado, ecualizar=False)
    boxes_eq = detectar_caracteres(imagen_resultado, ecualizar=True)

    # Combinar las cajas de caracteres detectados
    combined_boxes = combinar_bounding_boxes(boxes_no_eq, boxes_eq)

    # Dibujar las cajas combinadas sobre la imagen
    for box in combined_boxes:
        x, y, w, h = box
        cv2.rectangle(imagen_resultado, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Guardar la imagen con las cajas de caracteres detectados
    ruta_guardado_comb = os.path.join("bb_detectadas/", f"combined_{filename}")
    cv2.imwrite(ruta_guardado_comb, imagen_resultado)

    # Seleccionar las bounding boxes más cercanas entre sí
    cajas_cercanas = seleccionar_cajas_cercanas(combined_boxes)

    img_with_boxes = img.copy()
    img_with_boxes2 = img.copy()
    # Dibujar las cajas seleccionadas en la imagen original
    for box in cajas_cercanas:
        x, y, w, h = box
        cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # Guardar la imagen final con las cajas más cercanas seleccionadas
    ruta_guardado_final = os.path.join("FINALES", f"final_{filename}")
    cv2.imwrite(ruta_guardado_final, img_with_boxes)

    # Encontrar el rectángulo que contiene la patente
    patente = encontrar_rectangulo_patente(cajas_cercanas)
    print(patente)
    if patente and all(coord >= 0 for coord in patente):
        # Si se encuentra la patente, se dibuja y se guarda la imagen recortada
        x_min, y_min, x_max, y_max = patente
        cv2.rectangle(img_with_boxes2, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Verde
        cropped = img[y_min:y_max, x_min:x_max]
        patente_guardado_final = os.path.join("PATENTES_DETECTADAS/", f"patente_{filename}")
        cv2.imwrite(patente_guardado_final, cropped)
    else:
        print("No se encontró la patente")
        continue

    # Extraer y mostrar las bounding boxes, para corrección
    extraer_y_mostrar_bboxes(img, cajas_cercanas)

    # Después de procesar las bounding boxes, pegar las letras y guardarlas en una nueva imagen
    pegar_y_guardar_patente(nombre_imagen=filename.split('.')[0])

    # Limpiar la carpeta LETRAS para el procesamiento de la siguiente patente.
    for archivo in os.listdir("LETRAS"):
        archivo_path = os.path.join("LETRAS", archivo)
        os.remove(archivo_path)

print("Procesamiento completado.")