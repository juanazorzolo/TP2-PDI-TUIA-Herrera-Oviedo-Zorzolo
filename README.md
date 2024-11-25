# Detección y Clasificación de Monedas y Dados

Este proyecto implementa un algoritmo para procesar imágenes que contienen monedas y dados, clasificarlos automáticamente y contar sus valores.

## Descripción del Problema

Se tiene una imagen con monedas y dados colocados sobre un fondo de intensidad no uniforme. El objetivo del algoritmo es:

1. **Segmentar automáticamente las monedas y los dados.**
2. **Clasificar los distintos tipos de monedas y realizar un conteo automático.**
3. **Determinar el valor de la cara superior de cada dado y realizar un conteo automático.**

## Requisitos

Asegúrate de tener las siguientes dependencias instaladas:

- Python 3.8+
- OpenCV
- NumPy
- Matplotlib
- Pillow
- Scikit-learn

Puedes instalarlas con el siguiente comando:

```bash
pip install opencv-python numpy matplotlib pillow scikit-learn
```
## Ejecución del Programa

**Archivos necesarios**:  
- Imagen`monedas.jpg`.  

**Estructura de carpetas**:  
- El programa creará dos carpetas (`monedas` y `dados`) para guardar los objetos segmentados.

**Ejecución**:  
Ejecuta el archivo "Ej2.py". Los resultados se mostrarán en las imágenes procesadas y en la terminal.
   ```bash
   Ej2.py
   ```

## Archivos Generados

- Imágenes recortadas de monedas y dados en las carpetas `monedas` y `dados`.  
- Imagen final procesada con la clasificación visual de los objetos.


## Funciones principales

1. `procesar_imagen(imagen, t1, t2, pk)`:  
   - Convierte la imagen a escala de grises.  
   - Aplica un desenfoque gaussiano para reducir ruido.  
   - Detecta bordes con el método de Canny.  
   - Realiza operaciones morfológicas para limpiar los bordes y segmentar los objetos.  
   - Devuelve los contornos detectados.

2. `clasificar_tipo_objeto(img)`:  
   - Procesa la imagen y segmenta monedas y dados.  
   - Clasifica cada objeto basado en su área y factor de forma.  
   - Guarda las regiones de interés (ROIs) en carpetas separadas (`monedas` y `dados`).  
   - Dibuja rectángulos y etiquetas en la imagen original.

3. `procesar_imagen_dado(imagen_dado, t1, t2)`:  
   - Detecta puntos (caras) en la cara superior de los dados.  
   - Calcula el número total de puntos en la cara visible.

4. `detectar_tipo_monedas(img, areas_monedas, coordenadas_monedas)`:  
   - Aplica el algoritmo de K-Means para clasificar monedas en tres tipos según su tamaño.  
   - Asigna valores monetarios a cada tipo de moneda y calcula el total.

## Resultados

El script muestra los resultados de cada etapa de procesamiento:
- Imágenes con los objetos detectados y clasificados.
- Cantidad total de monedas y dados.
- Número total de puntos en las caras superiores de los dados.
- Total monetario calculado.

## Ejemplo de Salida

- **Cantidad de Monedas**: 17  
- **Cantidad de Dados**: 2
- **Número del dado detectado**: 3
- **Número del dado detectado**: 5
- **Total de los Dados**: 8  
- **Total de Dinero**: 7.40 pesos  




# Detección y Segmentación de Caracteres en Patentes

Este proyecto implementa un sistema para detectar y segmentar caracteres de patentes en imágenes utilizando técnicas de procesamiento de imágenes en Python. El código incluye funciones para preprocesar imágenes, ajustar parámetros de detección, identificar bounding boxes de caracteres y generar imágenes finales con las patentes detectadas.

## Requisitos

Asegúrate de tener las siguientes dependencias instaladas:

- Python 3.8+
- OpenCV
- NumPy
- Matplotlib
- Pillow

Puedes instalarlas con el siguiente comando:

```bash
pip install opencv-python numpy matplotlib pillow
```


---


## Estructura del Proyecto

El proyecto utiliza las siguientes carpetas:

- `patentes/`: Contiene las imágenes de entrada con las patentes a procesar.
- `resultados/`: Almacena las bounding boxes detectadas.
- `bb_detectadas/`: Almacena las bounding boxes detectadas y las bounding boxes de las componentes conectadas .
- `FINALES/`: Contiene las imágenes de entrada con las bounding boxes de cada caracter encontrado .
- `PATENTES_DETECTADAS/`: Contiene las imágenes finales con las patentes procesadas.
- `patente_caracteres/`: Almacena las composiciones finales de caracteres detectados.


## Funciones principales

- **`ajustar_canny(imagen_gray, rango_threshold1, rango_threshold2, area_min, area_max)`**  
  Ajusta los parámetros de Canny para optimizar la detección de bordes.

- **`procesar_patente(imagen, todos_contornos)`**  
  Filtra contornos por área y dibuja bounding boxes en la imagen.

- **`detectar_caracteres(img, ecualizar)`**  
  Detecta bounding boxes de caracteres utilizando componentes conectadas.

- **`combinar_bounding_boxes(boxes1, boxes2, iou_threshold)`**  
  Combina bounding boxes eliminando duplicados basados en el umbral de IoU.

- **`seleccionar_cajas_cercanas(bboxes)`**  
  Filtra las bounding boxes cercanas para identificar caracteres válidos.

- **`extraer_y_mostrar_bboxes(img, cajas_cercanas)`**  
  Recorta y guarda cada letra detectada en la carpeta `LETRAS`.

- **`pegar_y_guardar_patente(nombre_imagen)`**  
  Combina las letras detectadas en una imagen compuesta final.

- **`encontrar_rectangulo_patente(bounding_boxes)`**  
  Calcula las coordenadas de un rectángulo que encierra todas las letras.

## Ejecución

1. Coloca las imágenes de las patentes en la carpeta `patentes/`.
2. Ejecuta el script principal para procesar las imágenes:
   
   ```bash
   Ej2.py
   ```

   
---

## Resultados

- **Bounding boxes detectadas:** Se almacenan en `bb_detectadas/`.
- **Imágenes finales:** Las patentes procesadas se guardan en `PATENTES_DETECTADAS/`.
- **Composición final:** Las imágenes compuestas se almacenan en `patente_caracteres/`.


## Notas

- El código incluye filtros para optimizar la detección según el área y la relación de aspecto de los caracteres.
- Se pueden ajustar los parámetros como `area_min`, `area_max` o los rangos de thresholds en la función `ajustar_canny` para mejorar los resultados según las imágenes.

## Ejemplo

Una vez procesadas las imágenes, las letras detectadas y segmentadas se organizan en un formato legible, permitiendo analizar o exportar los caracteres de las patentes.






