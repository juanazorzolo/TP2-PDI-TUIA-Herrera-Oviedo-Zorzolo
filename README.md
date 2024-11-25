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
- `resultados/`: Carpeta de salida principal.
- `bb_detectadas/`: Almacena las bounding boxes detectadas.
- `FINALES/`: Contiene las imágenes finales con las patentes procesadas.
- `LETRAS/`: Guarda las imágenes recortadas de cada letra detectada.
- `patente_caracteres/`: Almacena las composiciones finales de caracteres detectados.


## Funcionalidades

### Funciones principales

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



