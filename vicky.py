import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

def imshow(img, title="Image"):
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()

# Cargar la imagen
image = cv2.imread('monedas.jpg')

# Convertir a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar ecualización local del histograma 
clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(5,5))
equ = clahe.apply(gray)

# Suavizado con filtro Gaussiano para reducir ruido
blurred = cv2.GaussianBlur(equ, (9, 9), 0)

# Umbralado adaptativo para separar objetos del fondo
thresh = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY_INV, 27, 2
)

# Operación morfológica para eliminar ruido
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
morph_close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
morph_open = cv2.morphologyEx(morph_close, cv2.MORPH_OPEN, kernel)  # Apertura para quitar ruido externo
#morph_dilate = cv2.dilate(morph_open, kernel, iterations=1)
morph_close = cv2.morphologyEx(morph_open, cv2.MORPH_CLOSE, kernel)  # Cierre para cerrar  

# Suavizado con filtro Gaussiano para reducir ruido
bl = cv2.GaussianBlur(morph_close, (1, 1), 0)

# Detección de contornos
contours, _ = cv2.findContours(bl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Visualización de contornos en la imagen original
contour_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
imshow(contour_img, title="Contornos Detectados")

"""""

# Calcular el área de cada contorno
areas = np.array([cv2.contourArea(contour) for contour in contours]).reshape(-1, 1)

# Clasificación con K-means (2 clusters: dados y monedas)
kmeans = KMeans(n_clusters=2, random_state=0).fit(areas)
labels = kmeans.labels_

# Separar los contornos según los clusters de K-means
monedas = [contours[i] for i in range(len(contours)) if labels[i] == 0]
dados = [contours[i] for i in range(len(contours)) if labels[i] == 1]

# Visualización de resultados en la imagen original
contour_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.drawContours(contour_img, monedas, -1, (255, 0, 0), 2)  # Monedas en azul
cv2.drawContours(contour_img, dados, -1, (0, 0, 255), 2)  # Dados en rojo
imshow(contour_img, title="Monedas y Dados Detectados (K-means)")

# Contar monedas y dados detectados
print(f"Cantidad de Monedas Detectadas: {len(monedas)}")
print(f"Cantidad de Dados Detectados: {len(dados)}")

# Detección de los valores de los dados (si es necesario)
valores_dados = []
for dado in dados:
    epsilon = 0.02 * cv2.arcLength(dado, True)
    approx = cv2.approxPolyDP(dado, epsilon, True)
    dado_img = np.zeros_like(gray)
    cv2.drawContours(dado_img, [approx], -1, 255, -1)
    _, dado_threshold = cv2.threshold(dado_img, 127, 255, cv2.THRESH_BINARY)
    dado_circles = cv2.HoughCircles(dado_threshold, cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=50, param2=30, minRadius=5, maxRadius=15)
    
    if dado_circles is not None:
        dado_circles = np.round(dado_circles[0, :]).astype("int")
        valores_dados.append(len(dado_circles))
    else:
        valores_dados.append(0)

for i, valor in enumerate(valores_dados):
    print(f"Valor del Dado {i + 1}: {valor}")

print("Clasificación y conteo completados.")
"""