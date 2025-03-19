import cv2
import numpy as np
import pytesseract
from scipy.ndimage import interpolation as inter

def extraer_texto_desde_imagen(ruta_imagen, num):
    """Extrae texto de una imagen aplicando preprocesamiento avanzado para OCR."""
    try:
        # Cargar imagen
        imagen = cv2.imread(ruta_imagen)
        if imagen is None:
            print(f"Error: No se pudo cargar la imagen en {ruta_imagen}")
            return ""

        # 🔹 Corregir inclinación del texto
        imagen = correct_skew(imagen)

        # 🔹 Eliminar líneas antes del preprocesamiento
        imagen = remove_lines(imagen)

        # 🔹 Aplicar CLAHE para mejorar contraste
        imagen = aplicar_CLAHE(imagen)

        # Convertir a escala de grises
        imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

        # Aplicar detección de bordes
        imagen_bordes = detectar_bordes(imagen)

        # Aplicar umbral adaptativo
        imagen_umbral = cv2.adaptiveThreshold(imagen_gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 31, 2)

        # 🔹 Eliminar ruido avanzado
        imagen_umbral = eliminar_ruido_avanzado(imagen_umbral)

        # 🔹 Escalar imagen dinámicamente
        altura, ancho = imagen_umbral.shape
        factor_escala = 2 if max(altura, ancho) < 1000 else 1.5
        imagen_redimensionada = cv2.resize(imagen_umbral, None, fx=factor_escala, fy=factor_escala, interpolation=cv2.INTER_CUBIC)

        # 🔹 Detectar si la inversión de colores es necesaria
        if np.mean(imagen_redimensionada) > 127:
            imagen_redimensionada = cv2.bitwise_not(imagen_redimensionada)

        # 🔹 Remover fondo de la imagen (opcional)
        imagen_redimensionada = eliminar_fondo(imagen_redimensionada)

        # 🔹 Aplicar super resolución (opcional)
        imagen_redimensionada = super_resolucion(imagen_redimensionada)

        # 🔹 Guardar la imagen procesada (opcional)
        guardar_imagen_procesada(imagen_redimensionada, num)

        # 🔹 Extraer texto con configuración optimizada de Tesseract
        custom_config = r'--oem 3 --psm 4 -l eng+spa'
        texto_extraido = pytesseract.image_to_string(imagen_redimensionada, config=custom_config)

        return texto_extraido.strip()
    
    except Exception as e:
        print(f"Error al procesar imagen {ruta_imagen}: {str(e)}")
        return ""

# ----------------------------------------------
# 🔽 FUNCIONES AUXILIARES MEJORADAS 🔽
# ----------------------------------------------

def aplicar_CLAHE(imagen):
    """Aplica CLAHE para mejorar el contraste de la imagen."""
    lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def eliminar_fondo(imagen):
    """Elimina el fondo de la imagen usando segmentación K-Means."""
    Z = imagen.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape(imagen.shape)

def eliminar_ruido_avanzado(imagen):
    """Elimina ruido preservando bordes."""
    return cv2.bilateralFilter(imagen, 9, 75, 75)

def detectar_bordes(imagen):
    """Detecta bordes en la imagen para mejorar OCR."""
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(imagen_gris, 50, 150)

def correct_skew(image, delta=1, limit=5):
    """Corrige la inclinación de la imagen basada en la alineación del texto."""
    try:
        def determine_score(arr, angle):
            data = inter.rotate(arr, angle, reshape=False, order=0)
            histogram = np.sum(data, axis=1, dtype=float)
            score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
            return histogram, score

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        scores = []
        angles = np.arange(-limit, limit + delta, delta)
        for angle in angles:
            _, score = determine_score(thresh, angle)
            scores.append(score)

        best_angle = angles[scores.index(max(scores))]

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return corrected
    
    except Exception as e:
        print(f"Error inesperado: {e}")
        return image

def remove_lines(image):
    """Elimina líneas horizontales y verticales de la imagen."""
    try:
        img_np = np.array(image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        img2 = cv2.filter2D(gray, -1, 3)
        _, img_tmp = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Eliminar líneas horizontales
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        remove_horizontal = cv2.morphologyEx(img_tmp, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

        cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts[0]:
            cv2.drawContours(img_np, [c], -1, (255, 255, 255), 5)

        # Eliminar líneas verticales
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        remove_vertical = cv2.morphologyEx(img_tmp, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

        cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts[0]:
            cv2.drawContours(img_np, [c], -1, (255, 255, 255), 5)

        return img_np
    except Exception as e:
        print(f"Error: {e}")
        return image
