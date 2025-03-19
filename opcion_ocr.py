import cv2
import numpy as np
import pytesseract
from scipy.ndimage import interpolation as inter
import os

def extraer_texto_desde_imagen(ruta_imagen, num, guardar_visualizacion=True):
    """Extrae texto de una imagen aplicando preprocesamiento avanzado para OCR."""
    try:
        # Cargar imagen
        imagen = cv2.imread(ruta_imagen)
        if imagen is None:
            print(f"Error: No se pudo cargar la imagen en {ruta_imagen}")
            return "", None

        # Guardar imagen original para comparación
        if guardar_visualizacion:
            cv2.imwrite(f"original_{num}.jpg", imagen)
        
        # 🔹 Corregir inclinación del texto
        imagen = correct_skew(imagen)
        
        # 🔹 Eliminar líneas antes del preprocesamiento
        imagen = remove_lines(imagen)
        
        # 🔹 Aplicar CLAHE para mejorar contraste
        imagen = aplicar_CLAHE(imagen)
        
        # Convertir a escala de grises
        imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        
        # Aplicar detección de bordes (solo para procesamiento, no para visualización final)
        imagen_bordes = detectar_bordes(imagen)
        
        # Guardar imagen de bordes para comparación
        if guardar_visualizacion:
            cv2.imwrite(f"bordes_{num}.jpg", imagen_bordes)
        
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
        
        # 🔹 Guardar la imagen procesada
        if guardar_visualizacion:
            guardar_imagen_procesada(imagen_redimensionada, num)
        
        # 🔹 Extraer texto con configuración optimizada de Tesseract
        custom_config = r'--oem 3 --psm 4 -l eng+spa'
        texto_extraido = pytesseract.image_to_string(imagen_redimensionada, config=custom_config)
        
        return texto_extraido.strip(), imagen_redimensionada
    
    except Exception as e:
        print(f"Error al procesar imagen {ruta_imagen}: {str(e)}")
        return "", None

# ----------------------------------------------
# 🛠 FUNCIONES AUXILIARES MEJORADAS 🛠
# ----------------------------------------------

def aplicar_CLAHE(imagen):
    """Aplica CLAHE para mejorar el contraste de la imagen."""
    try:
        lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    except Exception as e:
        print(f"Error en aplicar_CLAHE: {e}")
        return imagen

def eliminar_ruido_avanzado(imagen):
    """Elimina ruido preservando bordes."""
    try:
        # Aplicar filtrado bilateral
        filtered = cv2.bilateralFilter(imagen, 9, 75, 75)
        
        # Aplicar un filtro de mediana para eliminar ruido de sal y pimienta
        filtered = cv2.medianBlur(filtered, 3)
        
        return filtered
    except Exception as e:
        print(f"Error en eliminar_ruido_avanzado: {e}")
        return imagen

def detectar_bordes(imagen):
    """Detecta bordes en la imagen para mejorar OCR."""
    try:
        imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(imagen_gris, 50, 150)
    except Exception as e:
        print(f"Error en detectar_bordes: {e}")
        imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        return imagen_gris

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
        print(f"Error en correct_skew: {e}")
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
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]  # Compatibilidad con diferentes versiones de OpenCV
        for c in cnts:
            cv2.drawContours(img_np, [c], -1, (255, 255, 255), 5)

        # Eliminar líneas verticales
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        remove_vertical = cv2.morphologyEx(img_tmp, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

        cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]  # Compatibilidad con diferentes versiones de OpenCV
        for c in cnts:
            cv2.drawContours(img_np, [c], -1, (255, 255, 255), 5)

        return img_np
    except Exception as e:
        print(f"Error en remove_lines: {e}")
        return image

def guardar_imagen_procesada(imagen, num):
    """Guarda la imagen procesada en disco."""
    try:
        cv2.imwrite(f"procesada_{num}.jpg", imagen)
        print(f"Imagen procesada guardada como procesada_{num}.jpg")
    except Exception as e:
        print(f"Error al guardar imagen procesada: {str(e)}")

def mejorar_visualizacion(ruta_imagen, guardar_resultado=True):
    """Función específica para mejorar la visualización de imágenes con texto."""
    try:
        # Cargar imagen
        imagen = cv2.imread(ruta_imagen)
        if imagen is None:
            print(f"Error: No se pudo cargar la imagen en {ruta_imagen}")
            return None
            
        # Crear copia para comparación
        imagen_original = imagen.copy()
        
        # Aplicar mejoras específicas para visualización
        # 1. Mejorar contraste
        imagen = aplicar_CLAHE(imagen)
        
        # 2. Ajustar brillo
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        lim = 255 - 30
        v[v > lim] = 255
        v[v <= lim] += 30
        hsv = cv2.merge((h, s, v))
        imagen = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # 3. Aumentar nitidez
        kernel_sharpening = np.array([[-1,-1,-1], 
                                      [-1, 9,-1],
                                      [-1,-1,-1]])
        imagen = cv2.filter2D(imagen, -1, kernel_sharpening)
        
        # 4. Reducir ruido preservando bordes
        imagen = cv2.bilateralFilter(imagen, 9, 75, 75)
        
        # 5. Escalar imagen para mejor visualización
        altura, ancho = imagen.shape[:2]
        factor_escala = 2 if max(altura, ancho) < 1000 else 1.5
        imagen = cv2.resize(imagen, None, fx=factor_escala, fy=factor_escala, interpolation=cv2.INTER_CUBIC)
        imagen_original = cv2.resize(imagen_original, (imagen.shape[1], imagen.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        # Guardar resultado
        if guardar_resultado:
            nombre_base = os.path.basename(ruta_imagen)
            nombre_sin_ext = os.path.splitext(nombre_base)[0]
            cv2.imwrite(f"{nombre_sin_ext}_mejorada.jpg", imagen)
            print(f"Imagen mejorada guardada como {nombre_sin_ext}_mejorada.jpg")
            
            # Crear imagen comparativa
            comparacion = np.hstack((imagen_original, imagen))
            cv2.imwrite(f"{nombre_sin_ext}_comparacion.jpg", comparacion)
            print(f"Comparación guardada como {nombre_sin_ext}_comparacion.jpg")
            
        return imagen
        
    except Exception as e:
        print(f"Error al mejorar la visualización: {str(e)}")
        return None

# Función principal para visualizar
def visualizar_imagen_con_texto(ruta_imagen):
    """Función de alto nivel para visualizar mejor imágenes con texto."""
    # Mejorar imagen para visualización
    imagen_mejorada = mejorar_visualizacion(ruta_imagen)
    
    if imagen_mejorada is not None:
        # Mostrar la imagen si estamos en un entorno interactivo
        try:
            cv2.imshow("Imagen Mejorada", imagen_mejorada)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            print("No se puede mostrar la imagen en este entorno.")
    
    # También extraer texto (opcional)
    texto, _ = extraer_texto_desde_imagen(ruta_imagen, "final", guardar_visualizacion=True)
    print("Texto extraído:")
    print(texto)
    
    return imagen_mejorada

# Ejemplo de uso
if __name__ == "__main__":
    ruta_imagen = "ruta/a/tu/imagen.jpg"  # Reemplaza con la ruta de tu imagen
    visualizar_imagen_con_texto(ruta_imagen)