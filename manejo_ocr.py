import os
import cv2
import pytesseract
from PIL import Image
import numpy as np
import scipy.ndimage as inter
import re
import csv
import difflib
from datetime import datetime
from openpyxl import Workbook, load_workbook
from skimage import exposure

# Rutas base
ruta_tifs = r"D:\AD\ProyectoOCR\Nueva carpeta\CapPruebas" #"D:\AD\ProyectoOCR\Nueva carpeta\CapPruebas" #"D:\AD\ProyectoOCR\Nueva carpeta\Documentos_escaneados"
ruta_resultados = r"D:\AD\ProyectoOCR\Nueva carpeta\resultados"
path_tesseract = r"D:\AD\ProyectoCIC\tesseract"

# Palabras clave para filtrar los archivos
palabras_clave = ["Documento 1", "Doc 1", "1,"]

########################################################################################################
#                                   Crear directorios si no existen
########################################################################################################
os.makedirs(ruta_resultados, exist_ok=True)
os.makedirs(os.path.join(ruta_resultados, "imagenes"), exist_ok=True)
os.makedirs(os.path.join(ruta_resultados, "textos"), exist_ok=True)
os.makedirs(os.path.join(ruta_resultados, "csv"), exist_ok=True)

# Configurar Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = os.path.join(path_tesseract, 'tesseract.exe')

########################################################################################################
#               Funciones para el procesamiento y mejora de imágenes utilizando pytesseract
########################################################################################################

def extraer_texto_desde_imagen(ruta_imagen,num):
    #Extrae texto de una imagen aplicando preprocesamiento.
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

        # Convertir a escala de grises
        imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

        # Mejorar la imagen para OCR
        imagen_gris = cv2.medianBlur(imagen_gris, 3)
        _, imagen_umbral = cv2.threshold(imagen_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Eliminar ruido
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        imagen_umbral = cv2.dilate(imagen_umbral, kernel, iterations=1)
        imagen_umbral = cv2.erode(imagen_umbral, kernel, iterations=1)
        imagen_umbral = cv2.morphologyEx(imagen_umbral, cv2.MORPH_OPEN, kernel)

        # 🔹 Usar un factor de escala dinámico  
        altura, ancho = imagen_umbral.shape
        factor_escala = 2 if max(altura, ancho) < 1000 else 1.5

        imagen_redimensionada = cv2.resize(
            imagen_umbral, None, fx=factor_escala, fy=factor_escala, interpolation=cv2.INTER_CUBIC
        )
        
        # Convertir a color si es necesario
        if len(imagen_redimensionada.shape) == 2:
            imagen_redimensionada = cv2.cvtColor(imagen_redimensionada, cv2.COLOR_GRAY2BGR)

        # 🔹 Detectar si la inversión de colores es necesaria
        if np.mean(imagen_redimensionada) > 127:
            imagen_redimensionada = cv2.bitwise_not(imagen_redimensionada)

        imagen_redimensionada = eliminar_ruido(imagen_redimensionada)
        imagen_redimensionada = ajustar_contraste_brillo(imagen_redimensionada, metodo='gamma', gamma=1.5)

        # Guardar la imagen procesada
        guardar_imagen_procesada(imagen_redimensionada, num)

        # Extraer texto con Tesseract
        custom_config = r'--oem 3 --psm 6 -l eng+spa'
        texto_extraido = pytesseract.image_to_string(imagen_redimensionada, config=custom_config)

        return texto_extraido.strip()
    except Exception as e:
        print(f"Error al procesar imagen {ruta_imagen}: {str(e)}")
        return ""


def extraer_texto_desde_imagen_prueba(ruta_imagen, num):
    """Extrae texto de una imagen aplicando preprocesamiento."""
    try:
        imagen = cargar_imagen(ruta_imagen)
        if imagen is None:
            return ""

        # Corregir inclinación del texto
        imagen = correct_skew(imagen)

        # Eliminar líneas antes del preprocesamiento
        imagen = remove_lines(imagen)

        # Preprocesar imagen
        imagen = preprocesar_imagen(imagen)
        imagen = ajustar_contraste_brillo(imagen)
        imagen = eliminar_ruido(imagen)
        imagen_redimensionada = redimensionar_imagen(imagen)

        # Convertir a color si es necesario
        if len(imagen_redimensionada.shape) == 2:
            imagen_redimensionada = cv2.cvtColor(imagen_redimensionada, cv2.COLOR_GRAY2BGR)

        # Invertir colores si es necesario
        magen_redimensionada = invertir_colores_selectivamente(imagen_redimensionada)

        # Extraer texto con Tesseract
        texto_extraido = extraer_texto(imagen_redimensionada)

        # Guardar la imagen procesada
        guardar_imagen_procesada(imagen_redimensionada, num)

        return texto_extraido.strip()
    except Exception as e:
        print(f"Error al procesar imagen {ruta_imagen}: {str(e)}")
        return ""

def cargar_imagen(ruta_imagen):
    """Carga una imagen desde la ruta especificada."""
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        print(f"Error: No se pudo cargar la imagen en {ruta_imagen}")
    return imagen

def preprocesar_imagen(imagen):
    """Aplica preprocesamiento a la imagen."""
    imagen = correct_skew(imagen)
    imagen = remove_lines(imagen)
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imagen_gris = cv2.medianBlur(imagen_gris, 3)
    _, imagen_umbral = cv2.threshold(imagen_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    imagen_umbral = cv2.morphologyEx(imagen_umbral, cv2.MORPH_OPEN, kernel)
    return imagen_umbral

#def ajustar_contraste_brillo(imagen):
#    """Ajusta el contraste y brillo de la imagen."""
#    alpha = 1.5  # Contraste
#    beta = -40     # Brillo
#    return cv2.convertScaleAbs(imagen, alpha=alpha, beta=beta)

def ajustar_contraste_brillo(imagen, metodo='clahe', alpha=1.5, beta=-40, gamma=2.5):
    """Ajusta el contraste y brillo de la imagen según el método especificado."""
    if metodo == 'escala':
        # Ajuste de contraste y brillo usando convertScaleAbs
        return cv2.convertScaleAbs(imagen, alpha=alpha, beta=beta)
    elif metodo == 'gamma':
        # Ajuste de gamma
        return exposure.adjust_gamma(imagen, gamma=gamma, gain=1)
    elif metodo == 'clahe':
        # Ajuste de contraste adaptativo con ecualización de histograma (CLAHE)
        imagen_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(imagen_gray)
    else:
        raise ValueError("Método no reconocido. Usa 'escala', 'gamma' o 'clahe'.")

def eliminar_ruido(imagen):
    """Elimina el ruido de la imagen."""
    return cv2.GaussianBlur(imagen, (5, 5), 0)
    #return cv2.bilateralFilter(imagen, 9, 75, 75)
    #return cv2.medianBlur(imagen, 5)
    #return cv2.fastNlMeansDenoising(imagen, None, 30, 7, 21)

def redimensionar_imagen(imagen):
    """Redimensiona la imagen según un factor de escala dinámico."""
    altura, ancho = imagen.shape
    factor_escala = 2 if max(altura, ancho) < 1000 else 1.5
    return cv2.resize(imagen, None, fx=factor_escala, fy=factor_escala, interpolation=cv2.INTER_CUBIC)

def invertir_colores_si_necesario(imagen):
    """Invierte los colores de la imagen si es necesario."""
    if np.mean(imagen) > 127:
        imagen = cv2.bitwise_not(imagen)
    return imagen

def invertir_colores_selectivamente(imagen):
    """Invierte los colores de las áreas de la imagen que tienen texto blanco sobre fondo negro."""
    # Convertir la imagen a escala de grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Aplicar umbral para separar texto blanco sobre fondo negro
    _, umbral_invertido = cv2.threshold(imagen_gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Encontrar contornos de las áreas invertidas
    contornos, _ = cv2.findContours(umbral_invertido, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crear una máscara para las áreas que necesitan inversión
    mascara_inversion = np.zeros_like(imagen_gris)

    for contorno in contornos:
        # Rellenar el contorno en la máscara
        cv2.drawContours(mascara_inversion, [contorno], -1, 255, -1)

    # Invertir los colores en las áreas seleccionadas
    imagen_invertida = cv2.bitwise_not(imagen, mask=mascara_inversion)

    # Combinar la imagen original y la imagen invertida usando la máscara
    imagen_final = cv2.bitwise_and(imagen, imagen, mask=cv2.bitwise_not(mascara_inversion))
    imagen_final += imagen_invertida

    return imagen_final

def extraer_texto(imagen):
    """Extrae texto de la imagen utilizando Tesseract OCR."""
    custom_config = r'--oem 3 --psm 6 -l eng+spa'
    return pytesseract.image_to_string(imagen, config=custom_config)

def guardar_imagen_procesada(imagen, num):
    """Guarda la imagen procesada en la ruta especificada."""
    ruta_guardado = f"D:\\AD\\ProyectoOCR\\Nueva carpeta\\CapPruebas\\imagen_procesada_{num}.png"
    cv2.imwrite(ruta_guardado, imagen)

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
    
    except pytesseract.TesseractError as e:
        print(f"Error de Tesseract: {e}")
    except FileNotFoundError as e:
        print(f"Error: Archivo no encontrado - {e}")
    except Exception as e:
        print(f"Error inesperado: {e}")

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
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]  # Asegurar compatibilidad con OpenCV

        for c in cnts:
            cv2.drawContours(img_np, [c], -1, (255, 255, 255), 5)

        # Eliminar líneas verticales
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        remove_vertical = cv2.morphologyEx(img_tmp, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

        cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]  # Asegurar compatibilidad con OpenCV

        for c in cnts:
            cv2.drawContours(img_np, [c], -1, (255, 255, 255), 5)

        # Convertir la matriz NumPy resultante a una imagen de Pillow
        #img_oper_ths = Image.fromarray(img_np)

        return img_np
    except Exception as e:
        print(f"Error: {e}")
        return None

########################################################################################################
#               Funciones para la extracción de datos mediante OCR con pytesseract
########################################################################################################

def extract_data_campos(text):
    try:

        data = {}
        text = re.sub(r'[^a-zA-Z0-9\s:;/().]+', '', text)
        # Función auxiliar para encontrar la coincidencia más cercana para un patrón dado
        def find_closest_match(pattern, text):
            words = extraer_valores(text)
            # Encontrar la coincidencia más cercana y el porcentaje de similitud
            matches = difflib.get_close_matches(pattern, words, n=1, cutoff=0.0)
            if matches:
                closest_match = matches
                porcentaje_similitud = difflib.SequenceMatcher(None, pattern, closest_match).ratio() * 100
                #print(f"La coincidencia más cercana para '{pattern}' es: '{closest_match}' con un {porcentaje_similitud:.2f}% de similitud")
            else:
                print(f"No se encontró ninguna coincidencia cercana para '{pattern}'")
            
            return ''.join(matches) if matches else None
        
        # Apellido Paterno
        apellido_paterno_pattern = 'Apellido Paterno'
        apellido_paterno_match = find_closest_match(apellido_paterno_pattern, text)
        #apellido_paterno_match = re.escape(apellido_paterno_match)
        apellido_paterno_match = ''.join(c for c in apellido_paterno_match if c.isalnum() or c.isspace()).strip()
        print(f"========================={apellido_paterno_match}===============================")
        if apellido_paterno_match:
            apellido_paterno = re.search(rf'{apellido_paterno_match}[:;]\s* [^\w\s]* ?([A-ZÑÁÉÍÓÚ]+(?: DEL| DE)?(?: [A-ZÑÁÉÍÓÚ]+)?)\s', text)
        data['Apellido Paterno'] = re.sub(r'[^\w\s]+$', '', apellido_paterno.group(1)).strip() if apellido_paterno else None
        
        # Apellido Materno
        apellido_materno_pattern = 'Apellido Materno'
        apellido_materno_match = find_closest_match(apellido_materno_pattern, text)
        #apellido_materno_match = re.escape(apellido_materno_match)
        apellido_materno_match = ''.join(c for c in apellido_materno_match if c.isalnum() or c.isspace()).strip()
        #print(f"========================={apellido_materno_match}===============================")
        if apellido_materno_match:
            apellido_materno = re.search(rf'{apellido_materno_match}[:;]\s* [^\w\s]* ?([A-ZÑÁÉÍÓÚ]+(?: DEL| DE)?(?: [A-ZÑÁÉÍÓÚ]+)?)\s', text)
            data['Apellido Materno'] = re.sub(r'[^\w\s]+$', '', apellido_materno.group(1)).strip() if apellido_materno else None
        
        # Nombres
        nombres_pattern = 'Nombres'
        nombres_match = find_closest_match(nombres_pattern, text)
        #nombres_match = re.escape(nombres_match)
        nombres_match = ''.join(c for c in nombres_match if c.isalnum() or c.isspace()).strip()
        #print(f"========================={nombres_match}===============================")
        if nombres_match:
            nombres = re.search(rf'{nombres_match}[:;]\s* [^\w\s]* ?([A-ZÑÁÉÍÓÚ. ]+)\s', text)
        data['Nombres'] = re.sub(r'[^\w\s]+$', '', nombres.group(1)).strip() if nombres else None
        
        # Nit Empresa
        nit_empresa_pattern = 'Nit Empresa'
        nit_empresa_match = find_closest_match(nit_empresa_pattern, text)
        #nit_empresa_match = re.escape(nit_empresa_match)
        nit_empresa_match = ''.join(c for c in nit_empresa_match if c.isalnum() or c.isspace()).strip()
        #print(f"========================={nit_empresa_match}===============================")
        if nit_empresa_match:
            nit_empresa = re.search(rf'{nit_empresa_match}[:;]\s* ([A-Za-z0-9]+)', text)
            data['Nit Empresa'] = nit_empresa.group(1) if nit_empresa else None
        
        # Nombre Empresa
        nombre_empresa_pattern = 'Nombre Empresa'
        nombre_empresa_match = find_closest_match(nombre_empresa_pattern, text)
        #nombre_empresa_match = re.escape(nombre_empresa_match)
        nombre_empresa_match = ''.join(c for c in nombre_empresa_match if c.isalnum() or c.isspace()).strip()
        #print(f"========================={nombre_empresa_match}===============================")
        if nombre_empresa_match:
            nombre_empresa = re.search(rf'{nombre_empresa_match}[:;]\s* [^\w\s]* ?([A-Za-zÑñÁÉÍÓÚáéíóú. ]+)\s', text)
            data['Nombre Empresa'] = nombre_empresa.group(1).strip() if nombre_empresa else None
        
        # Tipo Segmento
        tipo_segmento_pattern = 'Tipo Segmento'
        tipo_segmento_match = find_closest_match(tipo_segmento_pattern, text)
        #tipo_segmento_match = re.escape(tipo_segmento_match)
        tipo_segmento_match = ''.join(c for c in tipo_segmento_match if c.isalnum() or c.isspace()).strip()
        #print(f"========================={tipo_segmento_match}===============================")
        if tipo_segmento_match:
            tipo_segmento = re.search(rf'{tipo_segmento_match}[:;]\s* ?[^\w\s]* ?([A-Za-zÑñÁÉÍÓÚáéíóú. ]+)\s', text)
            data['Tipo Segmento'] = tipo_segmento.group(1).strip() if tipo_segmento else None
        
        # Segmento
        segmento_pattern = 'Segmento'
        segmento_match = find_closest_match(segmento_pattern, text)
        #segmento_match = re.escape(segmento_match)
        segmento_match = ''.join(c for c in segmento_match if c.isalnum() or c.isspace()).strip()
        #print(f"========================={segmento_match}===============================")
        if segmento_match:
            segmento = re.search(rf'{segmento_match}[:;]\s* [^\w\s]* ?([A-Za-zÑñÁÉÍÓÚáéíóú.]+)\s', text)
            data['Segmento'] = segmento.group(1).strip() if segmento else None
        
        # Ingreso Bs
        ingreso_bs_pattern = 'Ingreso Bs'
        ingreso_bs_match = find_closest_match(ingreso_bs_pattern, text)
        #ingreso_bs_match = re.escape(ingreso_bs_match)
        ingreso_bs_match = ''.join(c for c in ingreso_bs_match if c.isalnum() or c.isspace()).strip()
        #print(f"========================={ingreso_bs_match}===============================")
        if ingreso_bs_match:
            ingreso_bs = re.search(rf'{ingreso_bs_match}[:;]\s* [^\w\s]* ?([\d.]+)', text)
            data['Ingreso Bs'] = float(re.sub(r'[^\w\s]+$', '', ingreso_bs.group(1))) if ingreso_bs else None
        
        # Tipo Ingreso
        tipo_ingreso_pattern = 'Tipo Ingreso'
        tipo_ingreso_match = find_closest_match(tipo_ingreso_pattern, text)
        #Eliminamos carracteres especial al inicio y al final y los espacios
        #tipo_ingreso_match = re.escape(tipo_ingreso_match)
        tipo_ingreso_match = ''.join(c for c in tipo_ingreso_match if not c.isdigit() and (c.isalnum() or c.isspace())).strip()
        #print(f"========================={tipo_ingreso_match}===============================")
        if tipo_ingreso_match:
            tipo_ingreso = re.search(rf'{tipo_ingreso_match}[:;]\s*[^\w\s]* ?([A-ZÑÁÉÍÓÚ0-9. ]+)', text)
            data['Tipo Ingreso'] = tipo_ingreso.group(1).strip() if tipo_ingreso else None
        
        # Ingreso Bruto Bs
        ingreso_bruto_bs_pattern = 'Ingreso Bruto Bs'
        ingreso_bruto_bs_match = find_closest_match(ingreso_bruto_bs_pattern, text)
        #ingreso_bruto_bs_match = re.escape(ingreso_bruto_bs_match)
        ingreso_bruto_bs_match = ''.join(c for c in ingreso_bruto_bs_match if c.isalnum() or c.isspace()).strip()
        #print(f"========================={ingreso_bruto_bs_match}===============================")
        if ingreso_bruto_bs_match:
            ingreso_bruto_bs = re.search(rf'{ingreso_bruto_bs_match}[:;]\s* ([\d.]+)', text)
            data['Ingreso Bruto Bs'] = float(re.sub(r'[^\w\s]+$', '', ingreso_bruto_bs.group(1))) if ingreso_bruto_bs else None
        
        # Justificación Ingreso
        justificacion_ingreso_pattern = 'Justificación Ingreso'
        justificacion_ingreso_match = find_closest_match(justificacion_ingreso_pattern, text)
        #justificacion_ingreso_match = re.escape(justificacion_ingreso_match)
        justificacion_ingreso_match = ''.join(c for c in justificacion_ingreso_match if c.isalnum() or c.isspace()).strip()
        #print(f"========================={justificacion_ingreso_match}===============================")
        if justificacion_ingreso_match:
            justificacion_ingreso = re.search(rf'{justificacion_ingreso_match}[:;]\s*(.*)\s', text)
            data['Justificación Ingreso'] = re.sub(r'[^\w\s]+$', '', justificacion_ingreso.group(1)).strip() if justificacion_ingreso else None

        # Calificación BCP
        calificacion_bcp_pattern = 'Calificacion BCP'
        calificacion_bcp_match = find_closest_match(calificacion_bcp_pattern, text)
        calificacion_bcp_match = re.escape(calificacion_bcp_match)
        #calificacion_bcp_match = ''.join(c for c in calificacion_bcp_match if c.isalnum() or c.isspace()).strip()
        #print(f"========================={calificacion_bcp_match}===============================")
        if calificacion_bcp_match:
            calificacion_bcp = re.search(rf'{calificacion_bcp_match}[:;]\s* [^\w\s]* ?([A-ZÑÁÉÍÓÚ. ]+)\s', text)
            data['Calificacion BCP'] = re.sub(r'[^\w\s]+$', '', calificacion_bcp.group(1)).strip() if calificacion_bcp else None
        
        # Calificación SF
        calificacion_sf_pattern = 'Calificacion SF'
        calificacion_sf_match = find_closest_match(calificacion_sf_pattern, text)
        calificacion_sf_match = re.escape(calificacion_sf_match)
        #calificacion_sf_match = ''.join(c for c in calificacion_sf_match if c.isalnum() or c.isspace()).strip()
        #print(f"========================={calificacion_sf_match}===============================")
        if calificacion_sf_match:
           calificacion_sf = re.search(rf'{re.escape(calificacion_sf_match)}[:;]\s*([A-ZÑÁÉÍÓÚa-zñáéíóú. )]+)', text)
           data['Calificacion SF'] = re.sub(r'[^\w\s]+$', '', calificacion_sf.group(1)).strip().upper() if calificacion_sf else None
        
        # Archivo Negativo
        archivo_negativo_pattern = 'Archivo Negativo'
        archivo_negativo_match = find_closest_match(archivo_negativo_pattern, text)
        #archivo_negativo_match = re.escape(archivo_negativo_match)
        archivo_negativo_match = ''.join(c for c in archivo_negativo_match if c.isalnum() or c.isspace()).strip()
        #print(f"========================={archivo_negativo_match}===============================")
        if archivo_negativo_match:
            #archivo_negativo = re.search(rf'{archivo_negativo_match}[:;]\s* ?[^\w\s]* ?([A-ZÑÁÉÍÓÚ. ]+)\s', text)
            archivo_negativo = re.search(rf'{re.escape(archivo_negativo_match)}[:;]?\s  *([A-Za-zÑñÁÉÍÓÚáéíóú.]+)', text)
            data['Archivo Negativo'] = re.sub(r'[^\w\s]+$', '', archivo_negativo.group(1)).strip().upper() if archivo_negativo else None
        
        # Ename Chequer
        ename_chequer_pattern = 'Ename Chequer'
        ename_chequer_match = find_closest_match(ename_chequer_pattern, text)
        ename_chequer_match = re.escape(ename_chequer_match)
        #print(f"========================={ename_chequer_match}===============================")
        if ename_chequer_match:
            ename_chequer = re.search(rf'{ename_chequer_match}[:;]\s* [^\w\s]* ?([A-ZÑÁÉÍÓÚ ]+)\s', text)
            data['Ename Chequer'] = re.sub(r'[^\w\s]+$', '', ename_chequer.group(1)).strip() if ename_chequer else None
        
        # Cliente CPOP
        cliente_cpop_pattern = 'Cliente CPOP'
        cliente_cpop_match = find_closest_match(cliente_cpop_pattern, text)
        print(cliente_cpop_match)
        cliente_cpop_match = re.escape(cliente_cpop_match)
        #cliente_cpop_match = ''.join(c for c in cliente_cpop_match if c.isalnum() or c.isspace()).strip()
        print(f"========================={cliente_cpop_match}===============================")
        if cliente_cpop_match:
            cliente_cpop = re.search(rf'{cliente_cpop_match}[:;]\s* [^\w\s]* ?([A-Za-zÑñÁÉÍÓÚáéíóú.)()]+)\s', text)
            data['Cliente CPOP'] = re.sub(r'[^\w\s]+$', '', cliente_cpop.group(1)).strip().upper() if cliente_cpop else None
        
        # Fecha de Consulta
        fecha_consulta_pattern = 'Fecha de Consulta'
        fecha_consulta_match = find_closest_match(fecha_consulta_pattern, text)
        fecha_consulta_match = re.escape(fecha_consulta_match)
        #fecha_consulta_match = ''.join(c for c in fecha_consulta_match if c.isalnum() or c.isspace()).strip()
        #print(f"========================={fecha_consulta_match}===============================")
        if fecha_consulta_match:
            fecha_consulta = re.search(rf"{fecha_consulta_match}[:;]\s*(\d{{2}}/\d{{2}}/\d{{4}})", text)
            data['Fecha de Consulta'] = fecha_consulta.group(1) if fecha_consulta else None

        return data        

    except AttributeError as e:
        print(f"Error: {e}")

"""
def extract_data_campos(text):
    #Extrae datos específicos de un texto utilizando patrones predefinidos.
    try:
        data = {}
        text = re.sub(r'[^a-zA-Z0-9\s:;/().]+', '', text)

        def find_closest_match(pattern, text):
            #Encuentra la coincidencia más cercana para un patrón dado en el texto.
            words = extraer_valores(text)
            matches = difflib.get_close_matches(pattern, words, n=1, cutoff=0.0)
            if matches:
                closest_match = matches[0]
                porcentaje_similitud = difflib.SequenceMatcher(None, pattern, closest_match).ratio() * 100
                print(f"La coincidencia más cercana para '{pattern}' es: '{closest_match}' con un {porcentaje_similitud:.2f}% de similitud")
                return closest_match
            else:
                print(f"No se encontró ninguna coincidencia cercana para '{pattern}'")
                return None

        def extract_field(pattern, text, regex):
            #Extrae un campo específico del texto utilizando un patrón y una expresión regular.
            match = find_closest_match(pattern, text)
            if match:
                match = ''.join(c for c in match if c.isalnum() or c.isspace()).strip()
                result = re.search(regex.format(match), text)
                return re.sub(r'[^\w\s]+$', '', result.group(1)).strip() if result else None
            return None

        # Definir los patrones y expresiones regulares para cada campo
        fields = {
            'Apellido Paterno': r'{}[:;]\s* [^\w\s]* ?([A-ZÑÁÉÍÓÚ]+(?: DEL| DE)?(?: [A-ZÑÁÉÍÓÚ]+)?)\s',
            'Apellido Materno': r'{}[:;]\s* [^\w\s]* ?([A-ZÑÁÉÍÓÚ]+(?: DEL| DE)?(?: [A-ZÑÁÉÍÓÚ]+)?)\s',
            'Nombres': r'{}[:;]\s* [^\w\s]* ?([A-ZÑÁÉÍÓÚ. ]+)\s',
            'Nit Empresa': r'{}[:;]\s* (\d+)',
            'Nombre Empresa': r'{}[:;]\s* [^\w\s]* ?([A-Za-zÑñÁÉÍÓÚáéíóú. ]+)\s',
            'Tipo Segmento': r'{}[:;]\s* ?[^\w\s]* ?([A-Za-zÑñÁÉÍÓÚáéíóú. ]+)\s',
            'Segmento': r'{}[:;]\s* [^\w\s]* ?([A-Za-zÑñÁÉÍÓÚáéíóú.]+)\s',
            'Ingreso Bs': r'{}[:;]\s* [^\w\s]* ?([\d.]+)',
            'Tipo Ingreso': r'{}[:;]\s*[^\w\s]* ?([A-ZÑÁÉÍÓÚ0-9. ]+)',
            'Ingreso Bruto Bs': r'{}[:;]\s* ([\d.]+)',
            'Justificación Ingreso': r'{}[:;]\s*(.*)\s',
            'Calificacion BCP': r'{}[:;]\s* [^\w\s]* ?([A-ZÑÁÉÍÓÚ ]+)\s',
            'Calificacion SF': r'{}[:;]\s* [^\w\s]* ?([A-ZÑÁÉÍÓÚ ]+)\s',
            'Archivo Negativo': r'{}[:;]\s* ?[^\w\s]* ?([A-ZÑÁÉÍÓÚ ]+)\s',
            'Ename Chequer': r'{}[:;]\s* [^\w\s]* ?([A-ZÑÁÉÍÓÚ ]+)\s',
            'Cliente CPOP': r'{}[:;]\s* [^\w\s]* ?([A-Za-zÑñÁÉÍÓÚáéíóú.]+)\s',
            'Fecha de Consulta': r'{}[:;]\s*(\d{{2}}/\d{{2}}/\d{{4}})'
        }

        # Extraer cada campo utilizando los patrones y expresiones regulares definidos
        for field, regex in fields.items():
            data[field] = extract_field(field, text, regex)

        return data

    except AttributeError as e:
        print(f"Error: {e}")
"""

def extraer_valores(text):
    """Extrae claves del texto que terminan en ':' o ';'."""
    pattern = re.findall(r'([^\n:;]+)[:;]', text)
    cleaned_pattern = [item.strip() for item in pattern]
    return cleaned_pattern

def procesar_pagina(nombre_carpeta, nombre_archivo, i, img, texto_extraido):
    """
    Procesa una página de un archivo TIFF y guarda los resultados.
    """
    try:
        # Generar nombres de archivo
        nombre_base = f"{nombre_carpeta}_pagina_{i+1}"
        
        # Guardar imagen
        ruta_imagen_guardada = os.path.join(ruta_resultados, "imagenes", f"{nombre_base}.jpeg")
        img.save(ruta_imagen_guardada, "JPEG")
        
        # Guardar texto (comentado)
        ruta_texto = os.path.join(ruta_resultados, "textos", f"{nombre_base}.txt")
        with open(ruta_texto, "w", encoding="utf-8") as f:
            f.write(texto_extraido)
        
        # Extraer campos específicos
        datos_extraidos = extract_data_campos(texto_extraido)
        print(f"DATOS DE LA IMAGEN : {datos_extraidos}")
        
        # Añadir información del archivo
        datos_extraidos['Archivo_Fuente'] = nombre_archivo
        datos_extraidos['Carpeta_Fuente'] = nombre_carpeta
        
        # Guardar datos en CSV
        #ruta_csv_guardado = guardar_datos_csv(datos_extraidos, 'Perfil Cliente')

        print(f"✅ Página de Perfil de Cliente extraída: {ruta_imagen_guardada}")
        #print(f"✅ Datos extraídos y guardados en: {ruta_csv_guardado}")

    except Exception as e:
        print(f"Error al procesar la página: {e}")

def guardar_datos_csv(datos, hoja_nombre):
    """
    Guarda los datos extraídos en un archivo Excel en una hoja específica.
    El archivo se guarda con el nombre del día actual.
    Maneja errores con try-except.
    """
    try:
        # Obtener la fecha actual
        fecha_actual = datetime.now().strftime("%Y-%m-%d")
        # Definir la ruta del archivo con el nombre del día actual
        ruta_salida = os.path.join(ruta_resultados, "csv", f"ResultadoOCR_{fecha_actual}.xlsx")
        
        # Verificar si el archivo existe
        archivo_existe = os.path.exists(ruta_salida)
        
        if archivo_existe:
            # Cargar el libro existente
            libro = load_workbook(ruta_salida)
            if hoja_nombre in libro.sheetnames:
                hoja = libro[hoja_nombre]
            else:
                hoja = libro.create_sheet(title=hoja_nombre)
        else:
            # Crear un nuevo libro y hoja
            libro = Workbook()
            hoja = libro.active
            hoja.title = hoja_nombre
        
        # Escribir encabezados si el archivo es nuevo o la hoja es nueva
        if not archivo_existe or hoja.max_row == 1:
            hoja.append(list(datos.keys()))
        
        # Escribir datos
        hoja.append(list(datos.values()))
        
        # Guardar el libro
        libro.save(ruta_salida)
        
        return ruta_salida

    except Exception as e:
        print(f"Error al guardar los datos en el archivo Excel: {e}")
        return None

########################################################################################################
#               Funcion Principal de Ejecucion
########################################################################################################

def procesar_archivos_tiff(ruta_base, pagina_inicio=19):
    """Procesa recursivamente los archivos TIFF para extraer páginas de Perfil de Cliente."""
    archivos_procesados = 0
    archivos_fallidos = 0

    # Ruta para el archivo CSV de resultados
    ruta_csv = os.path.join(ruta_resultados, "perfiles_cliente.csv")

    # Recorrer recursivamente todas las subcarpetas
    for root, dirs, files in os.walk(ruta_base):
        # Filtrar archivos TIFF que coinciden con palabras clave
        tif_files = [f for f in files if f.lower().endswith(('.tif', '.tiff')) 
                     and any(palabra in f for palabra in palabras_clave)]
        
        if not tif_files:
            continue  # Saltar carpetas sin archivos coincidentes

        print(f"\n📁 Procesando carpeta: {root}")
        print(f"Archivos encontrados: {tif_files}")
        
        for nombre_archivo in tif_files:
            try:
                tif_path = os.path.join(root, nombre_archivo)
                nombre_carpeta = os.path.basename(root)  # Usar el nombre de la carpeta padre
                
                with Image.open(tif_path) as img:
                    pagina_procesada = False
                    valor = 0
                    for i in range(pagina_inicio, img.n_frames):
                        img.seek(i)
                        
                        # Convertir la página a imagen temporal
                        page_image_path = os.path.join(ruta_resultados, "temp_page.jpg")
                        img.save(page_image_path, "JPEG")
                        
                        # Extraer texto
                        texto_extraido = extraer_texto_desde_imagen(page_image_path, valor)
                        valor += 1
                        #print(f"Verificar si la página contiene PERFIL DEL CLIENTE : {texto_extraido}")
                        # Verificar si la página contiene "PERFIL DEL CLIENTE" o ciertos campos específicos
                        if (("PERFIL DEL CLIENTE" in texto_extraido) or 
                            ("Fecha de Consulta" in texto_extraido and "Segmento" in texto_extraido)):
                            #print(f"Verificar si la página contiene PERFIL DEL CLIENTE : {texto_extraido}")
                            procesar_pagina(nombre_carpeta, nombre_archivo, i, img, texto_extraido)

                            pagina_procesada = True
                            archivos_procesados += 1
                            break  # Salir del bucle de páginas
                        
                    if not pagina_procesada:
                        print(f"❌ No se encontró página de Perfil de Cliente en {nombre_archivo}")
                        archivos_fallidos += 1

            except Exception as e:
                print(f"Error procesando {nombre_archivo}: {e}")
                archivos_fallidos += 1

    # Resumen final
    print("\n--- RESUMEN ---")
    print(f"Archivos procesados exitosamente: {archivos_procesados}")
    print(f"Archivos con errores: {archivos_fallidos}")
    print(f"Datos extraídos guardados en: {ruta_csv}")

# Ejecutar el proceso
procesar_archivos_tiff(ruta_tifs, pagina_inicio=47)
