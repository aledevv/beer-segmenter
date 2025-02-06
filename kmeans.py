import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

# Configura la cartella delle immagini
image_folder = "frames"  # Cambia con il percorso della tua cartella
points = 'points_to_crop.txt'
clusters = 2  # Numero di cluster per K-means

# Ordina i file in ordine alfabetico e permette di scegliere un frame iniziale
image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
image_index = 290  # Modifica questo valore per partire da un frame specifico

def preprocess_and_segment(img, n_clusters=3):
    # Leggi l'immagine
    if img is None:
        raise ValueError("Impossibile leggere l'immagine")
    
    # Converti in scala di grigi
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Aumenta il contrasto con CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(gray)
    
    # 2. Applica filtro bilaterale per ridurre il rumore preservando i bordi
    bilateral = cv2.bilateralFilter(img_clahe, 9, 75, 75)
    
    # 3. Calcola LBP per le texture
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    
    # 4. Estrai canale V dallo spazio HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:,:,2]
    
    # Normalizza tutte le features tra 0 e 1
    bilateral_norm = bilateral / 255.0
    lbp_norm = (lbp - lbp.min()) / (lbp.max() - lbp.min())
    v_norm = v_channel / 255.0
    
    # Combina le features
    h, w = gray.shape
    features = np.column_stack([
        bilateral_norm.reshape(-1),
        lbp_norm.reshape(-1),
        v_norm.reshape(-1)
    ])
    
    # Applica k-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    
    # Riorganizza i labels in forma di immagine
    segmented = labels.reshape(h, w)
    
    # Identifica il cluster della schiuma (assumiamo sia il cluster più chiaro)
    cluster_means = []
    for i in range(n_clusters):
        cluster_mean = np.mean(bilateral_norm.reshape(-1)[labels == i])
        cluster_means.append((i, cluster_mean))
    
    # Ordina i cluster per luminosità media
    foam_cluster = max(cluster_means, key=lambda x: x[1])[0]
    
    # Crea maschera binaria per la schiuma
    foam_mask = (segmented == foam_cluster).astype(np.uint8) * 255
    
    return img, segmented, foam_mask


def isolate_foam(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, _, _ = cv2.split(lab)
    _, foam_mask = cv2.threshold(l, 80, 255, cv2.THRESH_BINARY)
    foam_highlighted = cv2.bitwise_and(image, image, mask=foam_mask)
    return foam_highlighted

# Funzione per segmentare l'immagine con K-means
def segment_image(image, k):
    pixels = image.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.8)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 18, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()].reshape(image.shape)
    return segmented, labels.reshape(image.shape[:2]), centers


def find_most_bordering_contour(segmented, color, given_contour):
    mask = cv2.inRange(segmented, color, color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:  # Se non ci sono contorni, restituisci una lista vuota
        return []
    
    # Inizializza variabili per il massimo e l'indice del contorno corrispondente
    max_common = 0
    best_contour_idx = -1  # Indice del contorno con il massimo numero di punti comuni

    # Itera attraverso i contorni e confronta con given_contour
    for i, contour in enumerate(contours):
        # Trova gli elementi comuni tra given_contour e il contorno corrente
        common_elements = np.intersect1d(given_contour, contour)
        common_count = len(common_elements)  # Numero di punti comuni
        
        # Se il numero di punti comuni è maggiore del massimo trovato finora, aggiorna
        if common_count > max_common:
            max_common = common_count
            best_contour_idx = i  # Salva l'indice del contorno che ha il massimo numero di punti comuni
        
    # img_copy = segmented.copy()  # Crea una copia per non modificare l'originale
    # cv2.drawContours(img_copy, [contours[best_contour_idx]], -1, (0, 255, 0), 2)  # Disegna il contorno in verde
    # cv2.imshow("gray", img_copy)
    # cv2.waitKey(0)
    
    # # Mostra l'immagine con il contorno corrente
    # cv2.imshow(f"Contorno {i+1}/{len(contours)}", img_copy)
    # key = cv2.waitKey(0)  # Aspetta che l'utente prema un tasto
    
    # for i, contour in enumerate(contours):
    #     img_copy = segmented.copy()  # Crea una copia per non modificare l'originale
    #     cv2.drawContours(img_copy, [contour], -1, (0, 255, 0), 2)  # Disegna il contorno in verde
        
    #     # Mostra l'immagine con il contorno corrente
    #     cv2.imshow(f"Contorno {i+1}/{len(contours)}", img_copy)
    #     key = cv2.waitKey(0)  # Aspetta che l'utente prema un tasto
    
    return contours[best_contour_idx]  # Tieni solo il contorno più grande

def find_largest_cluster_contours(segmented, color):
    """Trova il contorno più grande per il cluster con la label specificata."""
    mask = cv2.inRange(segmented, color, color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:  # Se non ci sono contorni, restituisci una lista vuota
        return []

    # Ordina i contorni per area e prendi il più grande
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    return contours[:1]  # Tieni solo il contorno più grande


# Calcola l'area del contorno più grande
def calculate_contour_area(contour, scale_x, scale_y):
    if contour is not None and len(contour) > 0:
        scaled_contour = np.array([[(p[0] * scale_x, p[1] * scale_y)] for p in contour], dtype=np.float32)
        area = cv2.contourArea(scaled_contour)  # Area in pixel
        return area
    return 0

def remove_region(image, points):
    with open(points, 'r') as file:
        points = [tuple(map(int, line.strip().split(','))) for line in file]
    height, width = image.shape[:2]
    points.append((0, height))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
    inverted_mask = cv2.bitwise_not(mask)
    return cv2.bitwise_and(image, image, mask=inverted_mask)

def merge_contours_outer(image_shape, contour1, contour2):
    """Unisce due contorni e trova il contorno esterno della loro combinazione."""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)  # Usa solo la parte 2D dell'immagine

    # Disegna i due contorni riempiendoli
    cv2.drawContours(mask, [contour1], -1, 255, thickness=cv2.FILLED)
    cv2.drawContours(mask, [contour2], -1, 255, thickness=cv2.FILLED)

    # Trova il contorno esterno
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours[0] if contours else None  # Restituisce il contorno esterno più grande

def show_image():
    global image_index, image_files
    while True:
        image_path = os.path.join(image_folder, image_files[image_index])
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_height, original_width = original_image.shape[:2]

        #original_image = remove_region(original_image, points)
        
        # fixed_image = remove_region(original_image, points)
        
        # foam_image = isolate_foam(blurred_image)
        # hsv = cv2.cvtColor(foam_image, cv2.COLOR_RGB2HSV)
        
#!!!!!!!!!!!!!!!!
        resized_dim = (400, 400)
        resized_image = cv2.resize(original_image, resized_dim)
        segmented_image, labels, centers = segment_image(resized_image, clusters)
        
        centers = sorted(centers, key=lambda x: x[0], reverse=True)
        
        
        contours_white = find_largest_cluster_contours(segmented_image, centers[0])
        
        if (clusters>2):
            contours_grey = find_most_bordering_contour(segmented_image, centers[1], contours_white)
        
            # fondi i due contorni
            contours = merge_contours_outer(segmented_image.shape, contours_white[0], contours_grey)
        else:
            contours = contours_white[0]

        scale_x = original_width / resized_dim[0]
        scale_y = original_height / resized_dim[1]
        
        for contour in contours:
            resized_contour = np.array([[(int(p[0] * scale_x), int(p[1] * scale_y))] for p in contour])  # Fix per il punto p
            cv2.drawContours(segmented_image, [contour], -1, (0, 255, 0), 2)
            cv2.drawContours(original_image, [resized_contour], -1, (0, 255, 0), 2)  # Disegna anche sull'originale
            
            # Calcola e stampa l'area
            area = calculate_contour_area(contour, scale_x, scale_y)
            print(f"Frame {image_index}: Area del contorno = {area:.2f} pixel^2")

        # (Opzionale) Mostra l'area sull'immagine
        text_position = (10, 30)  # Posizione del testo
        cv2.putText(original_image, f"Area: {area:.2f}", text_position, 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        resized_original = cv2.resize(original_image, resized_dim)
        combined_image = np.vstack((resized_original, segmented_image))
        cv2.imshow("Image Segmentation", cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(0) & 0xFF
        if key == ord('n'):
            image_index = (image_index + 1) % len(image_files)
        elif key == ord('p'):
            image_index = (image_index - 1) % len(image_files)
        elif key == 27:
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if image_files:
        show_image()
    else:
        print("Nessuna immagine trovata nella cartella specificata.")


# AREA DOVE FARE IL CAMBIO DI METODO
# ~158326.08 pixel^2
# Ma occhio che l'altro metodo considera anche l'area della pipe. Per capirci tipo dal frame 290