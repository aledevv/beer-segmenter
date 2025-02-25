import cv2
import numpy as np
import os
from kmeans import segment_image, isolate_foam, remove_region, calculate_contour_area, find_largest_cluster_contours, find_most_bordering_contour, merge_contours_outer

# def apply_circular_mask(image, center, radius):
#     # Crea una maschera nera con le stesse dimensioni dell'immagine, ma con un solo canale (grayscale)
#     mask = np.zeros(image.shape[:2], dtype=np.uint8)  # La maschera deve essere in scala di grigi (2D)
    
#     # Disegna un cerchio bianco sulla maschera
#     cv2.circle(mask, center, radius, 255, -1)  # 255 per il cerchio bianco
    
#     # Applica la maschera sull'immagine
#     return cv2.bitwise_and(image, image, mask=mask)


# def compute_parameters(frame_idx):
#     # Funzione per calcolare il centro e il raggio della maschera circolare
#     radius = -0.0000 * (frame_idx ** 2) + 0.5295 * frame_idx + 145.4149
#     center_x = 0.0004 * (frame_idx ** 2) + 0.0198 * frame_idx + 300.1078
#     center_y = -0.0002 * (frame_idx ** 2) + 0.8531 * frame_idx + 31.0855
#     return (int(center_x), int(center_y)), int(radius)

# def isolate_foam(image):
#     # Converti l'immagine in spazio colore LAB per isolare la luminosità
#     lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
#     l, a, b = cv2.split(lab)

#     # Applica una soglia adattiva sul canale L (luminosità)
#     _, foam_mask = cv2.threshold(l, 80, 255, cv2.THRESH_BINARY)

#     # Usa la maschera per migliorare le regioni chiare
#     foam_highlighted = cv2.bitwise_and(image, image, mask=foam_mask)
#     return foam_highlighted

# # Funzione per segmentare l'immagine con K-means
# def segment_image(image, k):
#     pixels = image.reshape((-1, 3))  # Appiattisci l'immagine
#     pixels = np.float32(pixels)  # Converti in float32
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
#     _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
#     centers = np.uint8(centers)  # Converti i centri in interi
#     segmented = centers[labels.flatten()]  # Applica i colori dei cluster
#     segmented = segmented.reshape(image.shape)  # Ricostruisci la forma originale
#     return segmented

# def remove_region(image, points):
#     # Funzione per rimuovere la regione definita da un insieme di punti (maschera poligonale)
#     import numpy as np
#     import cv2

#     # Leggi i punti dal file
#     with open(points, 'r') as file:
#         points = [tuple(map(int, line.strip().split(','))) for line in file]

#     # Ottieni le dimensioni dell'immagine
#     height, width = image.shape[:2]

#     # Aggiungi il punto in basso a sinistra per chiudere il contorno
#     bottom_left = (0, height)  # Punto in basso a sinistra
#     points.append(bottom_left)

#     # Crea una maschera nera (singolo canale)
#     mask = np.zeros((height, width), dtype=np.uint8)

#     # Converte i punti in un array NumPy
#     points = np.array(points, dtype=np.int32)

#     # Riempi il poligono sulla maschera
#     cv2.fillPoly(mask, [points], 255)

#     # Inverti la maschera
#     inverted_mask = cv2.bitwise_not(mask)

#     # Applica la maschera invertita all'immagine
#     if len(image.shape) == 3:  # Immagine a colori
#         masked_image = cv2.bitwise_and(image, image, mask=inverted_mask)
#     else:  # Immagine in scala di grigi
#         masked_image = cv2.bitwise_and(image, image, mask=inverted_mask)

#     return masked_image

# Configura il video
video_path = "videos/5.mp4"  # Cambia con il percorso del tuo video
clusters = 2  # Numero di cluster per K-means

# Funzione per elaborare il video
def process_video():
    cap = cv2.VideoCapture(video_path)  # Carica il video
    frame_idx = 0  # Indice del frame

    while cap.isOpened():
        ret, frame = cap.read()  # Leggi il frame corrente
        if not ret:
            break  # Esci se non ci sono più frame

        original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Converti a RGB
        original_height, original_width = original_image.shape[:2]
        #(center_x, center_y), radius = compute_parameters(frame_idx)  # Calcola il centro e il raggio
        #original_image = remove_region(original_image, "points_to_crop.txt")  # Applica la maschera poligonale

        # Applica la maschera circolare
        #masked_image = apply_circular_mask(original_image, (center_x, center_y), radius)

        # Filtraggio e miglioramento dell'immagine
        blurred_image = cv2.medianBlur(cv2.GaussianBlur(original_image, (5, 5), 0), 5)
        blurred_image = cv2.convertScaleAbs(blurred_image, alpha=1.8, beta=1.0) # Aumenta il contrasto

        #foam_image = isolate_foam(blurred_image)  # Isola la schiuma

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
        
        # print(contours.__len__())
        
        for contour in contours:
            resized_contour = np.array([[(int(p[0] * scale_x), int(p[1] * scale_y))] for p in contour])
            cv2.drawContours(segmented_image, [contour], -1, (0, 255, 0), 2)
            cv2.drawContours(original_image, [resized_contour], -1, (0, 255, 0), 2)  # Disegna anche sull'originale
            
            # Calcola e stampa l'area
            area = calculate_contour_area(contour, scale_x, scale_y)
            #print(f"Frame {image_index}: Area del contorno = {area:.2f} pixel^2")

            # (Opzionale) Mostra l'area sull'immagine
            text_position = (10, 30)  # Posizione del testo
            cv2.putText(original_image, f"Area: {area:.2f}", text_position, 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        resized_original = cv2.resize(original_image, resized_dim)
        combined_image = np.vstack((resized_original, segmented_image))
        cv2.imshow("Video Segmentation (Press 'Esc' to exit)", cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))

        # Incrementa l'indice del frame
        frame_idx += 1

        # Interrompi con il tasto 'Esc'
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Esci premendo 'Esc'
            break

    cap.release()  # Rilascia il video
    cv2.destroyAllWindows()  # Chiudi tutte le finestre di visualizzazione

# Esegui il programma
if __name__ == "__main__":
    process_video()
