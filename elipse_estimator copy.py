import cv2
import numpy as np

# Funzione per filtrare contorni troppo corti
def filter_short_contours(contours, min_length):
    return [contour for contour in contours if len(contour) >= min_length]

# Leggi l'immagine
def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (7, 7), 0)
    blurred = cv2.medianBlur(blurred, 7)
    edges = cv2.Canny(blurred, threshold1=30, threshold2=80)
    return edges

# Funzione per applicare una maschera circolare
def apply_circular_mask(image, center, radius):
    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)  # Disegna un cerchio bianco sulla maschera nera
    return cv2.bitwise_and(image, image, mask=mask)

# Funzione principale
def process_and_visualize(image_path, points_path, min_length):
    global radius, center
    
    # Leggi l'immagine e i punti per rimuovere regioni
    edges = load_and_preprocess_image(image_path)
    with open(points_path, 'r') as file:
        points = [tuple(map(int, line.strip().split(','))) for line in file]

    # Rimuovi la regione specificata dai punti
    def remove_region(image, points):
        height, width = image.shape
        bottom_left = (0, height)
        points.append(bottom_left)
        mask = np.zeros(image.shape, dtype=np.uint8)
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
        return cv2.bitwise_and(image, image, mask=~mask)

    edges = remove_region(edges, points)

    # Applica una maschera circolare
    #center = (edges.shape[1] // 2, edges.shape[0] // 2)  # Centro dell'immagine (modificabile)
    #radius = 100  # Raggio del cerchio (modificabile)
    edges = apply_circular_mask(edges, center, radius)

    # Dilatazione per chiudere i bordi
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)

    # Trova tutti i contorni
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtra i contorni troppo corti
    filtered_contours = filter_short_contours(contours, min_length)

    original_img = cv2.imread(image_path)

    # Disegna i contorni filtrati (colorati di rosso)
    output_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(output_img, filtered_contours, -1, (0, 0, 255), 2)  # Colore rosso

    # Combina tutti i contorni filtrati in un unico insieme di punti
    all_points = np.vstack(filtered_contours) if filtered_contours else None

    # Fitta un'ellisse sui punti combinati
    if all_points is not None and len(all_points) >= 5:
        ellipse = cv2.fitEllipse(all_points)
        cv2.ellipse(output_img, ellipse, (0, 255, 0), 2)  # Ellisse verde
        cv2.ellipse(original_img, ellipse, (0, 255, 0), 2)  # Disegna l'ellisse anche sull'originale

    # Combina le due immagini affiancate
    combined_img = np.hstack((original_img, output_img))

    # Mostra l'immagine risultante
    cv2.imshow('Immagine Originale e Filtrata con Ellisse', combined_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Salva l'immagine risultante (opzionale)
    cv2.imwrite('output_combined_ellisse.jpg', combined_img)

radius = 500
center = (350, 300)
min_length = 300

# Esegui il processo con un esempio
process_and_visualize('frames/frame_no_0785.png', 'points.txt', min_length=min_length)

