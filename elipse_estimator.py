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

# Funzione principale
def process_and_visualize(image_path, points_path, min_length):
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

    # Mostra l'immagine risultante
    cv2.imshow('Contorni Filtrati e Ellisse', output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Salva l'immagine risultante (opzionale)
    cv2.imwrite('output_filtered_ellisse.jpg', output_img)

# Esegui il processo con un esempio
process_and_visualize('images/frame_no_0614.png', 'points_to_crop.txt', min_length=300)
