import cv2
import numpy as np
import os

# Configura la cartella delle immagini
image_folder = "images"  # Cambia con il percorso della tua cartella
points = 'points.txt'
clusters = 3  # Numero di cluster per K-means
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
image_index = 0  # Indice iniziale dell'immagine

def isolate_foam(image):
    # Converti l'immagine in spazio colore LAB per isolare la luminosità
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Applica una soglia adattiva sul canale L (luminosità)
    _, foam_mask = cv2.threshold(l, 80, 255, cv2.THRESH_BINARY)

    # Usa la maschera per migliorare le regioni chiare
    foam_highlighted = cv2.bitwise_and(image, image, mask=foam_mask)
    return foam_highlighted

# Funzione per segmentare l'immagine con K-means
def segment_image(image, k):
    pixels = image.reshape((-1, 3))  # Appiattisci l'immagine
    pixels = np.float32(pixels)  # Converti in float32
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)  # Converti i centri in interi
    segmented = centers[labels.flatten()]  # Applica i colori dei cluster
    segmented = segmented.reshape(image.shape)  # Ricostruisci la forma originale
    return segmented

def remove_region(image, points):
    import numpy as np
    import cv2

    # Read points from the file
    with open(points, 'r') as file:
        points = [tuple(map(int, line.strip().split(','))) for line in file]

    # Get the image dimensions
    if len(image.shape) == 2:  # Grayscale image
        height, width = image.shape
    elif len(image.shape) == 3:  # Color image
        height, width, _ = image.shape
    else:
        raise ValueError("Unexpected image shape: {}".format(image.shape))

    # Add the bottom-left point to close the contour
    bottom_left = (0, height)  # Bottom-left point
    points.append(bottom_left)

    # Create a black mask (single channel)
    mask = np.zeros((height, width), dtype=np.uint8)

    # Convert points to a NumPy array
    points = np.array(points, dtype=np.int32)

    # Fill the polygon on the mask
    cv2.fillPoly(mask, [points], 255)

    # Invert the mask
    inverted_mask = cv2.bitwise_not(mask)

    # Apply the inverted mask to the image
    if len(image.shape) == 3:  # Color image
        masked_image = cv2.bitwise_and(image, image, mask=inverted_mask)
    else:  # Grayscale image
        masked_image = cv2.bitwise_and(image, image, mask=inverted_mask)

    return masked_image


# Funzione principale per mostrare l'immagine
def show_image():
    global image_index, image_files
    while True:
        # Carica l'immagine corrente
        image_path = os.path.join(image_folder, image_files[image_index])
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        fixed_image = remove_region(original_image, points)
        
        blurred_image = cv2.GaussianBlur(fixed_image, (5, 5), 0)
        blurred_image = cv2.medianBlur(blurred_image, 5)
        
        foam_image = isolate_foam(blurred_image)
        
        # Ridimensiona per una visualizzazione più rapida
        resized_image = cv2.resize(foam_image, (400, 400))
        resized_original = cv2.resize(original_image, (400, 400))
        
        # Segmenta l'immagine (usiamo k=3 per separare meglio la schiuma)
        segmented_image = segment_image(resized_image, k=clusters)

        # Combina le due immagini (originale sopra, segmentata sotto)
        combined_image = np.vstack((resized_original, segmented_image))

        # Mostra l'immagine
        cv2.imshow("Image Segmentation (Press 'n' for next, 'p' for previous)", cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))

        # Comandi per navigare tra le immagini
        key = cv2.waitKey(0) & 0xFF
        if key == ord('n'):  # Prossima immagine
            image_index = (image_index + 1) % len(image_files)
        elif key == ord('p'):  # Immagine precedente
            image_index = (image_index - 1) % len(image_files)
        elif key == 27:  # Esci premendo 'Esc'
            break

    cv2.destroyAllWindows()

# Esegui il programma
if __name__ == "__main__":
    if len(image_files) == 0:
        print("Nessuna immagine trovata nella cartella specificata.")
    else:
        show_image()
