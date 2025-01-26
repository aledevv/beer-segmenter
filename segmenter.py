import cv2
import numpy as np
import os

def apply_canny(image, threshold1=40, threshold2=85):
    """
    Applica l'algoritmo Canny per il rilevamento dei bordi.
    """
    #threshold2 = threshold1 * 2  # Rapporto consigliato
    edges = cv2.Canny(image, threshold1, threshold2)
    return edges


def preprocess_image(image, clip_limit=1.0, tile_grid_size=(7, 7)):
    """
    Applica CLAHE con un filtro mediano per rimuovere il rumore.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 19)  # Riduce il rumore
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(blurred)
    return enhanced



def segment_foam(image_path):
    """
    Esegue il processo di segmentazione su un'immagine specifica.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Immagine non trovata: {image_path}")
    
    # Step 1: Preprocessamento con CLAHE
    enhanced_image = preprocess_image(image)

    # Step 2: Contorni con Canny
    edges = apply_canny(enhanced_image)

    return edges

# Per testare su un'immagine specifica
if __name__ == "__main__":
    image_path = "test_image.jpg"  # Sostituisci con il percorso dell'immagine
    result = segment_foam(image_path)  # Prova con o senza watershed

    # Visualizza il risultato
    cv2.imshow("Segmentazione", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
