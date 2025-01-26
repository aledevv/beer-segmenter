import cv2
import os
from pathlib import Path

# Importa le funzioni dal primo script
from segmenter import segment_foam
from watershed import apply_watershed

def batch_process_images(input_folder, output_folder):
    """
    Processa tutte le immagini in una cartella e salva i risultati in un'altra cartella.
    """
    # Crea la cartella di output se non esiste
    os.makedirs(output_folder, exist_ok=True)

    # Itera su tutte le immagini nella cartella
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        if not (file_path.endswith(".jpg") or file_path.endswith(".png")):
            continue  # Salta file non immagine

        try:
            # Segmentazione
            result = segment_foam(file_path)
            # image = cv2.imread(file_path)
            # markers, result = apply_watershed(image)

            # Salva il risultato
            output_path = os.path.join(output_folder, file_name)
            cv2.imwrite(output_path, result)
            print(f"Processato: {file_name}")
        except Exception as e:
            print(f"Errore con {file_name}: {e}")

if __name__ == "__main__":
    input_folder = "test_imgs/"
    output_folder = "test_output/"

    # Batch processing
    batch_process_images(input_folder, output_folder)
