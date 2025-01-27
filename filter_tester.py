import cv2
import numpy as np
import os

def nothing(x):
    pass

def segment_multiple_images_with_advanced_filters(folder_path, output_folder):
    # Crea la cartella di output se non esiste
    os.makedirs(output_folder, exist_ok=True)

    # Ottieni tutte le immagini dalla cartella
    supported_formats = ('.jpg', '.png', '.jpeg', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(folder_path) if f.endswith(supported_formats)]
    
    if not image_files:
        print("Nessuna immagine trovata nella cartella.")
        return
    
    current_image_index = 0
    
    # Crea la finestra e i controlli
    cv2.namedWindow('Segmentation')
    cv2.createTrackbar('Bilateral D', 'Segmentation', 9, 50, nothing)  # Bilateral Diameter
    cv2.createTrackbar('Bilateral Sigma', 'Segmentation', 75, 200, nothing)  # Bilateral Sigma Color/Space
    cv2.createTrackbar('Morph Kernel', 'Segmentation', 1, 20, nothing)  # Morph Kernel Size
    cv2.createTrackbar('Gaussian Kernel', 'Segmentation', 1, 50, nothing)
    cv2.createTrackbar('Canny Thresh1', 'Segmentation', 50, 255, nothing)
    cv2.createTrackbar('Canny Thresh2', 'Segmentation', 150, 255, nothing)
    cv2.createTrackbar('Adaptive Threshold', 'Segmentation', 0, 255, nothing)

    while True:
        # Carica l'immagine corrente
        image_path = os.path.join(folder_path, image_files[current_image_index])
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Leggi i valori delle trackbar
        bilateral_d = cv2.getTrackbarPos('Bilateral D', 'Segmentation') or 1
        bilateral_sigma = cv2.getTrackbarPos('Bilateral Sigma', 'Segmentation') or 1
        morph_kernel = cv2.getTrackbarPos('Morph Kernel', 'Segmentation') or 1
        gaussian_ksize = cv2.getTrackbarPos('Gaussian Kernel', 'Segmentation') or 1
        canny_thresh1 = cv2.getTrackbarPos('Canny Thresh1', 'Segmentation')
        canny_thresh2 = cv2.getTrackbarPos('Canny Thresh2', 'Segmentation')
        adaptive_thresh = cv2.getTrackbarPos('Adaptive Threshold', 'Segmentation')

        # Assicurati che i kernel siano dispari
        gaussian_ksize = gaussian_ksize if gaussian_ksize % 2 == 1 else gaussian_ksize + 1
        morph_kernel = morph_kernel if morph_kernel % 2 == 1 else morph_kernel + 1

        # Applica CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)

        # Applica filtro Bilaterale
        bilateral_filtered = cv2.bilateralFilter(enhanced, bilateral_d, bilateral_sigma, bilateral_sigma)

        # Applica operazioni morfologiche (Opening + Closing)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        morph_open = cv2.morphologyEx(bilateral_filtered, cv2.MORPH_OPEN, kernel)
        morph_close = cv2.morphologyEx(morph_open, cv2.MORPH_CLOSE, kernel)

        # Applica Gaussian Blur
        blurred = cv2.GaussianBlur(morph_close, (gaussian_ksize, gaussian_ksize), 0)

        # Applica filtro Canny
        edges = cv2.Canny(blurred, canny_thresh1, canny_thresh2)

        # Applica thresholding adattivo
        adaptive = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, adaptive_thresh
        )

        # Combina i risultati
        combined = np.hstack((image, bilateral_filtered, edges, adaptive))

        # Mostra i risultati
        cv2.imshow('Segmentation', combined)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Esci con ESC
            break
        elif key == ord('n'):  # Passa all'immagine successiva
            current_image_index = (current_image_index + 1) % len(image_files)
        elif key == ord('p'):  # Torna all'immagine precedente
            current_image_index = (current_image_index - 1) % len(image_files)
        elif key == ord('s'):  # Salva l'elaborazione corrente
            # Genera il nome del file
            base_name = os.path.splitext(image_files[current_image_index])[0]
            output_image_file = os.path.join(output_folder, f"{base_name}_processed.png")
            output_settings_file = os.path.join(output_folder, f"{base_name}_settings.txt")

            # Salva l'immagine elaborata
            cv2.imwrite(output_image_file, combined)

            # Salva i parametri in un file di testo
            with open(output_settings_file, 'w') as f:
                f.write(f"Bilateral D: {bilateral_d}\n")
                f.write(f"Bilateral Sigma: {bilateral_sigma}\n")
                f.write(f"Morph Kernel: {morph_kernel}\n")
                f.write(f"Gaussian Kernel: {gaussian_ksize}\n")
                f.write(f"Canny Thresh1: {canny_thresh1}\n")
                f.write(f"Canny Thresh2: {canny_thresh2}\n")
                f.write(f"Adaptive Threshold: {adaptive_thresh}\n")

            print(f"Immagine salvata: {output_image_file}")
            print(f"Parametri salvati: {output_settings_file}")

    cv2.destroyAllWindows()

# Esegui lo script specificando il percorso della cartella delle immagini e della cartella di output
segment_multiple_images_with_advanced_filters('test_imgs', 'filtered_output')
