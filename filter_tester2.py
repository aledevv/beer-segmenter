import cv2
import numpy as np
import os

def nothing(x):
    pass

def remove_region(image, points):
    # Aggiungi il punto in basso a sinistra per chiudere il contorno
    height, width = image.shape
    bottom_left = (0, height)  # Punto in basso a sinistra
    points.append(bottom_left)

    # Crea una maschera nera
    mask = np.zeros(image.shape, dtype=np.uint8)

    # Converte i punti in un array NumPy per usarli in un poligono
    points = np.array(points, dtype=np.int32)

    # Riempi il poligono sulla maschera
    cv2.fillPoly(mask, [points], 255)

    # Rimuovi la regione dell'immagine usando la maschera
    image = cv2.bitwise_and(image, image, mask=~mask)  # Applica la maschera invertita
    return image

def segment_multiple_images_with_advanced_filters(folder_path, output_folder, region_to_remove_file):
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
    
    # Aggiungi gli slider per luminosità e contrasto
    cv2.createTrackbar('Brightness', 'Segmentation', 50, 100, nothing)  # 0-100 per luminosità
    cv2.createTrackbar('Contrast', 'Segmentation', 50, 100, nothing)  # 0-100 per contrasto

    # Aggiungi gli slider per CLAHE
    cv2.createTrackbar('CLAHE Clip Limit', 'Segmentation', 2, 10, nothing)  # Limite di contrasto
    cv2.createTrackbar('CLAHE Tile Size', 'Segmentation', 8, 20, nothing)  # Dimensione griglia

    # Aggiungi gli slider per i parametri di decisione dei contorni
    cv2.createTrackbar('Min Contour Length', 'Segmentation', 30, 200, nothing)  # Lunghezza minima contorno
    cv2.createTrackbar('Min Ellipse Axis', 'Segmentation', 60, 200, nothing)  # Dimensione minima asse ellisse

    while True:
        # Carica l'immagine corrente
        image_path = os.path.join(folder_path, image_files[current_image_index])
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Carica i punti da points.txt
        with open(region_to_remove_file, 'r') as file:
            points = [tuple(map(int, line.strip().split(','))) for line in file]

        # Rimuovi la regione usando i punti
        image = remove_region(image, points)

        # Leggi i valori delle trackbar
        bilateral_d = cv2.getTrackbarPos('Bilateral D', 'Segmentation') or 1
        bilateral_sigma = cv2.getTrackbarPos('Bilateral Sigma', 'Segmentation') or 1
        morph_kernel = cv2.getTrackbarPos('Morph Kernel', 'Segmentation') or 1
        gaussian_ksize = cv2.getTrackbarPos('Gaussian Kernel', 'Segmentation') or 1
        canny_thresh1 = cv2.getTrackbarPos('Canny Thresh1', 'Segmentation')
        canny_thresh2 = cv2.getTrackbarPos('Canny Thresh2', 'Segmentation')
        adaptive_thresh = cv2.getTrackbarPos('Adaptive Threshold', 'Segmentation')

        # Aggiungi luminosità e contrasto
        brightness = cv2.getTrackbarPos('Brightness', 'Segmentation')
        contrast = cv2.getTrackbarPos('Contrast', 'Segmentation')
        
        # Regola luminosità e contrasto
        image = cv2.convertScaleAbs(image, alpha=contrast/50, beta=brightness-50)

        # Ottieni i parametri CLAHE
        clahe_clip_limit = cv2.getTrackbarPos('CLAHE Clip Limit', 'Segmentation')
        clahe_tile_size = cv2.getTrackbarPos('CLAHE Tile Size', 'Segmentation')

        # Ottieni i parametri per i contorni
        min_contour_length = cv2.getTrackbarPos('Min Contour Length', 'Segmentation')
        min_ellipse_axis = cv2.getTrackbarPos('Min Ellipse Axis', 'Segmentation')

        # Applica CLAHE
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(clahe_tile_size, clahe_tile_size))
        enhanced = clahe.apply(image)

        # Assicurati che i kernel siano dispari
        gaussian_ksize = gaussian_ksize if gaussian_ksize % 2 == 1 else gaussian_ksize + 1
        morph_kernel = morph_kernel if morph_kernel % 2 == 1 else morph_kernel + 1

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

        # Rileva contorni
        contours, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_with_contours = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)  # Converti in BGR per colorare i contorni

        # Filtra i contorni per trovare ellissi o cerchi (grandi)
        for contour in contours:
            if len(contour) >= 5:  # Solo i contorni con abbastanza punti per trovare un'ellisse
                ellipse = cv2.fitEllipse(contour)
                (center, axes, angle) = ellipse

                # Ignora ellissi troppo piccole
                if axes[0] > min_ellipse_axis and axes[1] > min_ellipse_axis:  # Modifica la dimensione minima degli assi
                    cv2.ellipse(image_with_contours, ellipse, (0, 255, 0), 2)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        bilateral_filtered = cv2.cvtColor(bilateral_filtered, cv2.COLOR_GRAY2BGR)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        adaptive = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)

        # Combina i risultati
        combined = np.hstack((image_rgb, bilateral_filtered, edges, adaptive, image_with_contours))

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
                f.write(f"Brightness: {brightness}\n")
                f.write(f"Contrast: {contrast}\n")
                f.write(f"CLAHE Clip Limit: {clahe_clip_limit}\n")
                f.write(f"CLAHE Tile Size: {clahe_tile_size}\n")
                f.write(f"Min Contour Length: {min_contour_length}\n")
                f.write(f"Min Ellipse Axis: {min_ellipse_axis}\n")

            print(f"Immagine salvata: {output_image_file}")
            print(f"Parametri salvati: {output_settings_file}")

    cv2.destroyAllWindows()

# Esegui lo script specificando il percorso della cartella delle immagini e della cartella di output
# La regione da rimuovere è specificata nel file "points.txt"
region_to_remove_file = 'points.txt'
segment_multiple_images_with_advanced_filters('images', 'filtered_output', region_to_remove_file)
