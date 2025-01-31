import cv2
import numpy as np
import os

# Configura la cartella delle immagini
image_folder = "frames"  # Cambia con il percorso della tua cartella
points = 'points_to_crop.txt'
clusters = 2  # Numero di cluster per K-means

# Ordina i file in ordine alfabetico e permette di scegliere un frame iniziale
image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
image_index = 280  # Modifica questo valore per partire da un frame specifico

def isolate_foam(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, _, _ = cv2.split(lab)
    _, foam_mask = cv2.threshold(l, 80, 255, cv2.THRESH_BINARY)
    foam_highlighted = cv2.bitwise_and(image, image, mask=foam_mask)
    return foam_highlighted

# Funzione per segmentare l'immagine con K-means
def segment_image(image, k):
    pixels = image.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 18, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()].reshape(image.shape)
    return segmented, labels.reshape(image.shape[:2]), centers

# Trova i bordi del cluster più chiaro
def find_lightest_cluster_contours(segmented, labels, centers):
    lightest_cluster_idx = np.argmax(np.mean(centers, axis=1))
    mask = (labels == lightest_cluster_idx).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours[:1]  # Tieni solo il contorno più grande

# Calcola l'area del contorno più grande
def calculate_contour_area(contour, scale_x, scale_y):
    if contour is not None and len(contour) > 0:
        scaled_contour = np.array([[(p[0][0] * scale_x, p[0][1] * scale_y)] for p in contour], dtype=np.float32)
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

def show_image():
    global image_index, image_files
    while True:
        image_path = os.path.join(image_folder, image_files[image_index])
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_height, original_width = original_image.shape[:2]

        fixed_image = remove_region(original_image, points)
        blurred_image = cv2.medianBlur(cv2.GaussianBlur(fixed_image, (5, 5), 0), 5)
        foam_image = isolate_foam(blurred_image)

        resized_dim = (400, 400)
        resized_image = cv2.resize(foam_image, resized_dim)
        segmented_image, labels, centers = segment_image(resized_image, clusters)
        contours = find_lightest_cluster_contours(segmented_image, labels, centers)

        scale_x = original_width / resized_dim[0]
        scale_y = original_height / resized_dim[1]
        
        for contour in contours:
            resized_contour = np.array([[(int(p[0][0] * scale_x), int(p[0][1] * scale_y))] for p in contour])
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