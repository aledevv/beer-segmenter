# Configura la cartella delle immagini
image_folder = "frames5"  # Cambia con il percorso della tua cartella
points = 'points_to_crop.txt'
clusters = 2  # Numero di cluster per K-means

import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

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

def visualize_results(original, segmented, foam_mask):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Immagine Originale')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(segmented, cmap='viridis')
    plt.title('Segmentazione K-means')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(foam_mask, cmap='gray')
    plt.title('Maschera Schiuma')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Funzione principale
def main(image_path, n_clusters=3):
    try:
        image = cv2.imread(image_path)
        original, segmented, foam_mask = preprocess_and_segment(image, n_clusters)
        visualize_results(original, segmented, foam_mask)
        
        # Opzionale: applica operazioni morfologiche per pulire la maschera
        kernel = np.ones((5,5), np.uint8)
        foam_mask_cleaned = cv2.morphologyEx(foam_mask, cv2.MORPH_OPEN, kernel)
        foam_mask_cleaned = cv2.morphologyEx(foam_mask_cleaned, cv2.MORPH_CLOSE, kernel)
        
        # Mostra anche la versione pulita della maschera
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(foam_mask, cmap='gray')
        plt.title('Maschera Originale')
        plt.axis('off')
        
        plt.subplot(122)
        plt.imshow(foam_mask_cleaned, cmap='gray')
        plt.title('Maschera Pulita')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return foam_mask_cleaned
        
    except Exception as e:
        print(f"Errore durante l'elaborazione: {str(e)}")
        return None

# Uso
if __name__ == "__main__":
    image_path = "frames5/frame_no_0108.png"
    foam_mask = main(image_path, n_clusters=3)