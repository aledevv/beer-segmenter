import cv2
import numpy as np
import os

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Rimuove i riflessi di luce eliminando i pixel troppo luminosi
    _, img = cv2.threshold(img, 190, 255, cv2.THRESH_TRUNC)
    
    
    
    blurred = cv2.GaussianBlur(img, (11, 11), 0)
    blurred = cv2.medianBlur(blurred, 13)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    blurred = clahe.apply(blurred)
    edges = cv2.Canny(blurred, threshold1=30, threshold2=80)
    # cv2.imshow("Preprocessed Image", edges)
    # cv2.waitKey(0)
    return edges

def apply_circular_mask(image, center, radius):
    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    return cv2.bitwise_and(image, image, mask=mask)

def remove_region(image, path):
    with open(path, 'r') as file:
        points = [tuple(map(int, line.strip().split(','))) for line in file]
    
    height, width = image.shape
    bottom_left = (0, height)
    points.append(bottom_left)
    mask = np.zeros(image.shape, dtype=np.uint8)
    points = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    return cv2.bitwise_and(image, image, mask=~mask)

def find_inner_contour(edges, center, num_rays=360, window_size=50):
    height, width = edges.shape
    angles = np.linspace(0, 2 * np.pi, num_rays)
    # Prima troviamo tutti i punti di bordo candidati
    candidate_points = []
    distances = []
    
    for angle in angles:
        found_point = False
        for r in range(1, min(width, height)):
            x = int(center[0] + r * np.cos(angle))
            y = int(center[1] + r * np.sin(angle))
            
            if 0 <= x < width and 0 <= y < height and edges[y, x] > 0:
                dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                candidate_points.append([x, y])
                distances.append(dist)
                found_point = True
                break
        
        if not found_point:
            # Se non troviamo un punto, mettiamo None per mantenere l'allineamento
            candidate_points.append(None)
            distances.append(None)
    
    # Ora analizziamo ogni punto considerando solo i suoi vicini
    final_points = []
    half_window = window_size // 2
    
    for i in range(len(candidate_points)):
        if candidate_points[i] is None:
            continue
            
        # Raccoglie le distanze dei punti vicini (considerando la circolarità)
        local_distances = []
        for j in range(-half_window, half_window + 1):
            idx = (i + j) % len(distances)
            if distances[idx] is not None:
                local_distances.append(distances[idx])
        
        if not local_distances:
            continue
        
        # Calcola statistiche locali
        local_mean = np.mean(local_distances)
        local_std = np.std(local_distances)
        current_dist = distances[i]
        
        # Definisce le soglie locali
        lower_threshold = local_mean - 2 * local_std
        upper_threshold = local_mean + 2 * local_std
        
        if lower_threshold <= current_dist <= upper_threshold:
            # Il punto è nella norma rispetto ai suoi vicini
            final_points.append(candidate_points[i])
        else:
            # Il punto è un outlier, cerchiamo un punto migliore
            found_better_point = False
            for r in range(int(current_dist), min(width, height)):
                x = int(center[0] + r * np.cos(angles[i]))
                y = int(center[1] + r * np.sin(angles[i]))
                
                if 0 <= x < width and 0 <= y < height and edges[y, x] > 0:
                    new_dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                    if lower_threshold <= new_dist <= upper_threshold:
                        final_points.append([x, y])
                        found_better_point = True
                        break
            
            if not found_better_point:
                # Se non troviamo un punto migliore, interpola tra i vicini validi
                if len(final_points) > 0:
                    # Cerca il prossimo punto valido
                    next_valid_point = None
                    for j in range(1, half_window):
                        next_idx = (i + j) % len(candidate_points)
                        if candidate_points[next_idx] is not None and \
                           lower_threshold <= distances[next_idx] <= upper_threshold:
                            next_valid_point = candidate_points[next_idx]
                            break
                    
                    if next_valid_point and len(final_points) > 0:
                        # Interpola tra l'ultimo punto valido e il prossimo punto valido
                        prev_point = final_points[-1]
                        x = int((prev_point[0] + next_valid_point[0]) / 2)
                        y = int((prev_point[1] + next_valid_point[1]) / 2)
                        final_points.append([x, y])
    
    if len(final_points) < 3:  # Serve un minimo di punti per formare un contorno
        return None
        
    return np.array(final_points, dtype=np.int32).reshape((-1, 1, 2))

def process_image(image_path, center, radius):
    edges = load_and_preprocess_image(image_path)
    edges = remove_region(edges, "points_to_crop.txt")
    #edges = apply_circular_mask(edges, center, radius)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    
    inner_contour = find_inner_contour(edges, center)
    original_img = cv2.imread(image_path)
    output_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    if inner_contour is not None:
        cv2.drawContours(output_img, [inner_contour], -1, (0, 0, 255), 2)
        cv2.drawContours(original_img, [inner_contour], -1, (0, 0, 255), 2)  # Disegna anche nell'originale
    
    # cv2.circle(original_img, center, radius, (255, 0, 0), 2)
    # cv2.circle(output_img, center, radius, (255, 0, 0), 2)
    cv2.circle(original_img, center, 5, (0, 255, 0), -1)  # Disegna il centro del cerchio
    cv2.circle(output_img, center, 5, (0, 255, 0), -1)  # Disegna il centro del cerchio
    
    combined_img = np.hstack((original_img, output_img))
    return combined_img

def main():
    image_path = "frames/frame_no_0010.png"
    center = (300, 95)
    radius = 235
    result_img = process_image(image_path, center, radius)
    cv2.imshow("Processed Image", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# Initial center:
# (290, 105)
# Initial radus
# 235