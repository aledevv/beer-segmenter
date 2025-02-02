import cv2
import numpy as np
import os
from kmeans import segment_image, find_lightest_cluster_contours, isolate_foam, remove_region
from tqdm import tqdm
from kmeans_test import preprocess_and_segment
use_kmeans = False

def load_and_preprocess_image(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 200, 255, cv2.THRESH_TRUNC)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    blurred = cv2.medianBlur(blurred, 13)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(4, 4))
    blurred = clahe.apply(blurred)
    edges = cv2.Canny(blurred, threshold1=30, threshold2=80)
    return edges

def remove_region_from_edges(image, path):
    if not os.path.exists(path):
        return image
    
    with open(path, 'r') as file:
        points = [tuple(map(int, line.strip().split(','))) for line in file]
    
    height, width = image.shape
    bottom_left = (0, height)
    points.append(bottom_left)
    mask = np.zeros(image.shape, dtype=np.uint8)
    points = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    return cv2.bitwise_and(image, image, mask=~mask)

def find_inner_contour(edges, center, prev_contour=None, alpha=0.7, num_rays=360, window_size=50, history=None, history_length=5):
    height, width = edges.shape
    angles = np.linspace(0, 2 * np.pi, num_rays)
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
            candidate_points.append(None)
            distances.append(None)

    final_points = []
    half_window = window_size // 2

    for i in range(len(candidate_points)):
        if candidate_points[i] is None:
            continue
            
        local_distances = [distances[idx] for j in range(-half_window, half_window + 1)
                           if (idx := (i + j) % len(distances)) is not None and distances[idx] is not None]
        
        if not local_distances:
            continue
        
        local_mean = np.mean(local_distances)
        local_std = np.std(local_distances)
        current_dist = distances[i]

        lower_threshold = local_mean - 1 * local_std / 2
        upper_threshold = local_mean + 1 * local_std

        if lower_threshold <= current_dist <= upper_threshold:
            final_points.append(candidate_points[i])

    if len(final_points) < 3:
        return prev_contour
    
    new_contour = np.array(final_points, dtype=np.int32).reshape((-1, 1, 2))

    if prev_contour is not None and prev_contour.shape == new_contour.shape:
        new_contour = (alpha * prev_contour + (1 - alpha) * new_contour).astype(np.int32)

    # Mantiene una storia dei contorni e ne calcola la media
    if history is not None:
        history.append(new_contour)
        if len(history) > history_length:
            history.pop(0)
        avg_contour = np.mean(np.array(history), axis=0).astype(np.int32)
        return avg_contour

    return new_contour


def calculate_contour_area(contour):
    if contour is None:
        return 0
    return cv2.contourArea(contour)

def move_center_smoothly(initial, final, step, total_steps):
    progress = step / total_steps
    easing = progress  # Movimento lineare
    new_x = int(initial[0] + (final[0] - initial[0]) * easing)
    new_y = int(initial[1] + (final[1] - initial[1]) * easing)
    return (new_x, new_y)

def find_kmeans_contour(frame):
    original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    original_height, original_width = original_image.shape[:2]

    #fixed_image = remove_region(original_image, "points_to_crop_kmeans.txt")
    blurred_image = cv2.medianBlur(cv2.GaussianBlur(original_image, (5, 5), 0), 5)
    foam_image = isolate_foam(blurred_image)

    resized_dim = (400, 400)
    resized_image = cv2.resize(foam_image, resized_dim)
    segmented_image, labels, centers = segment_image(resized_image, k=2)
    contours = find_lightest_cluster_contours(segmented_image, labels, centers)

    scale_x = original_width / resized_dim[0]
    scale_y = original_height / resized_dim[1]
    
    return np.array([[(int(p[0][0] * scale_x), int(p[0][1] * scale_y))] for p in contours[0]])


def remove_inner_contours(contours, mask):
    filtered_contours = []
    for contour in contours:
        mask_filled = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(mask_filled, [contour], -1, 255, thickness=cv2.FILLED)
        if np.any(cv2.bitwise_and(mask, mask_filled)):  # Se il contorno è dentro la regione mascherata
            continue
        filtered_contours.append(contour)
    return filtered_contours

def merge_contours(contours):
    merged = np.vstack(contours) if contours else np.array([])
    return merged.reshape((-1, 1, 2)) if merged.size else None

def find_kmeans_contour(frame):
    original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    original_height, original_width = original_image.shape[:2]
    blurred_image = cv2.medianBlur(cv2.GaussianBlur(original_image, (5, 5), 0), 5)
    foam_image = isolate_foam(blurred_image)
    resized_dim = (400, 400)
    resized_image = cv2.resize(foam_image, resized_dim)
    segmented_image, labels, centers = segment_image(resized_image, k=2)
    contours = find_lightest_cluster_contours(segmented_image, labels, centers)
    scale_x = original_width / resized_dim[0]
    scale_y = original_height / resized_dim[1]
    return [np.array([[(int(p[0][0] * scale_x), int(p[0][1] * scale_y))] for p in contour]) for contour in contours]

def process_frame(frame, center, prev_contour, max_area):
    global use_kmeans
    #edges = load_and_preprocess_image(frame)
    
    original, segmented, foam_mask = preprocess_and_segment(frame, 3)
    edges = cv2.Canny(foam_mask, threshold1=30, threshold2=80)
    
    # Rimuovi i bordi che cadono dentro i punti nel file "points_to_crop.txt"
    mask = remove_region_from_edges(np.zeros_like(edges), "points_to_crop.txt")
    edges = cv2.bitwise_and(edges, edges, mask=~mask)  # Applica la maschera per rimuovere i bordi
    
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    if use_kmeans:
        contours = find_kmeans_contour(frame)
        filtered_contours = remove_inner_contours(contours, mask)
        inner_contour = merge_contours(filtered_contours)
    else:
        inner_contour = find_inner_contour(edges, center, prev_contour)
    
    area = calculate_contour_area(inner_contour)
    output_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    if area > 150000 and not use_kmeans:
        use_kmeans = True
    
    if area < 0.5 * max_area:
        inner_contour = prev_contour
    else:
        max_area = max(area, max_area)
    
    if inner_contour is not None:
        cv2.drawContours(output_img, [inner_contour], -1, (0, 0, 255), 2)
        cv2.drawContours(frame, [inner_contour], -1, (0, 0, 255), 2)

        # Aggiungi il fitting dell'ellisse
        if len(inner_contour) >= 5:  # FitEllipse richiede almeno 5 punti
            ellipse = cv2.fitEllipse(inner_contour)
            cv2.ellipse(frame, ellipse, (0, 255, 0), 2)  # Disegna l'ellisse verde
            cv2.ellipse(output_img, ellipse, (0, 255, 0), 2)
    
    # Mostra l'area calcolata
    cv2.putText(frame, f"Area: {area:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.circle(frame, center, 5, (0, 255, 0), -1)
    cv2.circle(output_img, center, 5, (0, 255, 0), -1)
    
    return frame, inner_contour, max_area



def process_video(video_path, output_path, initial_center, final_center):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Errore nell'apertura del video")
        return
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    temp_output = "temp.avi"
    out = cv2.VideoWriter(temp_output, fourcc, fps, (frame_width, frame_height))
    
    prev_contour = None
    max_area = 0  # Inizializza l'area massima
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Progress bar con tqdm
    for step in tqdm(range(frame_count), desc="Elaborazione video", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break
        
        center = move_center_smoothly(initial_center, final_center, step, frame_count*0.9)
        processed_frame, prev_contour, max_area = process_frame(frame, center, prev_contour, max_area)
        out.write(processed_frame)
    
    cap.release()
    out.release()
    
    os.system(f"ffmpeg -i temp.avi -vcodec libx264 {output_path}")
    os.remove("temp.avi")
    print("Elaborazione completata. Video salvato in:", output_path)


def main():
    video_path = "videos/5.mp4"
    output_path = "output5e.mp4"
    initial_center = (300, 65)
    final_center = (386, 279)
    process_video(video_path, output_path, initial_center, final_center)

if __name__ == "__main__":
    main()
