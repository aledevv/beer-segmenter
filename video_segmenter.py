import cv2
import numpy as np
import os

def load_and_preprocess_image(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 200, 255, cv2.THRESH_TRUNC)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    blurred = cv2.medianBlur(blurred, 11)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(4, 4))
    blurred = clahe.apply(blurred)
    edges = cv2.Canny(blurred, threshold1=30, threshold2=80)
    return edges

def remove_region(image, path):
    if not os.path.exists(path):
        return image  # Se il file non esiste, non rimuovere nulla
    
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
            
        local_distances = []
        for j in range(-half_window, half_window + 1):
            idx = (i + j) % len(distances)
            if distances[idx] is not None:
                local_distances.append(distances[idx])
        
        if not local_distances:
            continue
        
        local_mean = np.mean(local_distances)
        local_std = np.std(local_distances)
        current_dist = distances[i]
        
        lower_threshold = local_mean - 1 * local_std/2
        upper_threshold = local_mean + 1 * local_std
        
        if lower_threshold <= current_dist <= upper_threshold:
            final_points.append(candidate_points[i])
    
    if len(final_points) < 3:
        return None
        
    return np.array(final_points, dtype=np.int32).reshape((-1, 1, 2))

def process_frame(frame, center):
    edges = load_and_preprocess_image(frame)
    edges = remove_region(edges, "points_to_crop.txt")
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    inner_contour = find_inner_contour(edges, center)
    output_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    if inner_contour is not None:
        cv2.drawContours(output_img, [inner_contour], -1, (0, 0, 255), 2)
        cv2.drawContours(frame, [inner_contour], -1, (0, 0, 255), 2)
    
    cv2.circle(frame, center, 5, (0, 255, 0), -1)
    cv2.circle(output_img, center, 5, (0, 255, 0), -1)
    
    return frame

def process_video(video_path, output_path, center):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Errore nell'apertura del video")
        return
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = process_frame(frame, center)
        out.write(processed_frame)
    
    cap.release()
    out.release()
    print("Elaborazione completata. Video salvato in:", output_path)

def main():
    video_path = "videos/1.mp4"
    output_path = "1.avi"
    center = (300, 65)
    process_video(video_path, output_path, center)

if __name__ == "__main__":
    main()
