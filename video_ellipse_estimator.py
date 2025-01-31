import cv2
import numpy as np

class EllipseStabilizer:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.ellipse_params = None  # (center, axes, angle)

    def update(self, new_ellipse):
        center, axes, angle = new_ellipse  # Estrai i valori
        
        if self.ellipse_params is None:
            self.ellipse_params = (center, axes, angle)
        else:
            # Applica EMA separatamente a ogni componente
            smoothed_center = (
                (1 - self.alpha) * np.array(self.ellipse_params[0]) + self.alpha * np.array(center)
            )
            smoothed_axes = (
                (1 - self.alpha) * np.array(self.ellipse_params[1]) + self.alpha * np.array(axes)
            )
            smoothed_angle = (1 - self.alpha) * self.ellipse_params[2] + self.alpha * angle

            self.ellipse_params = (tuple(smoothed_center), tuple(smoothed_axes), smoothed_angle)

        return self.ellipse_params




def filter_short_contours(contours, min_length):
    return [contour for contour in contours if len(contour) >= min_length]

def load_and_preprocess_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    blurred = cv2.medianBlur(blurred, 7)
    edges = cv2.Canny(blurred, threshold1=30, threshold2=80)
    return edges

def apply_circular_mask(image, center, radius):
    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    return cv2.bitwise_and(image, image, mask=mask)

def remove_region(image, points):
    height, width = image.shape
    bottom_left = (0, height)
    points.append(bottom_left)
    mask = np.zeros(image.shape, dtype=np.uint8)
    points = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    return cv2.bitwise_and(image, image, mask=~mask)

def compute_parameters(frame_idx):
    radius = -0.0000 * (frame_idx ** 2) + 0.5295 * frame_idx + 145.4149
    center_x = 0.0004 * (frame_idx ** 2) + 0.0198 * frame_idx + 300.1078
    center_y = -0.0002 * (frame_idx ** 2) + 0.8531 * frame_idx + 31.0855
    return (int(center_x), int(center_y)), int(radius)

def process_video(video_path, points_path, output_path):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    
    with open(points_path, 'r') as file:
        points = [tuple(map(int, line.strip().split(','))) for line in file]
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None
    
    ellipse_stabilizer = EllipseStabilizer(alpha=0.2)  # Istanza della classe

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        (center_x, center_y), radius = compute_parameters(frame_idx)
        
        # Processamento immagine
        edges = load_and_preprocess_image(frame)
        edges = remove_region(edges, points.copy())
        edges = apply_circular_mask(edges, (center_x, center_y), radius)
        
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = filter_short_contours(contours, min_length=80)
        
        output_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(output_img, filtered_contours, -1, (0, 0, 255), 2)
        
        if filtered_contours:
            all_points = np.vstack(filtered_contours)
            if len(all_points) >= 5:
                new_ellipse = cv2.fitEllipse(all_points)
                smoothed_ellipse = ellipse_stabilizer.update(new_ellipse)
                
                cv2.ellipse(output_img, smoothed_ellipse, (0, 255, 0), 2)
                cv2.ellipse(frame, smoothed_ellipse, (0, 255, 0), 2)
        
        combined_img = np.hstack((frame, output_img))
        
        if out is None:
            height, width, _ = combined_img.shape
            out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
        
        out.write(combined_img)
        frame_idx += 1

        
    cap.release()
    out.release()
    cv2.destroyAllWindows()

process_video('videos/3.mov', 'points_to_crop.txt', 'output_video3.avi')
