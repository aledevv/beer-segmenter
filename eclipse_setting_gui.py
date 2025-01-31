import cv2
import numpy as np
import os

def filter_short_contours(contours, min_length):
    return [contour for contour in contours if len(contour) >= min_length]

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    blurred = cv2.medianBlur(blurred, 7)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    blurred = clahe.apply(blurred)
    edges = cv2.Canny(blurred, threshold1=30, threshold2=80)
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

def process_image(image_path, min_length, center, radius):
    edges = load_and_preprocess_image(image_path)
    edges = remove_region(edges, "points_to_crop.txt")
    edges = apply_circular_mask(edges, center, radius)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = filter_short_contours(contours, min_length)
    original_img = cv2.imread(image_path)
    output_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(output_img, filtered_contours, -1, (0, 0, 255), 2)
    
    # Disegna il cerchio blu sia sull'immagine originale che su quella con i filtri
    cv2.circle(original_img, center, radius, (255, 0, 0), 2)
    cv2.circle(output_img, center, radius, (255, 0, 0), 2)
    
    all_points = np.vstack(filtered_contours) if filtered_contours else None
    if all_points is not None and len(all_points) >= 5:
        ellipse = cv2.fitEllipse(all_points)
        cv2.ellipse(output_img, ellipse, (0, 255, 0), 2)
        cv2.ellipse(original_img, ellipse, (0, 255, 0), 2)
    combined_img = np.hstack((original_img, output_img))
    return combined_img

def save_parameters(filename, frame, radius, center, min_length):
    with open(filename, 'a') as f:
        f.write(f"{frame},{radius},{center[0]},{center[1]},{min_length}\n")

def compute_parameters(frame_idx):
    radius = -0.0000 * (frame_idx ** 2) + 0.5295 * frame_idx + 145.4149
    center_x = 0.0004 * (frame_idx ** 2) + 0.0198 * frame_idx + 300.1078
    center_y = -0.0002 * (frame_idx ** 2) + 0.8531 * frame_idx + 31.0855
    return (int(center_x), int(center_y)), int(radius)

def main():
    folder = 'frames'
    output_file = 'saved_parameters.txt'
    frames = sorted([f for f in os.listdir(folder) if f.endswith('.png')])
    if not frames:
        print("No frames found!")
        return
    index = 0
    global radius, center, min_length
    radius, center, min_length = 140, (300, 60), 100
    while True:
        frame_path = os.path.join(folder, frames[index])
        # center, radius = compute_parameters(index)
        min_length = 60 if index < 100 else 100
        print(f"Frame: {frames[index]} - Radius: {radius}, Center: {center}, Min Length: {min_length}")
        result_img = process_image(frame_path, min_length, center, radius)
        cv2.imshow('Frame Analysis', result_img)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n') and index < len(frames) - 1:
            index += 1
        elif key == ord('p') and index > 0:
            index -= 1
        elif key == ord('+'):
            radius += 5
        elif key == ord('-'):
            radius = max(5, radius - 5)
        elif key == ord('w'):
            center = (center[0], center[1] - 5)
        elif key == ord('s'):
            center = (center[0], center[1] + 5)
        elif key == ord('a'):
            center = (center[0] - 5, center[1])
        elif key == ord('d'):
            center = (center[0] + 5, center[1])
        elif key == ord('e'):
            min_length += 5
        elif key == ord('r'):
            min_length = max(5, min_length - 5)
        elif key == ord('x'):
            save_parameters(output_file, frames[index], radius, center, min_length)
            print(f"Saved: {frames[index]} - Radius: {radius}, Center: {center}, Min Length: {min_length}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


# initial circle: Radius: 145, Center: (310, 55)