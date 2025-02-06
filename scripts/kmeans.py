import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

# Configure the image folder
image_folder = "frames"  # Change to the path of your folder
points = 'points_to_crop.txt'
clusters = 2  # Number of clusters for K-means

# Sort files alphabetically and allow choosing a starting frame
image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
image_index = 290  # Modify this value to start from a specific frame

def preprocess_and_segment(img, n_clusters=3):
    # Read the image
    if img is None:
        raise ValueError("Unable to read the image")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Increase contrast with CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(gray)
    
    # 2. Apply bilateral filter to reduce noise while preserving edges
    bilateral = cv2.bilateralFilter(img_clahe, 9, 75, 75)
    
    # 3. Compute LBP for textures
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    
    # 4. Extract V channel from HSV space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:,:,2]
    
    # Normalize all features between 0 and 1
    bilateral_norm = bilateral / 255.0
    lbp_norm = (lbp - lbp.min()) / (lbp.max() - lbp.min())
    v_norm = v_channel / 255.0
    
    # Combine features
    h, w = gray.shape
    features = np.column_stack([
        bilateral_norm.reshape(-1),
        lbp_norm.reshape(-1),
        v_norm.reshape(-1)
    ])
    
    # Apply k-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    
    # Reshape labels into image form
    segmented = labels.reshape(h, w)
    
    # Identify the foam cluster (assume it is the brightest cluster)
    cluster_means = []
    for i in range(n_clusters):
        cluster_mean = np.mean(bilateral_norm.reshape(-1)[labels == i])
        cluster_means.append((i, cluster_mean))
    
    # Sort clusters by average brightness
    foam_cluster = max(cluster_means, key=lambda x: x[1])[0]
    
    # Create binary mask for foam
    foam_mask = (segmented == foam_cluster).astype(np.uint8) * 255
    
    return img, segmented, foam_mask


def isolate_foam(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, _, _ = cv2.split(lab)
    _, foam_mask = cv2.threshold(l, 80, 255, cv2.THRESH_BINARY)
    foam_highlighted = cv2.bitwise_and(image, image, mask=foam_mask)
    return foam_highlighted

# Function to segment the image with K-means
def segment_image(image, k):
    pixels = image.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.8)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 18, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()].reshape(image.shape)
    return segmented, labels.reshape(image.shape[:2]), centers


def find_most_bordering_contour(segmented, color, given_contour):
    mask = cv2.inRange(segmented, color, color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:  # If there are no contours, return an empty list
        return []
    
    # Initialize variables for the maximum and the corresponding contour index
    max_common = 0
    best_contour_idx = -1  # Index of the contour with the maximum number of common points

    # Iterate through the contours and compare with given_contour
    for i, contour in enumerate(contours):
        # Find common elements between given_contour and the current contour
        common_elements = np.intersect1d(given_contour, contour)
        common_count = len(common_elements)  # Number of common points
        
        # If the number of common points is greater than the maximum found so far, update
        if common_count > max_common:
            max_common = common_count
            best_contour_idx = i  # Save the index of the contour with the maximum number of common points
        
    return contours[best_contour_idx]  # Keep only the largest contour

def find_largest_cluster_contours(segmented, color):
    """Find the largest contour for the cluster with the specified label."""
    mask = cv2.inRange(segmented, color, color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:  # If there are no contours, return an empty list
        return []

    # Sort contours by area and take the largest
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    return contours[:1]  # Keep only the largest contour


# Calculate the area of the largest contour
def calculate_contour_area(contour, scale_x, scale_y):
    if contour is not None and len(contour) > 0:
        scaled_contour = np.array([[(p[0] * scale_x, p[1] * scale_y)] for p in contour], dtype=np.float32)
        area = cv2.contourArea(scaled_contour)  # Area in pixels
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

def merge_contours_outer(image_shape, contour1, contour2):
    """Merge two contours and find the outer contour of their combination."""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)  # Use only the 2D part of the image

    # Draw the two contours filling them
    cv2.drawContours(mask, [contour1], -1, 255, thickness=cv2.FILLED)
    cv2.drawContours(mask, [contour2], -1, 255, thickness=cv2.FILLED)

    # Find the outer contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours[0] if contours else None  # Return the largest outer contour

def show_image():
    global image_index, image_files
    while True:
        image_path = os.path.join(image_folder, image_files[image_index])
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_height, original_width = original_image.shape[:2]

        resized_dim = (400, 400)
        resized_image = cv2.resize(original_image, resized_dim)
        segmented_image, labels, centers = segment_image(resized_image, clusters)
        
        centers = sorted(centers, key=lambda x: x[0], reverse=True)
        
        contours_white = find_largest_cluster_contours(segmented_image, centers[0])
        
        if (clusters > 2):
            contours_grey = find_most_bordering_contour(segmented_image, centers[1], contours_white)
        
            # Merge the two contours
            contours = merge_contours_outer(segmented_image.shape, contours_white[0], contours_grey)
        else:
            contours = contours_white[0]

        scale_x = original_width / resized_dim[0]
        scale_y = original_height / resized_dim[1]
        
        for contour in contours:
            resized_contour = np.array([[(int(p[0] * scale_x), int(p[1] * scale_y))] for p in contour])  # Fix for point p
            cv2.drawContours(segmented_image, [contour], -1, (0, 255, 0), 2)
            cv2.drawContours(original_image, [resized_contour], -1, (0, 255, 0), 2)  # Also draw on the original
            
            # Calculate and print the area
            area = calculate_contour_area(contour, scale_x, scale_y)
            print(f"Frame {image_index}: Contour area = {area:.2f} pixel^2")

        # (Optional) Show the area on the image
        text_position = (10, 30)  # Text position
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
        print("No images found in the specified folder.")
