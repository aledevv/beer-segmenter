import cv2
import numpy as np

"""
This script allows the user to draw a polyline on an image and filter out the "bubbles" in the polyline.
It has been used to manually annotate the region containing the metal pipe pouring the beer in the cup.
"""

# Global variables
points = []  # List of polyline points

# Function to handle mouse clicks
def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:  # If left mouse button is clicked
        points.append((x, y))  # Add the point to the list
        # Copy of the original image to avoid overwriting
        img_copy = img.copy()
        # Draw the polyline up to the current point
        if len(points) > 1:
            cv2.polylines(img_copy, [np.array(points)], isClosed=False, color=(0, 255, 0), thickness=2)
        # Draw the points
        for point in points:
            cv2.circle(img_copy, point, 5, (0, 0, 255), -1)  # Red points
        # Show the updated image
        cv2.imshow("Polyline in progress", img_copy)

# Function to save the points to a .txt file
def save_points_to_file(points, filename="points_to_crop_kmeans.txt"):
    with open(filename, 'w') as f:
        for point in points:
            f.write(f"{point[0]}, {point[1]}\n")
    print(f"Points saved to file {filename}")

# Load the image
img = cv2.imread("frames/frame_no_0300.png")

# Create a window and set the mouse callback function
cv2.imshow("Polyline in progress", img)
cv2.setMouseCallback("Polyline in progress", mouse_callback)

# Wait for the user to press a key
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to exit
        break
    elif key == ord('s'):  # 's' to save the points
        save_points_to_file(points)  # Save the points to a file

cv2.destroyAllWindows()

# Print the list of points
print("Polyline points:")
for point in points:
    print(f"({point[0]}, {point[1]})")
