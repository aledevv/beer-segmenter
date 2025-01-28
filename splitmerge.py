import cv2
import numpy as np

def is_homogeneous(region, threshold):
    """Check if the region is homogeneous based on the intensity threshold."""
    min_val, max_val = np.min(region), np.max(region)
    return (max_val - min_val) <= threshold

def split_and_merge(image, threshold):
    """Segment the image by recursively splitting and merging regions."""
    
    def recursive_split(region):
        rows, cols = region.shape
        if rows <= 1 or cols <= 1:
            return np.zeros_like(region, dtype=np.uint8)
        
        if is_homogeneous(region, threshold):
            return np.ones_like(region, dtype=np.uint8)
        
        # Split the region into four quadrants
        mid_row, mid_col = rows // 2, cols // 2
        
        # Ensure quadrants are correctly sized
        top_left = region[:mid_row, :mid_col]
        top_right = region[:mid_row, mid_col:]
        bottom_left = region[mid_row:, :mid_col]
        bottom_right = region[mid_row:, mid_col:]
        
        # Create empty segmented image of the same size
        segmented_quadrants = np.zeros_like(region, dtype=np.uint8)
        
        # Recursive splitting and assignment to segmented_quadrants
        segmented_quadrants[:mid_row, :mid_col] = recursive_split(top_left)
        segmented_quadrants[:mid_row, mid_col:] = recursive_split(top_right)
        segmented_quadrants[mid_row:, :mid_col] = recursive_split(bottom_left)
        segmented_quadrants[mid_row:, mid_col:] = recursive_split(bottom_right)
        
        return segmented_quadrants

    def merge_regions(segmented):
        """Merge adjacent regions if they are similar."""
        # Placeholder function for merging adjacent regions if needed
        return segmented

    # Ensure the image is grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply the region splitting and merging algorithm
    segmented_image = recursive_split(image)
    segmented_image = merge_regions(segmented_image)
    
    return segmented_image

def main():
    # Load the image
    # url = 'https://www.experian.com/blogs/news/wp-content/uploads/2012/06/cars.png'  # Replace with your image URL
    # file_path = 'R.png'  # Fallback file path
    # url = None
    image = cv2.imread('images/frame_no_0666.png', cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print("Error loading image.")
        return

    # Set the threshold for homogeneity
    threshold = 6  # Adjust this value as needed

    # Segment the image
    result = split_and_merge(image, threshold)

    # Save and display the segmented image
    cv2.imwrite('Qsegmented_image.png', result * 255)
    cv2.imshow('Segmented Image', result * 255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()