import cv2
import numpy as np
import os

from moviepy import VideoFileClip

input_video = "videos/raw_video.mov"  # Change to your input file
output_video = "raw_video.mp4"

clip = VideoFileClip(input_video)
clip.write_videofile(output_video, codec="libx264", audio_codec="aac")
print("Conversion complete!")



"""
def segment_foam(image_path):
    
    Esegue il processo di segmentazione su un'immagine specifica.
    
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Immagine non trovata: {image_path}")
    
    # Step 1: Preprocessamento con CLAHE
    enhanced_image = preprocess_image(image)

    # Step 2: Contorni con Canny
    edges = apply_canny(enhanced_image)

    return edges

# Per testare su un'immagine specifica
if __name__ == "__main__":
    image_path = "test_image.jpg"  # Sostituisci con il percorso dell'immagine
    result = segment_foam(image_path)  # Prova con o senza watershed

    # Visualizza il risultato
    cv2.imshow("Segmentazione", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""