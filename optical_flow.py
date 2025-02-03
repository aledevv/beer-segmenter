import cv2
import numpy as np

# Carica i frame in scala di grigi
frame1 = cv2.imread('frames5/frame_no_0009.png', cv2.IMREAD_GRAYSCALE)
frame2 = cv2.imread('frames5/frame_no_0023.png', cv2.IMREAD_GRAYSCALE)

# Calcola il flusso ottico
flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

# Estrai componenti del flusso
hsv = np.zeros((frame1.shape[0], frame1.shape[1], 3), dtype=np.uint8)
hsv[..., 1] = 255  # Imposta la saturazione al massimo

mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])  # Magnitudine e angolo del movimento
hsv[..., 0] = ang * 180 / np.pi / 2  # Converti angolo in scala di colori
hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Normalizza la magnitudine

# Converti da HSV a BGR per visualizzazione
flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Mostra il risultato
cv2.imshow('Optical Flow', flow_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
