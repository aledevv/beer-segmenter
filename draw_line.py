import cv2
import numpy as np

# Variabili globali
points = []  # Lista dei punti della polilinea

# Funzione per gestire i click del mouse
def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:  # Se si clicca col tasto sinistro
        points.append((x, y))  # Aggiungi il punto alla lista
        # Copia dell'immagine originale per non sovrascrivere
        img_copy = img.copy()
        # Disegna la polilinea fino al punto corrente
        if len(points) > 1:
            cv2.polylines(img_copy, [np.array(points)], isClosed=False, color=(0, 255, 0), thickness=2)
        # Disegna i punti
        for point in points:
            cv2.circle(img_copy, point, 5, (0, 0, 255), -1)  # Punti rossi
        # Mostra l'immagine aggiornata
        cv2.imshow("Polilinea in evoluzione", img_copy)

# Funzione per salvare i punti in un file .txt
def save_points_to_file(points, filename="points.txt"):
    with open(filename, 'w') as f:
        for point in points:
            f.write(f"{point[0]}, {point[1]}\n")
    print(f"Punti salvati nel file {filename}")

# Carica l'immagine
img = cv2.imread("test_imgs/frame_no_0396.png")

# Crea una finestra e imposta la funzione di callback per il mouse
cv2.imshow("Polilinea in evoluzione", img)
cv2.setMouseCallback("Polilinea in evoluzione", mouse_callback)

# Attendi che l'utente prema un tasto
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC per uscire
        break
    elif key == ord('s'):  # 's' per salvare i punti
        save_points_to_file(points)  # Salva i punti in un file

cv2.destroyAllWindows()

# Stampa la lista dei punti
print("Punti della polilinea:")
for point in points:
    print(f"({point[0]}, {point[1]})")
