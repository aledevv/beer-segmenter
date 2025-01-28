import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class ImageEditor:
    def __init__(self, root, image_folder):
        self.root = root
        self.image_folder = image_folder

        # Filtra i file immagine (escludendo file come .DS_Store)
        self.images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        self.current_image_index = 0
        self.dots = []

        # Carica la prima immagine
        self.load_image()

        # Crea canvas per disegnare sopra
        self.canvas = tk.Canvas(root, width=self.image.width, height=self.image.height)
        self.canvas.pack()

        # Associa l'evento click del mouse per disegnare il puntino
        self.canvas.bind("<Button-1>", self.draw_dot)

        # Aggiungi i tasti per navigare tra le immagini
        self.root.bind("<n>", self.next_image)
        self.root.bind("<p>", self.previous_image)

    def load_image(self):
        """Carica e visualizza l'immagine corrente"""
        image_path = os.path.join(self.image_folder, self.images[self.current_image_index])
        self.image = Image.open(image_path)
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def draw_dot(self, event):
        """Disegna un puntino rosso sull'immagine dove si fa clic"""
        x, y = event.x, event.y
        self.dots.append((x, y))
        self.canvas.create_oval(x-5, y-5, x+5, y+5, fill='red', outline='red')

    def next_image(self, event=None):
        """Vai all'immagine successiva"""
        self.current_image_index = (self.current_image_index + 1) % len(self.images)
        self.load_image()
        self.dots.clear()  # Reset dots when changing image

    def previous_image(self, event=None):
        """Vai all'immagine precedente"""
        self.current_image_index = (self.current_image_index - 1) % len(self.images)
        self.load_image()
        self.dots.clear()  # Reset dots when changing image

# Crea la finestra principale
root = tk.Tk()

# Scegli la cartella da cui caricare le immagini
image_folder = 'images/'

if image_folder:
    editor = ImageEditor(root, image_folder)
    root.mainloop()
else:
    print("Nessuna cartella selezionata!")
