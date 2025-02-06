import cv2
import os
import json
from tqdm import tqdm  # Importa tqdm per la barra di progresso

# Configura i percorsi dei file
input_video_path = 'videos/5.mp4'
output_frames_dir = 'frames/'  # Cartella per i frame
annotations_file = 'annotations.json'
class_name = 'cup'  # Modifica con il nome della tua classe
frame_number = 0

# Crea una directory per i frame se non esiste
os.makedirs(output_frames_dir, exist_ok=True)

# Funzione per salvare il frame come immagine
def save_frame(frame, frame_number):
    frame_filename = os.path.join(output_frames_dir, f'frame_no_{frame_number:04d}.png')
    cv2.imwrite(frame_filename, frame)
    return frame_filename

# Carica il video e dividi in frame
def extract_frames_from_video(video_path):
    global frame_number
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Ottieni il numero totale di frame
    
    # annotations = {
    #     'images': [],
    #     'annotations': [],
    #     'categories': [{'id': 1, 'name': class_name}]
    # }
    # annotation_id = 1

    # Usa tqdm per la barra di progresso
    with tqdm(total=total_frames, desc="Elaborazione dei frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Salva il frame come immagine
            frame_filename = save_frame(frame, frame_number)

            # Ottieni dimensioni del frame
            height, width, _ = frame.shape

            # # Aggiungi dettagli dell'immagine a COCO
            # annotations['images'].append({
            #     'id': frame_number,
            #     'file_name': os.path.basename(frame_filename),
            #     'width': width,
            #     'height': height
            # })

            # # Aggiungi annotazione per la classe
            # annotations['annotations'].append({
            #     'id': annotation_id,
            #     'image_id': frame_number,
            #     'category_id': 1,
            #     'segmentation': [],
            #     'area': width * height,
            #     'bbox': [0, 0, width, height],
            #     'iscrowd': 0
            # })

            frame_number += 1
            # annotation_id += 1

            pbar.update(1)  # Aggiorna la barra di progresso

    cap.release()

    # Salva il file di annotazione in formato COCO
    # with open(annotations_file, 'w') as f:
    #     json.dump(annotations, f, indent=4)

# Esegui la funzione di estrazione
#for i in range(3, 6):
input_video_path = input_video_path
extract_frames_from_video(input_video_path)
print(f"Frame estratti dal video {input_video_path}")