import csv
from collections import defaultdict

def process_frames_file(input_filename, output_filename):
    """
    Processa un file CSV contenente dati dei frames e rimuove i duplicati,
    mantenendo solo l'ultima occorrenza di ogni frame.
    
    Args:
        input_filename (str): Nome del file di input
        output_filename (str): Nome del file di output
    """
    # Dizionario per memorizzare l'ultima occorrenza di ogni frame
    frames_dict = {}
    
    # Leggi il file CSV
    with open(input_filename, 'r') as file:
        csv_reader = csv.reader(file)
        
        # Itera su ogni riga e mantieni solo l'ultima occorrenza
        for row in csv_reader:
            if len(row) != 5:  # Verifica che la riga abbia tutti i campi necessari
                print(f"Riga saltata per formato non valido: {row}")
                continue
                
            frame_name = row[0]
            frames_dict[frame_name] = row
    
    # Scrivi i risultati nel file di output
    with open(output_filename, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        
        # Ordina i frame per nome prima di scrivere
        for frame_name in sorted(frames_dict.keys()):
            csv_writer.writerow(frames_dict[frame_name])

    print(f"Elaborazione completata!")
    print(f"Frame unici processati: {len(frames_dict)}")

# Esempio di utilizzo
if __name__ == "__main__":
    input_file = "saved_parameters.txt"  # Sostituisci con il nome del tuo file
    output_file = "saved_parameters.txt"
    
    try:
        process_frames_file(input_file, output_file)
    except FileNotFoundError:
        print(f"Errore: Il file {input_file} non è stato trovato.")
    except Exception as e:
        print(f"Si è verificato un errore: {str(e)}")