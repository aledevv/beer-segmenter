# beer-segmenter
Project for the course of Signal Image Video regarding a simple segmenter to detect beer foam through image processing techniques. This task was previously solved using AI..


- add to repo final test images (currently ignored due to .gitignore)


IDEA:
Usare kmeans per ottenere una segmentazione più dettagliata
usare il canny per dedurre un ellisse. Questa viene dedotta a partire da un centro che viene fornito.
    -> fare script per disegnare centro e ottenere ellisse, poi quando si dà conferma con un tasto, si aggiunge quel punto alla lista.

    Obiettivo: avere o una lista di punti o un'equazione che descrive il movimento del centro

    
A quel punto dovresti avere un'ellisse da dedurre per ogni frame. Usa l'ellisse per prendere solo una parte del kmeans.
