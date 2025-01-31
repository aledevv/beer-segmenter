# beer-segmenter
Project for the course of Signal Image Video regarding a simple segmenter to detect beer foam through image processing techniques. This task was previously solved using AI..


- add to repo final test images (currently ignored due to .gitignore)


IDEA:
Usare kmeans per ottenere una segmentazione più dettagliata. ✅
usare il canny per dedurre un ellisse. Questa viene dedotta a partire da un centro che viene fornito. ✅
    

    ✅ avere o una lista di punti o un'equazione che descrive il movimento del centro. OTTENERE UNA SORTA DI EQUAZIONE per centro, radius e min_length, parametri dell'ellisse.

    Utilizzando lo script dell'ellipse estimator coi 3 parametri ho provato circa 400 frame e ho ottenuto i seguenti risultati

    Risultati dati:

    Raggio (Radius):
    y = -0.0000x² + 0.5295x + 145.4149

    Centro X:
    y = 0.0004x² + 0.0198x + 300.1078

    Centro Y:
    y = -0.0006x² + 1.0520x + 9.7382

    Min Length:
    

    

    DA FARE: unire ellisse e kmeans (usare come segmentazione roba racchiusa, ma anche fuori con una certa approssimazione, dall'ellisse)
