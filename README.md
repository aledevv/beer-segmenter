# beer-segmenter
Project for the course of Signal Image Video regarding a simple segmenter to detect beer foam through image processing techniques. This task was previously solved using AI..


- add to repo final test images (currently ignored due to .gitignore)


IDEA:
Usare kmeans per ottenere una segmentazione più dettagliata. ✅
usare il canny per dedurre un ellisse. Questa viene dedotta a partire da un centro che viene fornito. ✅
    

    ✅ avere o una lista di punti o un'equazione che descrive il movimento del centro. OTTENERE UNA SORTA DI EQUAZIONE per centro, radius e min_length, parametri dell'ellisse.

    Utilizzando lo script dell'ellipse estimator coi 3 parametri ho provato circa 400 frame e ho ottenuto i seguenti risultati

    Risultati dati:

    radius: “f(x)\=0.0002816⋅x2+0.2938⋅x−28.405”

    center x: “f(x)\=0.0000751⋅x2+0.082⋅x+250.987”
    center y: “f(x)\=0.000777⋅x2−0.2065⋅x+5.47”

    min length = “f(x)\=0.003077⋅x2−3.267⋅x+925.81”
    
    VEDERE SE VA ANCHE CON ALTRI VIDEO!!!

    

    DA FARE: unire ellisse e kmeans (usare come segmentazione roba racchiusa, ma anche fuori con una certa approssimazione, dall'ellisse)
