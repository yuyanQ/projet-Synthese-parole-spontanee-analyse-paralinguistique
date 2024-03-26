clearinfo
# yuyanq/Desktop/M2-1/UE2Parole/
directory$ = "/Users/yuyanq/Desktop/M2-1/UE2Parole/Module3_synthese_de_la_parole/projet/ressources/"
baseName$ = "T11"
wavName$ = baseName$ + ".wav"
textgridName$ = baseName$ + ".TextGrid"

# Chargement des fichiers .wav et .TextGrid
Read from file... 'directory$''wavName$'
Read from file... 'directory$''textgridName$'

# Sélection du TextGrid
selectObject: "TextGrid 'baseName$'"

# Obtention du nombre d'intervalles dans le troisième tier
numberOfIntervals = Get number of intervals: 3

appendInfoLine: ";Pho   ", "Durée(ms)  ", " Pitch/F0 (Hz)"

# Sélection du Sound correspondant pour extraire la F0 et l'intensité
selectObject: "Sound 'baseName$'"

# Création de l'objet Pitch
To Pitch: 0.0, 75, 600 


# Boucle sur chaque intervalle
for i from 1 to numberOfIntervals
    # Sélection du TextGrid
    selectObject: "TextGrid 'baseName$'"
    
    # Obtention de l'étiquette (label) et des points de temps de l'intervalle
    label$ = Get label of interval: 3, i
    startTime = Get starting point: 3, i
    endTime = Get end point: 3, i
	
	
	# transformer en Milleseconde
	duration = round((endTime - startTime) * 1000)
    
    # Sélection de l'objet Pitch pour obtenir la valeur de F0
    selectObject: "Pitch 'baseName$'"
    f0 = Get value at time: startTime, "Hertz", "Linear"
	

    
    # Affichage des résultats et élimier les séquences sans f0
	 if label$ == ""
		appendInfoLine: "- " ,duration
	else 
		if f0 > 1
			roundF0 = round(f0)
			appendInfoLine: label$, " " ,duration, " 50 " ,roundF0 
    	else
			appendInfoLine: label$, " " ,duration
		endif
	endif
endfor

