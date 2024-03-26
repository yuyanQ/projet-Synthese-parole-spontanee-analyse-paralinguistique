clearinfo
directory$ = "/Users/yuyanq/Desktop/M2-1/UE2Parole/Module3_synthese_de_la_parole/projet/ressources/"
baseName$ = "T11"

textgridName$ = baseName$ + ".TextGrid"

# Chargement  .TextGrid
Read from file... 'directory$''textgridName$'

# Sélection du TextGrid
selectObject: "TextGrid 'baseName$'"

# Obtention du nombre d'intervalles dans le troisième tier
numberOfIntervals = Get number of intervals: 1



# Boucle sur chaque intervalle
for i from 1 to numberOfIntervals
    # Sélection du TextGrid
    selectObject: "TextGrid 'baseName$'"
    
    # Obtention de l'étiquette (label) et des points de temps de l'intervalle
    label$ = Get label of interval: 1, i

    
    # Affichage des résultats
	 if label$ <> ""
			appendInfoLine: label$
	 endif
endfor
