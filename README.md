Ce projet individuel sont l'analyse-synthèse, la synthèse de la parole à partir du texte ainsi que l'analyse paralinguistique de la parole.
Les logiciels MBrola et auDeep y permettent de réaliser une synthèse acoustique et une analyse paralinguistique. 

- Extraction de la prosodie avec les scripts sur Praat (Durée phonémique, label, silence, F0 moyenne, F0 initiale et centrale pour les sons voisés)
- Génération des durées avec la règle de Klatt et le logiciel eSpeck 
- Génération des pauses et de la courbe mélodique avec l'arbre syntaxique, la structure de performance, et le logiciel Emofilt (l'ajout d'une émotion de joie, de colère et de tristesse)
- Analyse paralinguistique des tours de parole (autoencoder avec DNN et classification non-supervisé)

En somme, il y a 5 fichiers d'origine prononcé par l'humain, 5 fichiers de synthèse par l'exemple (avec la prosodie extraite), 5 fichiers par génération des durées, 5 fichiers par génération des pauses et de la courbe mélodique). Les 20 fichiers synthétisés sont générés à l'aide des fichiers correspondants avec entension pho compatible avec le logiciel Mbrola. Les fichiers pho sont encodés en UTF-8. À part les opérations sur Praat en MacOs et l'utilisation d'Audeep, toutes les autres sont executées en MV Linux ubutun. 

Un corpus étiquete avec les états paralinguistiques « agréable » et « désagréable » est déjà mis en apprentissage dans la classification.

Ce qui est intéressant, c'est l'installation d'Audeep et la manipulation s'est fait sur Steam Deck avec une machine virtuelle Boxer + Fedora. 
