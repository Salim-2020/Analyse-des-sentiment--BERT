# Analyse-des-sentiment--BERT
Etape d'implementation de BERT

Les modèles pré-entraînés (BERT, Cammembert) peut affiné sur des ensembles de données plus petits pour effectuer une analyse des sentiments. ou les taches de classifications du texts.

Préparation des données.

1- Importer La bibliothèque Transformers de Hugging Face fournit tous les modèles(Comme : BERT, Cammembert, RoBERTa, etc.) à utiliser avec TF 2.0

2- Convertir les données dans le format attendu par BERT sur Huggin Face, en séparant les deux ensembles train et test en deux fichier.CSV ou TSV. Telle ques: 

      • Colonne 0 : Un identifiant pour la ligne
      • Colonne 1 : Les émail
      • Colonne 2 : Les labels

3- Charger le modele pré-entraînés 

4- Créant un dossier «bert-output» où le modèle affiné sera sauvegardé

5- Reglages les Hyper-paramètres

6- Pour ajouter une régression suffit mettre ::  régression == True dans les Hyper-paramètres.

7- Exécution, de préférence sur un gpu.

8- Métrique : MSE, MAE, RMSE, Précision.

9- Prédiction.
