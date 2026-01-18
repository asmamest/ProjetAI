# RSNA Breast Density Classification Project

Ce projet vise à automatiser la classification de la densité mammaire (scores BI-RADS A, B, C, D) à partir d'images mammographiques en utilisant des architectures de deep learning de pointe.

## 1. Pipeline de Pré-traitement des Images

Pour préparer les données à l'entraînement, nous avons transformé les fichiers médicaux bruts (**DICOM**) en images exploitables (**PNG**) via les étapes suivantes :

*   **Lecture DICOM** : Extraction des données de pixels et des métadonnées (vues LCC, LMLO, RCC, RMLO).
*   **Windowing & Normalisation** : Ajustement du contraste pour mettre en évidence les tissus denses et normalisation des intensités (0-255).
*   **Cropping** : Suppression des bordures noires inutiles pour se concentrer sur le sein.
*   **Mise au format Carré (Padding)** : Ajout de bordures pour conserver l'aspect original sans déformé l'image.
*   **Redimensionnement** : Finalisation en 512x512 pixels pour une qualité optimale avant l'entraînement.

## 2. Modèles de Classification

Nous avons exploré deux architectures majeures pour comparer leur efficacité sur les images mammographiques :

### 2.1. DeiT (Data-efficient Image Transformer)
Utilisation du modèle **DeiT-Small** (`deit_small_patch16_224`).
*   **Fonctionnement** : Découpe l'image en patches de 16x16 et utilise des mécanismes d'**Attention** pour capturer les relations globales entre les tissus.
*   **Avantage** : Très performant pour saisir le contexte global d'un sein.

### 2.2. DenseNet121 (Convolutional Neural Network)
Utilisation de l'architecture **DenseNet121**.
*   **Fonctionnement** : Contrairement aux Transformers, DenseNet utilise des convolutions. Sa particularité est que chaque couche reçoit les caractéristiques (features) de **toutes les couches précédentes** (Dense Blocks).
*   **Pourquoi DenseNet ?** : Cette architecture excelle dans la réutilisation des caractéristiques et l'apprentissage de motifs fins (comme les textures de densité mammaire). Elle réduit le risque de disparition du gradient et est souvent plus stable sur des datasets médicaux de taille moyenne.
*   **Optimisation Kaggle** : Le script inclut des augmentations robustes (CLAHE, rotations à 90°, flips) pour maximiser la robustesse du modèle.

## 3. Conception de l'Entraînement (Training Design)

Le modèle a été conçu pour maximiser l'accuracy tout en respectant des contraintes de temps sur une machine locale (GTX 1650) :

*   **Répartition des données** : 70% pour l'apprentissage, 30% pour le test final.
*   **Équilibrage des classes** : Utilisation de **Class Weights** pour s'assurer que les classes rares (comme la densité A ou D) soient aussi bien apprises que les classes fréquentes.
*   **Robustesse** : 
    *   **Label Smoothing** : Empêche l'IA d'être "trop confiante" et améliore sa capacité de généralisation.
    *   **Data Augmentation** : Rotations aléatoires, zooms et ajustements de contraste pendant l'apprentissage pour rendre le modèle plus intelligent.
*   **Optimisation Flash** : Utilisation de **Mixed Precision (AMP)** et du modèle **Small** pour réduire le temps d'entraînement tout en augmentant la puissance de calcul.

## 4. Organisation du Projet

Le projet est structuré par famille de modèles pour faciliter les comparaisons :

*   📂 **DeiT/** : Contient les scripts de pré-traitement et d'entraînement pour les modèles Transformers.
*   📂 **DenseNet121/** : Contient l'implémentation spécifique du modèle DenseNet et ses paramètres optimisés.
*   📂 **processed_images/** : Images PNG prêtes à l'emploi (Issues du DICOM).

## 5. Résultats Obtenus
*   **DeiT-Small** : **76.15%** de précision par patient.
*   **DenseNet121** : **75.84%** de précision par patient.

---
*Projet développé dans le cadre de l'analyse RSNA Breast Density.*
