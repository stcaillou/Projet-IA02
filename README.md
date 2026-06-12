# README - Projet IA02 : Classification d'images et reconnaissance vocale

**Auteurs** : Daria EZHOVA, Nina COIFFIN, Pierre RIBET<br>
**Date** : Juin 2026<br>
**Contexte** : Projet en intelligence artificielle (IA02) - Classification d'images (CIFAR-10) et *keyword spotting* (Speech Commands).


## **Objectifs du projet**
1. **Partie I : Classification d'images (CIFAR-10)**
   - Comparer des architectures de **machine learning classique**
   - Comparer des architectures de **CNN** (simple, profond, VGGSmall, MiniResNet,...).
   - Réaliser un **CNN Hybride**

3. **Partie II : Reconnaissance vocale (Speech Commands)**
   - Classifier des mots-clés à partir de **spectrogrammes** (MelSpectrogram).
   - Tester des **architectures** (CNN, CNNBiLSTM, CNN1D2D).
   - Réaliser un **augmentation de données** et en mesurer l'impact.
   - Implémenter un **test en temps réel** avec microphone.



## **Structure du projet**
```bash
IA02/
├── cifar-10-batches-py/          # Dataset CIFAR-10 (images)
├── SpeechCommands/               # Dataset Speech Commands (audio)
├── LogsOptuna/                   # Logs des optimisations Optuna
├── LogsTrain/                    # Logs d'entraînement des modèles
├── Modeles/                      # Code des architectures de modèles
├── PoidsModeles/                 # Poids sauvegardés des modèles entraînés
├── 'Logs - Augmentation'/        # Logs avec augmentation de données
├── 'Poids - Augmentation'/       # Poids des modèles avec augmentation
├── .venv/                        # Environnement virtuel Python
│
├── *.ipynb                       # Notebooks d'exploration et d'analyse
│   ├── ExplorationDonnées.ipynb # Analyse exploratoire des datasets
│   └── PartieI.ipynb             # Expérimentations sur CIFAR-10
│
├── *.py                         # Scripts utilitaires
│   ├── data_aug.py               # Augmentation de données (CIFAR-10/Speech Commands)
│   ├── direct_test.py            # Test en direct avec microphone
│   ├── optuna-training.py        # Optimisation des hyperparamètres (Optuna)
│   ├── test_CNNModule.py         # Tests unitaires pour les CNN
│   └── training_CNNModule.py     # Script d'entraînement des CNN
│
├── Rapport_IA02.pdf            # Rapport détaillé du projet
└── .gitignore                    # Fichiers exclus du versionnage Git
```
