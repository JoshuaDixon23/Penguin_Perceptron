# Penguin Perceptron

Ce projet étudie un perceptron monocouche pour classer les manchots Adélie,
Papou et à jugulaire à partir de quatre mesures morphologiques.

Le notebook est rédigé en français. Le README principal est disponible en
[anglais](./README.md).

## Installation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Exécution

Ouvrir [`penguin_perceptron_analysis.ipynb`](./penguin_perceptron_analysis.ipynb)
dans Jupyter ou VS Code, puis utiliser **Restart & Run All**.

Les dépendances sont listées dans [`requirements.txt`](./requirements.txt).
Le notebook utilise la graine `42` pour rendre reproductibles la séparation
des données, la génération du bruit et l'initialisation du perceptron.

## Chargement des données

Le chargement est réalisé par `src.data.load_data`. Cette fonction accepte un
chemin de fichier local ou une URL. L'import du package ne télécharge donc
aucune donnée automatiquement, ce qui permet aussi un import hors ligne.

## Notes d'implémentation

- Le perceptron ne possède pas de terme de biais explicite.
- Les variables sont normalisées avant l'apprentissage et la validation.
- Les seuils de décision sont déterminés sur l'ensemble de validation, puis
  appliqués à l'ensemble de test.
- Les métriques ROC et précision-rappel sont implémentées à des fins
  pédagogiques.
- L'AUC et l'AP utilisent une intégration trapézoïdale. Le code utilise
  `numpy.trapezoid` lorsqu'il est disponible et `numpy.trapz` avec les anciennes
  versions de NumPy.

## Structure du projet

```text
penguin_perceptron_analysis.ipynb
src/
  data.py
  metrics.py
  perceptron.py
  plots.py
  validation.py
requirements.txt
README.md
```
