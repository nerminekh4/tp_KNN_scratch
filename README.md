# TP — KNN from scratch

Implémentation **from scratch** (vanilla Python uniquement) d'un classifieur
K-Nearest Neighbors, dans le cadre du premier FOAD.

## Contraintes

- Aucune bibliothèque tierce (numpy, sklearn, pandas...) n'est utilisée
  dans le code de la classe `KNN`.
- Seul le module `data_loader.py` (fourni par l'enseignant) est autorisé
  pour le chargement et la normalisation des données.
- Le code est orienté objet et suit les conventions vues en cours.

## Structure du projet

| Fichier            | Rôle                                               |
| ------------------ | -------------------------------------------------- |
| `data_loader.py`   | Module fourni : chargement + normalisation         |
| `knn.py`           | Classe `KNN` (fit / predict / evaluate / grid)     |
| `utils.py`         | `train_test_split` from scratch                    |
| `main.py`          | Pipeline complet (load → split → train → evaluate) |
| `bienetre.csv`     | Jeu de données                                     |
| `requirements.txt` | Dépendances de `data_loader.py`                    |

## Lancer le projet

```bash
# 1. Créer et activer le venv
python3 -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows

# 2. Installer les dépendances de data_loader
pip install -r requirements.txt

# 3. Lancer le pipeline
python main.py
```

## La classe `KNN`

```python
from knn import KNN

model = KNN(k=5)
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
accuracy = model.evaluate(X_test, Y_test)

result = KNN.grid_search(X_train, Y_train, X_val, Y_val,
                        k_values=[1, 3, 5, 7, 9])
print(result["best_k"], result["best_score"])
```

## Méthodes principales

- `fit(X, Y)` — mémorise les données d'entraînement (KNN est lazy).
- `predict(X)` — vote majoritaire parmi les k plus proches voisins.
- `evaluate(X, Y)` — accuracy sur un jeu de test.
- `confusion_matrix(X, Y)` — TP / TN / FP / FN.
- `grid_search(...)` — méthode de classe pour trouver le meilleur k.