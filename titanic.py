"""
Prediction de la survie d'un individu sur le Titanic
"""

import os
from dotenv import load_dotenv
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix

MAX_DEPTH = None
MAX_FEATURES = "sqrt"

load_dotenv()

# ENVIRONMENT CONFIGURATION ---------------------------

parser = argparse.ArgumentParser(description="Paramètres du random forest")
parser.add_argument(
    "--n_trees", type=int, default=20, help="Nombre d'arbres"
)
parser.add_argument("--max_depth", type=int, default=None, help="Profondeur max (default: None)")
parser.add_argument("--max_features", type=str, default="sqrt", help="Features max (default: 'sqrt')")
args = parser.parse_args()

n_trees = args.n_trees
jeton_api = os.environ.get("JETON_API", "")

if jeton_api.startswith("$"):
    print("API token has been configured properly")
else:
    print("API token has not been configured")


# IMPORT ET EXPLORATION DONNEES --------------------------------

TrainingData = pd.read_csv("data.csv")

TrainingData.isnull().sum()

# Statut socioéconomique
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig1_pclass = sns.countplot(data=TrainingData, x="Pclass", ax=axes[0]).set_title(
    "fréquence des Pclass"
)
fig2_pclass = sns.barplot(
    data=TrainingData, x="Pclass", y="Survived", ax=axes[1]
).set_title("survie des Pclass")

# Age
sns.histplot(data=TrainingData, x="Age", bins=15, kde=False).set_title(
    "Distribution de l'âge"
)
plt.show()


# SPLIT TRAIN/TEST --------------------------------

# On _split_ notre _dataset_ d'apprentisage
# Prenons arbitrairement 10% du dataset en test et 90% pour l'apprentissage.

y = TrainingData["Survived"]
X = TrainingData.drop("Survived", axis="columns")

X_train, X_TEST, y_train, y_test = train_test_split(X, y, test_size=0.1)
pd.concat([X_train, y_train], axis = 1).to_csv("train.csv")
pd.concat([X_TEST, y_test], axis = 1).to_csv("test.csv")


# PIPELINE ----------------------------


# Définition des variables
numeric_features = ["Age", "Fare"]
categorical_features = ["Embarked", "Sex"]


def get_pipeline(n_trees, numeric_features, categorical_features, max_depth=None, max_features="sqrt"):
    """
    Crée un pipeline complet intégrant preprocessing et modèle.
    """
    # 1. Variables numériques
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
    ])

    # 2. Variables catégorielles
    categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder()),
    ]
)

    # 3. Preprocessing (ColumnTransformer)
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    # 4. Pipeline final (Preprocessing + Classifier)
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=max_depth,
            max_features=max_features
        )),
    ])
    
    return pipe


# ESTIMATION ET EVALUATION ----------------------
pipe = get_pipeline(
        n_trees=args.n_trees,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        max_depth=args.max_depth,
        max_features=args.max_features
    )

pipe.fit(X_train, y_train)

# score
rdmf_score = pipe.score(X_TEST, y_test)
print(f"{rdmf_score:.1%} de bonnes réponses sur les données de test pour validation")

print(20 * "-")

print(20 * "-")
print("matrice de confusion")
print(confusion_matrix(y_test, pipe.predict(X_TEST)))
