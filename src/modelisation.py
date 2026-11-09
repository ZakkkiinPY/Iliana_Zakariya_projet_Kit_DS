"""
Prépare la table des variables pour scikit-learn, construit les modèles et mesure leurs classements.

"""

import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, QuantileTransformer, StandardScaler

# Variables de chaque famille de src/transformation.py (FAMILLES), dans le même ordre.
COLONNES_PAR_FAMILLE = {
    "activité": ["albums", "eps", "singles", "disques_avant", "anciennete", "annees_actives", "recence", "tendance"],
    "traction": ["editions_par_disque", "pays_de_sortie", "part_physique"],
    "documentation": ["delai_median", "part_ajout_meme_annee"],
    "label": ["part_label_connu", "labels", "label_tremplin"],
    "proximité": ["label_distribue_major", "distributeur_major", "edition_major_isolee", "invite_major"],
    "réseau": ["collaborateurs", "collaborateurs_major", "lies", "lies_major", "part_collaborateurs_major"],
    "profil": ["type", "pays", "type_manquant", "pays_manquant"],
    "marché": ["taux_pays"],
}
# Famille ajoutée par le notebook 06, dans data/features_web.parquet seulement.
WEB = {"web": ["liens_url", "site_officiel", "reseaux_sociaux", "streaming", "bandcamp", "bases"]}
CATEGORIES = ["type", "pays"]
# Sélection des colonnes par type, pour que les modèles marchent aussi sans une famille (ablation).
NOMBRES = make_column_selector(dtype_exclude="category")
CATEGORIELLES = make_column_selector(dtype_include="category")

# Réglages choisis sur la validation du notebook 05 (parties 5 à 7), repris par le notebook 06.
LOGISTIQUE = dict(quantiles=True, C=0.1)
FORET = dict(n_estimators=300, min_samples_leaf=50)
BOOSTING = dict(learning_rate=0.05, max_iter=111, max_leaf_nodes=7, min_samples_leaf=300)


def preparer(table, familles=COLONNES_PAR_FAMILLE):
    """Variables X des `familles` (nombres en float avec NaN, type et pays en catégories) et cible y."""
    colonnes = [c for variables in familles.values() for c in variables]
    X = table[colonnes].copy()
    for colonne in colonnes:
        X[colonne] = X[colonne].astype("category") if colonne in CATEGORIES else X[colonne].astype(float)
    return X, table["y"].to_numpy()


def auc(x, y):
    """AUC ROC d'une variable seule, sur les lignes où elle est renseignée (0,5 si elle ne sert à rien)."""
    ok = x.notna()
    return roc_auc_score(y[ok], x[ok].astype(float))


def mesures(y, score, ks=(100, 500)):
    """AUC précision-rappel, AUC ROC, et précision et rappel parmi les k artistes les mieux classés."""
    resultat = {"AUC PR": average_precision_score(y, score), "AUC ROC": roc_auc_score(y, score)}
    ordre = np.argsort(-np.asarray(score), kind="stable")
    for k in ks:
        trouves = y[ordre[:k]].sum()
        resultat[f"précision@{k}"] = trouves / k
        resultat[f"rappel@{k}"] = trouves / y.sum()
    return resultat


def intervalle_ecart(y, score_a, score_b, tirages=200, graine=0):
    """Intervalle à 95 % de AUC PR(a) - AUC PR(b), en retirant les artistes avec remise (bootstrap)."""
    hasard = np.random.default_rng(graine)
    ecarts = []
    for _ in range(tirages):
        i = hasard.integers(0, len(y), len(y))
        ecarts.append(average_precision_score(y[i], score_a[i]) - average_precision_score(y[i], score_b[i]))
    return np.percentile(ecarts, [2.5, 97.5])


def logistique(quantiles=False, C=1.0, l1_ratio=0.0, class_weight=None):
    """Régression logistique : médiane et indicateur pour les manquants, z-score (ou rangs ramenés à une loi
    normale avec quantiles=True), one hot des catégories (les valeurs vues moins de 500 fois sont regroupées)."""
    echelle = QuantileTransformer(output_distribution="normal", random_state=0) if quantiles else StandardScaler()
    preparation = ColumnTransformer([
        ("nombres", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), echelle), NOMBRES),
        ("categories", OneHotEncoder(min_frequency=500, handle_unknown="infrequent_if_exist"), CATEGORIELLES),
    ])
    solveur = "lbfgs" if l1_ratio == 0 else "liblinear"
    modele = LogisticRegression(
        C=C, l1_ratio=l1_ratio, class_weight=class_weight, solver=solveur, max_iter=2000, random_state=0,
    )
    return make_pipeline(preparation, modele)


def foret(**reglages):
    """Forêt aléatoire : les catégories en codes entiers (label encoding), les manquants laissés en NaN."""
    codes = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, encoded_missing_value=np.nan)
    preparation = ColumnTransformer([("categories", codes, CATEGORIELLES)], remainder="passthrough")
    return make_pipeline(preparation, RandomForestClassifier(n_jobs=-1, random_state=0, **reglages))


def boosting(**reglages):
    return HistGradientBoostingClassifier(early_stopping=False, random_state=0, **reglages)
