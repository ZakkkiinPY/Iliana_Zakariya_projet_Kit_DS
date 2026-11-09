"""
Lit les tables du dump et leur donne leurs vrais types.

`donnees.py` écrit toutes les colonnes en texte. Ici, chaque colonne reçoit son type d'après son nom :
identifiants, années et compteurs en entiers, dates de mise à jour en dates, `t` / `f` en booléens.

"""

from pathlib import Path

import pandas as pd

DOSSIER_PARQUET = Path(__file__).resolve().parent.parent / "data" / "parquet"

# Colonnes de texte. Toutes les colonnes absentes des trois listes sont des entiers.
TEXTES = {
    "gid", "name", "sort_name", "comment", "description", "barcode", "catalog_number", "code",
    "join_phrase", "entity_type0", "entity_type1", "link_phrase", "reverse_link_phrase",
    "long_link_phrase", "entity0_credit", "entity1_credit", "url", "time", "setlist",
}
DATES = {"last_updated", "created"}
BOOLEENS = {"ended", "is_deprecated", "has_dates", "has_discids", "cancelled"}


def en_entiers(colonne):
    """Texte vers entier. Une valeur manquante reste manquante (type Int64)."""
    return colonne.astype("Int64")


def en_dates(colonne):
    """Horodatage du dump (`2012-05-15 18:57:13.252186+00`) vers sa date, sans l'heure."""
    return pd.to_datetime(colonne.str[:10], format="%Y-%m-%d")


def en_booleens(colonne):
    return colonne.map({"t": True, "f": False}).astype("boolean")


def typer(table):
    for colonne in table.columns:
        if colonne in DATES:
            table[colonne] = en_dates(table[colonne])
        elif colonne in BOOLEENS:
            table[colonne] = en_booleens(table[colonne])
        elif colonne not in TEXTES:
            table[colonne] = en_entiers(table[colonne])
    return table


def lire(table, colonnes=None):
    return typer(pd.read_parquet(DOSSIER_PARQUET / f"{table}.parquet", columns=colonnes))
