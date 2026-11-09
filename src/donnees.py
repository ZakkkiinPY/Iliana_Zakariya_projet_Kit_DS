"""
Télécharge le dump MusicBrainz et extrait les tables utiles en parquet.

Lancer depuis la racine du projet : `python -m src.donnees`.

"""

import hashlib
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq
import requests
from tqdm import tqdm

SERVEUR = "https://data.metabrainz.org/pub/musicbrainz/data/fullexport/"
VERSION_DUMP = "20261104-002222"  # version des résultats du projet
DOSSIER_ARCHIVES = Path("data/archives")
DOSSIER_PARQUET = Path("data/parquet")

# Empreintes SHA256 officielles de VERSION_DUMP, copiées de son fichier SHA256SUMS.
EMPREINTES = {
    "mbdump.tar.bz2": "fa3ed54bb6371c1737959f3631c8e5001a79e72c2376433697950c1185470cf1",
    "mbdump-derived.tar.bz2": "313cdedc7f03e704a2cbbac31167bb8035d6fb9ab813c13d83c608d853f619b9",
}

# Colonnes de chaque table, dans l'ordre de admin/sql/CreateTables.sql (schéma 31).
COLONNES = {
    "area": [
        "id", "gid", "name", "type", "edits_pending", "last_updated",
        "begin_date_year", "begin_date_month", "begin_date_day",
        "end_date_year", "end_date_month", "end_date_day", "ended", "comment",
    ],
    "artist": [
        "id", "gid", "name", "sort_name",
        "begin_date_year", "begin_date_month", "begin_date_day",
        "end_date_year", "end_date_month", "end_date_day",
        "type", "area", "gender", "comment", "edits_pending", "last_updated",
        "ended", "begin_area", "end_area",
    ],
    "artist_credit_name": ["artist_credit", "position", "artist", "name", "join_phrase"],
    "artist_type": ["id", "name", "parent", "child_order", "description", "gid"],
    "event": [  # concerts et festivals (couverture, notebook 06)
        "id", "gid", "name",
        "begin_date_year", "begin_date_month", "begin_date_day",
        "end_date_year", "end_date_month", "end_date_day",
        "time", "type", "cancelled", "setlist", "comment", "edits_pending", "last_updated", "ended",
    ],
    "gender": ["id", "name", "parent", "child_order", "description", "gid"],
    "iso_3166_1": ["area", "code"],  # code pays des zones (profil, marché)
    "l_area_area": [  # ville ou région vers son pays (profil, marché)
        "id", "link", "entity0", "entity1", "edits_pending", "last_updated",
        "link_order", "entity0_credit", "entity1_credit",
    ],
    "l_artist_artist": [  # membres des groupes (réseau)
        "id", "link", "entity0", "entity1", "edits_pending", "last_updated",
        "link_order", "entity0_credit", "entity1_credit",
    ],
    "l_artist_event": [  # artistes des concerts et festivals (couverture, notebook 06)
        "id", "link", "entity0", "entity1", "edits_pending", "last_updated",
        "link_order", "entity0_credit", "entity1_credit",
    ],
    "l_artist_label": [
        "id", "link", "entity0", "entity1", "edits_pending", "last_updated",
        "link_order", "entity0_credit", "entity1_credit",
    ],
    "l_artist_url": [  # liens vers les sites de l'artiste (web)
        "id", "link", "entity0", "entity1", "edits_pending", "last_updated",
        "link_order", "entity0_credit", "entity1_credit",
    ],
    "l_label_label": [
        "id", "link", "entity0", "entity1", "edits_pending", "last_updated",
        "link_order", "entity0_credit", "entity1_credit",
    ],
    "label": [
        "id", "gid", "name",
        "begin_date_year", "begin_date_month", "begin_date_day",
        "end_date_year", "end_date_month", "end_date_day",
        "label_code", "type", "area", "comment", "edits_pending", "last_updated", "ended",
    ],
    "label_type": ["id", "name", "parent", "child_order", "description", "gid"],
    "link": [
        "id", "link_type",
        "begin_date_year", "begin_date_month", "begin_date_day",
        "end_date_year", "end_date_month", "end_date_day",
        "attribute_count", "created", "ended",
    ],
    "link_type": [
        "id", "parent", "child_order", "gid", "entity_type0", "entity_type1",
        "name", "description", "link_phrase", "reverse_link_phrase", "long_link_phrase",
        "last_updated", "is_deprecated", "has_dates",
        "entity0_cardinality", "entity1_cardinality",
    ],
    "medium": [  # formats physiques (traction)
        "id", "release", "position", "format", "name", "edits_pending", "last_updated",
        "track_count", "gid",
    ],
    "medium_format": ["id", "name", "parent", "child_order", "year", "has_discids", "description", "gid"],
    "release": [
        "id", "gid", "name", "artist_credit", "release_group", "status", "packaging",
        "language", "script", "barcode", "comment", "edits_pending", "quality",
        "last_updated",
    ],
    "release_country": ["release", "country", "date_year", "date_month", "date_day"],
    "release_group": [
        "id", "gid", "name", "artist_credit", "type", "comment", "edits_pending",
        "last_updated",
    ],
    "release_group_meta": [
        "id", "release_count",
        "first_release_date_year", "first_release_date_month", "first_release_date_day",
        "rating", "rating_count",
    ],
    "release_group_primary_type": ["id", "name", "parent", "child_order", "description", "gid"],
    "release_group_secondary_type": ["id", "name", "parent", "child_order", "description", "gid"],
    "release_group_secondary_type_join": ["release_group", "secondary_type", "created"],
    "release_label": ["id", "release", "label", "catalog_number", "last_updated"],
    "release_status": ["id", "name", "parent", "child_order", "description", "gid"],
    "release_unknown_country": ["release", "date_year", "date_month", "date_day"],
    "url": ["id", "gid", "url", "edits_pending", "last_updated"],  # adresses des liens (web)
}


def empreinte(nom):
    with open(DOSSIER_ARCHIVES / nom, "rb") as fichier:
        return hashlib.file_digest(fichier, "sha256").hexdigest()


def choisir_version():
    """Version du dump et empreintes de ses archives.

    Le serveur ne garde que les dernières versions. On prend VERSION_DUMP tant qu'il l'a,
    ou que ses archives sont déjà sur le disque. Sinon, on prend la dernière version (fichier LATEST)
    avec les empreintes de son fichier SHA256SUMS.
    """
    if requests.head(f"{SERVEUR}{VERSION_DUMP}/", timeout=60).ok:
        return VERSION_DUMP, EMPREINTES
    # Archives de VERSION_DUMP déjà sur le disque : on les garde, `verifier()` les supprimerait avec une autre version.
    if all((DOSSIER_ARCHIVES / nom).exists() and empreinte(nom) == valeur for nom, valeur in EMPREINTES.items()):
        return VERSION_DUMP, EMPREINTES

    reponse = requests.get(f"{SERVEUR}LATEST", timeout=60)
    reponse.raise_for_status()
    version = reponse.text.strip()
    reponse = requests.get(f"{SERVEUR}{version}/SHA256SUMS", timeout=60)
    reponse.raise_for_status()
    empreintes = {}
    for ligne in reponse.text.splitlines():
        valeur, nom = ligne.split(" *")  # une ligne par archive : « empreinte *nom »
        if nom in EMPREINTES:
            empreintes[nom] = valeur
    return version, empreintes


def telecharger(version, nom):
    reponse = requests.get(f"{SERVEUR}{version}/{nom}", stream=True, timeout=60)
    reponse.raise_for_status()
    barre = tqdm(total=int(reponse.headers["Content-Length"]), unit="B", unit_scale=True, desc=nom)
    with open(DOSSIER_ARCHIVES / nom, "wb") as fichier:
        for morceau in reponse.iter_content(chunk_size=1024 * 1024):
            fichier.write(morceau)
            barre.update(len(morceau))
    barre.close()


def verifier(nom, attendue):
    """Supprime l'archive et arrête tout si son SHA256 n'est pas l'empreinte `attendue`."""
    if empreinte(nom) != attendue:
        (DOSSIER_ARCHIVES / nom).unlink()
        raise SystemExit(f"{nom} est corrompu : fichier supprimé, relance le script.")
    print(f"{nom} : vérifié")


def lire_table(fichier, table):
    """Lit une table TSV du dump : colonnes en texte, `\\N` devient une valeur manquante."""
    return pv.read_csv(
        fichier,
        read_options=pv.ReadOptions(column_names=COLONNES[table]),
        parse_options=pv.ParseOptions(delimiter="\t", quote_char=False),
        convert_options=pv.ConvertOptions(
            column_types={colonne: pa.string() for colonne in COLONNES[table]},
            null_values=["\\N"],
            strings_can_be_null=True,
        ),
    )


def extraire(nom):
    """Écrit en parquet les tables de COLONNES trouvées dans l'archive, en une seule lecture.

    Une table déjà extraite n'est pas réécrite.
    """
    extraites = []
    with tarfile.open(DOSSIER_ARCHIVES / nom, mode="r|bz2") as archive:
        for membre in archive:
            table = membre.name.replace("mbdump/", "")
            if table in COLONNES and not (DOSSIER_PARQUET / f"{table}.parquet").exists():
                donnees = lire_table(archive.extractfile(membre), table)
                pq.write_table(donnees, DOSSIER_PARQUET / f"{table}.parquet")
                print(f"{table} : {donnees.num_rows} lignes")
                extraites.append(table)
    return extraites


if __name__ == "__main__":
    DOSSIER_ARCHIVES.mkdir(parents=True, exist_ok=True)
    DOSSIER_PARQUET.mkdir(parents=True, exist_ok=True)
    version, empreintes = choisir_version()
    print(f"Version du dump : {version}")
    if version != VERSION_DUMP:
        print(f"Le serveur n'a plus {VERSION_DUMP}, la version des résultats du projet.")
        print("Les chiffres peuvent un peu changer.")
    extraites = []
    for nom in EMPREINTES:
        if not (DOSSIER_ARCHIVES / nom).exists():
            telecharger(version, nom)
        verifier(nom, empreintes[nom])
        extraites += extraire(nom)
    presentes = len(list(DOSSIER_PARQUET.glob("*.parquet")))
    print(f"{len(extraites)} tables extraites, {presentes} présentes sur {len(COLONNES)}")
