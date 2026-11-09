"""
Construit la cible (labels des majors, sorties retenues, cohortes d'artistes), puis les variables.

Lancer depuis la racine du projet, après src/donnees.py : `python -m src.transformation`.
Écrit alors la table des modèles, data/features.parquet.

"""

from functools import cache
from math import inf
from pathlib import Path

import pandas as pd

from .nettoyage import lire

# Holdings des trois majors (id de label).
HOLDINGS = {
    "Universal": [151],    # Universal Music Group
    "Sony": [87262, 366],  # Sony Music Entertainment, Sony Music Entertainment (Japan)
    "Warner": [29],        # Warner Music Group Corp.
}

# Types de liens suivis depuis les holdings, du plus strict au plus large.
VARIANTES = {
    "ownership": ["label ownership"],
    "+ imprint": ["label ownership", "imprint"],
    "+ rename": ["label ownership", "imprint", "label rename"],
    "+ distribution": ["label ownership", "imprint", "label rename", "label distribution"],
    "+ reissue": ["label ownership", "imprint", "label rename", "label distribution", "label reissue"],
}

NO_LABEL = "157afde4-4bf5-4039-8ad2-5a15acc85176"  # gid de [no label]
# gid de Various Artists et de [unknown]
SPECIAUX = ["89ad4ac3-39f7-470e-963a-56509c546377", "125ec42a-7229-4250-afc5-e057484327fe"]

W, H = 3, 3    # années de passé observé et de fenêtre de cible (notebook 01)
PASSEES = 5    # cohortes passées pour les taux de passage en major (notebook 03)
LISSAGE = 100  # poids du taux moyen dans les taux lissés (notebook 03)

COHORTES = [2014, 2015, 2016, 2017, 2018, 2021]  # entraînement 2014 à 2018, test 2021 (notebook 03)
ECARTEES = ["sexe", "sexe_manquant", "age", "age_manquant"]  # en partie complétés après coup (notebook 03, partie 7)
FICHIER_VARIABLES = Path(__file__).resolve().parent.parent / "data" / "features.parquet"


def liens_labels():
    types = lire("link_type", ["id", "name", "entity_type0", "entity_type1", "long_link_phrase"])
    types = types[(types["entity_type0"] == "label") & (types["entity_type1"] == "label")]
    types = types.rename(columns={"id": "link_type", "name": "type_lien"})
    link = lire("link", ["id", "link_type", "begin_date_year", "end_date_year"]).rename(columns={"id": "link"})
    liens = lire("l_label_label", ["link", "entity0", "entity1"])
    return liens.merge(link, on="link").merge(types, on="link_type")


def labels_sous(liens, depart, types):
    """Labels atteints depuis `depart` en suivant les liens `types`, avec leur profondeur."""
    choisis = liens[liens["type_lien"].isin(types)]
    # Un renommage relie deux noms du même label : on le suit dans les deux sens.
    renommages = choisis[choisis["type_lien"] == "label rename"]
    inverses = renommages.rename(columns={"entity0": "entity1", "entity1": "entity0"})
    suite = pd.concat([choisis, inverses]).groupby("entity0")["entity1"].apply(list).to_dict()

    profondeur = {lab: 0 for lab in depart}
    a_voir = list(depart)
    while a_voir:
        prochains = []
        for lab in a_voir:
            for autre in suite.get(lab, []):
                if autre not in profondeur:
                    profondeur[autre] = profondeur[lab] + 1
                    prochains.append(autre)
        a_voir = prochains
    return profondeur


def annees_de_renommage(liens):
    renommages = liens[liens["type_lien"] == "label rename"]
    annees = {}
    for cote in ["entity0", "entity1"]:
        for date in ["begin_date_year", "end_date_year"]:
            connues = renommages[[cote, date]].dropna()
            for label, annee in zip(connues[cote], connues[date]):
                annees.setdefault(label, set()).add(annee)
    return annees


def periodes_sous(liens, depart, types):
    """Labels atteints depuis `depart` en suivant les liens `types`, avec les années où le chemin tient (debut, fin).

    Un chemin tient de la plus tardive des dates de début de ses liens à la plus précoce de leurs dates de fin.
    Une date inconnue ne limite rien. Une fin à un an près d'un renommage d'un des deux labels non plus :
    le lien s'arrête alors parce qu'un label change de nom, pas parce qu'il est vendu (notebook 02).
    """
    renomme = annees_de_renommage(liens)
    choisis = liens[liens["type_lien"].isin(types)]
    renommages = choisis[choisis["type_lien"] == "label rename"]
    choisis = pd.concat([choisis, renommages.rename(columns={"entity0": "entity1", "entity1": "entity0"})])
    suite = {}
    for lab, autre, debut, fin in zip(
        choisis["entity0"], choisis["entity1"], choisis["begin_date_year"], choisis["end_date_year"]
    ):
        proches = renomme.get(lab, set()) | renomme.get(autre, set())
        if pd.isna(fin) or any(abs(fin - annee) <= 1 for annee in proches):
            fin = inf
        suite.setdefault(lab, []).append((autre, -inf if pd.isna(debut) else debut, fin))

    vus = {(lab, -inf, inf) for lab in depart}
    a_voir = list(vus)
    while a_voir:
        prochains = []
        for lab, debut, fin in a_voir:
            for autre, debut_lien, fin_lien in suite.get(lab, []):
                chemin = (autre, max(debut, debut_lien), min(fin, fin_lien))
                if chemin[1] <= chemin[2] and chemin not in vus:
                    vus.add(chemin)
                    prochains.append(chemin)
        a_voir = prochains
    return pd.DataFrame(sorted(vus), columns=["label", "debut", "fin"])


def departs(liens):
    """Holdings de chaque major, avec leurs anciens noms."""
    return {major: list(labels_sous(liens, ids, ["label rename"])) for major, ids in HOLDINGS.items()}


def labels_majors(liens, types):
    lignes = []
    for major, ids in departs(liens).items():
        for label in labels_sous(liens, ids, types):
            lignes.append({"label": label, "major": major})
    return pd.DataFrame(lignes)


def labels_majors_dates(liens, types):
    morceaux = [periodes_sous(liens, ids, types).assign(major=major) for major, ids in departs(liens).items()]
    return pd.concat(morceaux, ignore_index=True)


def sans_distributeurs(label_major):
    """Retire les labels de type Distributor : ils distribuent des disques signés ailleurs."""
    label = lire("label", ["id", "type"])
    types = lire("label_type", ["id", "name"])
    distributeurs = label.loc[label["type"].isin(types.loc[types["name"] == "Distributor", "id"]), "id"]
    return label_major[~label_major["label"].isin(distributeurs)]


def charger_sorties():
    release = lire("release", ["id", "artist_credit", "release_group", "status"]).rename(columns={"id": "release"})

    # Une date par pays : on garde l'année la plus ancienne.
    dates = pd.concat([
        lire("release_country", ["release", "date_year"]),
        lire("release_unknown_country", ["release", "date_year"]),
    ])
    release["annee"] = release["release"].map(dates.groupby("release")["date_year"].min())

    # Label connu : ni vide, ni [no label].
    label = lire("label", ["id", "gid"])
    no_label = label.loc[label["gid"] == NO_LABEL, "id"].iloc[0]
    release_label = lire("release_label", ["release", "label"])
    connu = release_label["label"].notna() & (release_label["label"] != no_label)
    release["label_connu"] = release["release"].isin(release_label.loc[connu, "release"])

    statut = lire("release_status", ["id", "name"]).set_index("id")["name"]
    release["statut"] = release["status"].map(statut)

    type_disque = lire("release_group", ["id", "type"]).set_index("id")["type"]
    type_principal = lire("release_group_primary_type", ["id", "name"]).set_index("id")["name"]
    release["type_principal"] = release["release_group"].map(type_disque).map(type_principal)

    secondaire = lire("release_group_secondary_type_join", ["release_group"])
    release["a_type_secondaire"] = release["release_group"].isin(secondaire["release_group"])

    # Réédition : sortie après la première édition publiée du disque, sans les promos, bootlegs, pseudo-sorties
    # ni sorties annulées. release_group_meta les compte toutes (notebook 01), on ne s'en sert donc pas.
    publiees = release[~release["statut"].isin(["Promotion", "Bootleg", "Pseudo-Release", "Cancelled"])]
    release["annee_disque"] = release["release_group"].map(publiees.groupby("release_group")["annee"].min())
    release["reedition"] = (release["annee"] > release["annee_disque"]).fillna(False)
    return release


def en_major(paires, majors):
    """Vrai pour chaque ligne (label, annee) de `paires` dont le label est à une major cette année-là.

    `majors` donne les périodes de chaque label (label, debut, fin), comme labels_majors_dates.
    """
    lignes = paires[["label", "annee"]].assign(ligne=range(len(paires)))
    lignes = lignes.merge(majors[["label", "debut", "fin"]], on="label")
    dedans = (lignes["debut"] <= lignes["annee"]) & (lignes["annee"] <= lignes["fin"])
    return pd.Series(range(len(paires))).isin(lignes.loc[dedans.fillna(False), "ligne"]).to_numpy()


def sur_major(release, majors):
    """Vrai pour les sorties qui portent, l'année de leur sortie, un label de `majors`."""
    release_label = lire("release_label", ["release", "label"])
    paires = release[["release", "annee"]].merge(release_label, on="release")
    return release["release"].isin(paires.loc[en_major(paires, majors), "release"])


def garde(release):
    """Sorties qui comptent : Official, Album, EP ou Single sans type secondaire, première édition, datée."""
    return (
        (release["statut"] == "Official")
        & release["type_principal"].isin(["Album", "EP", "Single"])
        & ~release["a_type_secondaire"]
        & ~release["reedition"]
        & release["annee"].notna()
    )


def marquer_majors(sorties, paires, majors, seuil=0.5):
    """Ajoute à `sorties` les colonnes `major` et `major_un_jour`, d'après les labels `paires` (release, annee, label).

    `major` : un label de la sortie est à une major l'année de la sortie, et au moins `seuil` des éditions
    de son disque dans `sorties` le sont aussi. Une édition isolée (licence ou distribution dans un pays)
    ne compte donc pas. `major_un_jour` fait pareil sans regarder les dates, avant comme après un rachat.
    """
    for colonne, periodes in [("major", majors), ("major_un_jour", majors.assign(debut=-inf, fin=inf))]:
        sorties[colonne] = sorties["release"].isin(paires.loc[en_major(paires, periodes), "release"])
        part = sorties.groupby("release_group")[colonne].transform("mean")
        sorties[colonne] = sorties[colonne] & (part >= seuil)
    return sorties


def retenir(release, majors, seuil=0.5):
    """Sorties gardées, avec `major` et `major_un_jour` (marquer_majors) calculés avec toute la base."""
    gardees = release[garde(release)].copy()
    paires = gardees[["release", "annee"]].merge(lire("release_label", ["release", "label"]), on="release")
    return marquer_majors(gardees, paires, majors, seuil)


@cache
def artiste_principal():
    """Artiste en position 0 de chaque crédit, sans Various Artists ni [unknown]."""
    credit = lire("artist_credit_name", ["artist_credit", "position", "artist"])
    principal = credit[credit["position"] == 0][["artist_credit", "artist"]]
    artist = lire("artist", ["id", "gid"])
    speciaux = artist.loc[artist["gid"].isin(SPECIAUX), "id"]
    return principal[~principal["artist"].isin(speciaux)]


@cache
def date_ajout():
    """Date d'ajout estimée de chaque sortie.

    Les id sont croissants : une sortie a été ajoutée avant toutes celles d'id plus grand,
    donc au plus tard à la plus petite de leurs dates de mise à jour.
    """
    ajout = lire("release", ["id", "last_updated"]).rename(columns={"id": "release"})
    ajout = ajout.sort_values("release")
    ajout["date_ajout"] = ajout["last_updated"][::-1].cummin()[::-1]
    return ajout[["release", "date_ajout"]]


def sorties_du_projet(release=None, majors=None, seuil=0.5):
    """Sorties retenues (retenir), avec leur date d'ajout et leur artiste principal.

    Sans argument, celles de la cible du projet (notebook 01) : liens ownership et imprint datés, sans les
    distributeurs. Le notebook 02 passe d'autres `majors` et `seuil` pour comparer.
    """
    release = charger_sorties() if release is None else release
    if majors is None:
        majors = sans_distributeurs(labels_majors_dates(liens_labels(), VARIANTES["+ imprint"]))
    sorties = retenir(release, majors, seuil).merge(date_ajout(), on="release")
    return sorties.merge(artiste_principal(), on="artist_credit")


def premiere_sortie_major(sorties, colonne="major"):
    return sorties[sorties[colonne]].groupby("artist")["annee"].min()


def cohorte(sorties, t0, w=W, h=H, connu_a_t0=True):
    """Artistes actifs les w ans avant t0 et jamais sortis chez un label de major avant t0, même avant son rachat.

    Renvoie pour chacun s'il est positif : première sortie en major entre t0 et t0 + h - 1,
    chez un label qui était à une major l'année de la sortie. Un rachat n'est pas une signature.
    Avec connu_a_t0, le passé ne compte que les sorties déjà dans la base à t0.
    """
    passe = sorties[sorties["annee"].between(t0 - w, t0 - 1)]
    if connu_a_t0:
        passe = passe[passe["date_ajout"].dt.year < t0]
    artistes = passe["artist"].unique()
    deja = premiere_sortie_major(sorties, "major_un_jour").reindex(artistes)
    major = premiere_sortie_major(sorties).reindex(artistes)
    population = major[deja.isna() | (deja >= t0)]
    return population.between(t0, t0 + h - 1).fillna(False).astype(bool)


# Variables. Chaque famille prend la base vue de t0 (vue_de_t0) et t0, et renvoie une ligne par artiste actif
# les W ans avant t0. Tout vient de ce que la base connaissait à t0 : les sorties ajoutées avant t0, les lignes
# des autres tables (labels, supports, liens, liens URL) créées avant t0, et le statut en major recalculé avec elles.


def date_creation(table):
    """Date de création estimée de chaque ligne d'une table, par la règle de date_ajout.

    Les id sont croissants : une ligne existait au plus tard à la plus petite date de mise à jour
    des lignes d'id plus grand. Une ligne modifiée après t0 mais créée avant reste donc connue à t0.
    """
    table = table.sort_values("id")
    table["creation"] = table["last_updated"][::-1].cummin()[::-1]
    return table.drop(columns=["id", "last_updated"])


def connues(sorties, t0):
    """Sorties que la base connaissait à t0 : sorties avant t0 et ajoutées avant t0."""
    return sorties[(sorties["annee"] < t0) & (sorties["date_ajout"].dt.year < t0)]


def recentes(sorties, t0):
    """Sorties connues à t0 des W ans avant t0, le passé observé de la cohorte."""
    passe = connues(sorties, t0)
    return passe[passe["annee"] >= t0 - W]


def activite(sorties, t0):
    """Disques par type sur les W ans, disques plus anciens, ancienneté, années actives, récence, tendance."""
    disques = connues(sorties, t0).drop_duplicates(["artist", "release_group"])
    recents = disques[disques["annee"] >= t0 - W]
    table = pd.crosstab(recents["artist"], recents["type_principal"])
    table = table.reindex(columns=["Album", "EP", "Single"], fill_value=0)
    table.columns = ["albums", "eps", "singles"]
    table["disques_avant"] = disques[disques["annee"] < t0 - W].groupby("artist").size()
    annees = disques.groupby("artist")["annee"]
    table["anciennete"] = t0 - annees.min()
    table["annees_actives"] = annees.nunique()
    table["recence"] = t0 - annees.max()
    derniere = (recents["annee"] == t0 - 1).groupby(recents["artist"]).sum()
    premiere = (recents["annee"] == t0 - W).groupby(recents["artist"]).sum()
    table["tendance"] = derniere - premiere
    return table.fillna({"disques_avant": 0})


@cache
def pays_des_sorties():
    return lire("release_country", ["release", "country"])


@cache
def supports_physiques():
    """Supports physiques de chaque édition (release, creation) : un format connu autre que Digital Media."""
    medium = date_creation(lire("medium", ["id", "release", "format", "last_updated"]))
    formats = lire("medium_format", ["id", "name"])
    numerique = formats.loc[formats["name"] == "Digital Media", "id"]
    return medium[medium["format"].notna() & ~medium["format"].isin(numerique)]


def traction(sorties, t0):
    """Éditions par disque, pays de sortie et part d'éditions sur support physique, sur les W ans."""
    recents = recentes(sorties, t0)
    par_artiste = recents.groupby("artist")
    table = pd.DataFrame({"editions_par_disque": par_artiste.size() / par_artiste["release_group"].nunique()})
    pays = recents[["artist", "release"]].merge(pays_des_sorties(), on="release")
    table["pays_de_sortie"] = pays.groupby("artist")["country"].nunique()
    physiques = supports_physiques()
    physiques = physiques.loc[physiques["creation"].dt.year < t0, "release"]
    table["part_physique"] = recents["release"].isin(physiques).groupby(recents["artist"]).mean()
    return table.fillna({"pays_de_sortie": 0})


def documentation(sorties, t0):
    """Délai entre la sortie et son ajout estimé dans la base, sur les W ans."""
    recents = recentes(sorties, t0)
    delai = recents["date_ajout"].dt.year - recents["annee"]
    return pd.DataFrame({
        "delai_median": delai.groupby(recents["artist"]).median(),
        "part_ajout_meme_annee": (delai <= 0).groupby(recents["artist"]).mean(),
    })


@cache
def labels_des_sorties():
    """Labels de chaque édition (release, label, creation), sans les labels vides."""
    labels = date_creation(lire("release_label", ["id", "release", "label", "last_updated"]))
    return labels.dropna(subset="label").drop_duplicates(["release", "label"])


def labels_connus(recents, t0):
    """Labels des sorties `recents` déjà dans la base à t0 (artist, release, label)."""
    labels = recents[["artist", "release"]].merge(labels_des_sorties(), on="release")
    return labels.loc[labels["creation"].dt.year < t0, ["artist", "release", "label"]]


def cohortes_passees(sorties, t0, n=PASSEES):
    """Artistes et cible des n cohortes dont la fenêtre de cible finit avant t0, vues avec la base de t0."""
    base = connues(sorties, t0)
    morceaux = []
    for t in range(t0 - H - n + 1, t0 - H + 1):
        y = cohorte(base, t)
        morceaux.append(pd.DataFrame({"t0": t, "artist": y.index, "y": y.values}))
    return pd.concat(morceaux, ignore_index=True)


def taux_lisse(table, cle, m=LISSAGE):
    """Taux de positifs par valeur de `cle`, tiré vers le taux moyen : (positifs + m × moyen) / (artistes + m).

    Renvoie aussi le taux moyen, à donner aux valeurs jamais vues.
    """
    moyen = table["y"].mean()
    stats = table.groupby(cle)["y"].agg(["sum", "count"])
    return (stats["sum"] + m * moyen) / (stats["count"] + m), moyen


def label(sorties, t0, m=LISSAGE, n=PASSEES):
    """Label tremplin : meilleur taux de passage en major, sur les cohortes passées, parmi les labels de l'artiste.

    Les labels des cohortes passées sont ceux connus à t0 : la base ne date pas les labels ajoutés avant 2011.
    """
    base = connues(sorties, t0)
    paires = []
    for t, cohorte_t in cohortes_passees(sorties, t0, n).groupby("t0"):
        labels_t = labels_connus(recentes(base, t), t0)[["artist", "label"]].drop_duplicates()
        paires.append(cohorte_t.merge(labels_t, on="artist"))
    taux, moyen = taux_lisse(pd.concat(paires), "label", m)

    recents = recentes(sorties, t0)
    labels = labels_connus(recents, t0)
    actuels = labels[["artist", "label"]].drop_duplicates()
    actuels["taux"] = actuels["label"].map(taux).fillna(moyen)
    tous = lire("label", ["id", "gid"])
    avec_label = labels.loc[~labels["label"].isin(tous.loc[tous["gid"] == NO_LABEL, "id"]), "release"]
    table = pd.DataFrame({"part_label_connu": recents["release"].isin(avec_label).groupby(recents["artist"]).mean()})
    table["labels"] = actuels.groupby("artist").size()
    table["label_tremplin"] = actuels.groupby("artist")["taux"].max()
    return table.fillna({"labels": 0})


@cache
def liens_labels_crees():
    creation = date_creation(lire("l_label_label", ["id", "link", "entity0", "entity1", "last_updated"]))
    return liens_labels().merge(creation, on=["link", "entity0", "entity1"])


@cache
def arbre_connu(t0):
    """Labels des majors avec leurs périodes, distributeurs compris, d'après les liens entre labels créés avant t0."""
    liens = liens_labels_crees()
    return labels_majors_dates(liens[liens["creation"].dt.year < t0], VARIANTES["+ imprint"])


@cache
def distribues_par_major(t0):
    """Labels indépendants distribués par un label de major, avec un lien label distribution créé
    et commencé avant t0 (ou sans date de début)."""
    liens = liens_labels_crees()
    arbre = arbre_connu(t0)["label"]
    distribution = (
        (liens["type_lien"] == "label distribution") & liens["entity0"].isin(arbre) & ~liens["entity1"].isin(arbre)
        & (liens["creation"].dt.year < t0) & ~(liens["begin_date_year"] >= t0).fillna(False)
    )
    return liens.loc[distribution, "entity1"]


@cache
def credits_multiples():
    credit = lire("artist_credit_name", ["artist_credit", "position", "artist"])
    return credit[credit.groupby("artist_credit")["artist"].transform("size") > 1]


def proximite(sorties, t0):
    """Sorties sur les W ans chez un label distribué par une major, chez un distributeur de major,
    éditions isolées chez une major, et apparitions comme invité sur un disque de major."""
    recents = recentes(sorties, t0)
    labels = labels_connus(recents, t0).merge(recents[["release", "annee"]].drop_duplicates(), on="release")
    arbre = arbre_connu(t0)
    cible = sans_distributeurs(arbre)
    distribues = distribues_par_major(t0)

    table = pd.DataFrame(index=recents["artist"].unique())
    table["label_distribue_major"] = labels[labels["label"].isin(distribues)].groupby("artist")["release"].nunique()
    distributeurs = en_major(labels, arbre) & ~en_major(labels, cible)
    table["distributeur_major"] = labels[distributeurs].groupby("artist")["release"].nunique()
    table["edition_major_isolee"] = labels[en_major(labels, cible)].groupby("artist")["release"].nunique()

    invites = recents.loc[recents["major"], ["release", "artist_credit"]].merge(credits_multiples(), on="artist_credit")
    table["invite_major"] = invites[invites["position"] > 0].groupby("artist")["release"].nunique()
    return table.fillna(0)


@cache
def membres():
    """Liens « member of band » entre artistes (membre, groupe), avec leur date de création estimée."""
    types = lire("link_type", ["id", "name", "entity_type0", "entity_type1"])
    entre_artistes = (types["entity_type0"] == "artist") & (types["entity_type1"] == "artist")
    membre = (types["name"] == "member of band") & entre_artistes
    link = lire("link", ["id", "link_type"])
    liens = date_creation(lire("l_artist_artist", ["id", "link", "entity0", "entity1", "last_updated"]))
    liens = liens[liens["link"].isin(link.loc[link["link_type"].isin(types.loc[membre, "id"]), "id"])]
    return liens.rename(columns={"entity0": "membre", "entity1": "groupe"})[["membre", "groupe", "creation"]]


def reseau(sorties, t0):
    """Collaborateurs crédités sur les sorties connues à t0, membres ou groupes liés, et combien sont déjà en major."""
    base = connues(sorties, t0)
    en_major = premiere_sortie_major(base).index

    credits = credits_multiples()
    credits = credits[credits["artist_credit"].isin(base["artist_credit"])]
    paires = credits.merge(credits, on="artist_credit", suffixes=("", "_autre"))
    paires = paires.loc[paires["artist"] != paires["artist_autre"], ["artist", "artist_autre"]].drop_duplicates()

    liens = membres()
    liens = liens[liens["creation"].dt.year < t0]
    lies = pd.concat([
        liens.rename(columns={"membre": "artist", "groupe": "artist_autre"}),
        liens.rename(columns={"groupe": "artist", "membre": "artist_autre"}),
    ])[["artist", "artist_autre"]].drop_duplicates()

    table = pd.DataFrame(index=recentes(sorties, t0)["artist"].unique())
    table["collaborateurs"] = paires.groupby("artist").size()
    table["collaborateurs_major"] = paires["artist_autre"].isin(en_major).groupby(paires["artist"]).sum()
    table["lies"] = lies.groupby("artist").size()
    table["lies_major"] = lies["artist_autre"].isin(en_major).groupby(lies["artist"]).sum()
    table = table.fillna(0)
    avec = table["collaborateurs"].where(table["collaborateurs"] > 0)
    table["part_collaborateurs_major"] = table["collaborateurs_major"] / avec
    return table


@cache
def pays_des_zones():
    """Code pays de chaque zone : on remonte les liens « part of » (ville, région) jusqu'à un pays."""
    codes = lire("iso_3166_1").drop_duplicates("area").set_index("area")["code"]
    lien = lire("l_area_area", ["entity0", "entity1"])  # entity0 contient entity1, tous ces liens sont des « part of »
    contenant = lien.drop_duplicates("entity1").set_index("entity1")["entity0"]
    zones = lire("area", ["id"])["id"]
    pays = zones.map(codes)
    courant = zones
    for _ in range(10):  # 10 étages suffisent (ville, région, pays)
        courant = courant.map(contenant)
        pays = pays.fillna(courant.map(codes))
    return pd.Series(pays.values, index=zones.values)


@cache
def profils():
    """Type, sexe, pays et année de début de chaque artiste. Valeurs d'aujourd'hui, sans historique."""
    artist = lire("artist", ["id", "type", "gender", "area", "begin_date_year"]).set_index("id")
    types = lire("artist_type", ["id", "name"]).set_index("id")["name"]
    sexes = lire("gender", ["id", "name"]).set_index("id")["name"]
    return pd.DataFrame({
        "type": artist["type"].map(types),
        "sexe": artist["gender"].map(sexes),
        "pays": artist["area"].map(pays_des_zones()),
        "debut": artist["begin_date_year"],
    })


def profil(sorties, t0):
    """Type, sexe, pays, âge (années depuis le début de l'artiste) et indicateurs de valeur manquante."""
    table = profils().reindex(recentes(sorties, t0)["artist"].unique())
    table["age"] = t0 - table.pop("debut")
    for colonne in ["type", "sexe", "pays", "age"]:
        table[f"{colonne}_manquant"] = table[colonne].isna()
    return table


def marche(sorties, t0, m=LISSAGE, n=PASSEES):
    """Taux de passage en major du pays de l'artiste sur les cohortes passées, lissé."""
    pays = profils()["pays"]
    passees = cohortes_passees(sorties, t0, n)
    passees["pays"] = passees["artist"].map(pays)
    taux, moyen = taux_lisse(passees.dropna(subset="pays"), "pays", m)
    pays_actifs = pays.reindex(recentes(sorties, t0)["artist"].unique())
    return pays_actifs.map(taux).fillna(moyen).where(pays_actifs.notna()).rename("taux_pays").to_frame()


FAMILLES = [activite, traction, documentation, label, proximite, reseau, profil, marche]


def vue_de_t0(sorties, t0, seuil=0.5):
    """Sorties que la base connaissait à t0, avec `major` et `major_un_jour` recalculés comme à t0.

    Seuls comptent les labels de sortie et les liens entre labels déjà créés à t0, et la règle
    du disque à moitié en major ne voit que les éditions déjà connues.
    """
    vue = connues(sorties, t0).copy()
    labels = labels_des_sorties()
    connus = labels.loc[labels["creation"].dt.year < t0, ["release", "label"]]
    paires = vue[["release", "annee"]].merge(connus, on="release")
    return marquer_majors(vue, paires, sans_distributeurs(arbre_connu(t0)), seuil)


def variables(sorties, t0):
    """Cible y et variables de toutes les familles pour la cohorte t0, une ligne par artiste.

    La cible utilise toute la base, les variables seulement la base vue de t0.
    """
    table = cohorte(sorties, t0).rename("y").to_frame()
    vue = vue_de_t0(sorties, t0)
    for famille in FAMILLES:
        table = table.join(famille(vue, t0))
    table.insert(0, "t0", t0)
    return table


# Famille ajoutée par le notebook 06, dans data/features_web.parquet seulement (hors de FAMILLES).

# Groupes de types de liens artiste-URL (https://musicbrainz.org/relationships/artist-url).
GROUPES_URL = {
    "site_officiel": ["official homepage"],
    "reseaux_sociaux": ["social network", "myspace", "purevolume", "soundcloud", "video channel", "youtube"],
    "streaming": [
        "streaming", "free streaming", "apple music", "youtube music",
        "purchase for download", "download for free", "purchase for mail-order", "CD Baby",
    ],
    "bandcamp": ["bandcamp"],
    "bases": [
        "other databases", "allmusic", "bandsintown", "BookBrainz", "CPDL", "discogs", "IMDb", "IMSLP", "last.fm",
        "secondhandsongs", "setlistfm", "songkick", "vgmdb", "VIAF", "wikidata", "wikipedia",
    ],
}


@cache
def liens_url():
    types = lire("link_type", ["id", "name"]).rename(columns={"id": "link_type", "name": "type_lien"})
    link = lire("link", ["id", "link_type"]).rename(columns={"id": "link"})
    liens = date_creation(lire("l_artist_url", ["id", "link", "entity0", "entity1", "last_updated"]))
    liens = liens.merge(link, on="link").merge(types, on="link_type")
    return liens.rename(columns={"entity0": "artist", "entity1": "url"})[["artist", "url", "type_lien", "creation"]]


def web(sorties, t0, liens=None):
    """Liens URL créés avant t0 : leur nombre et, pour chaque groupe de GROUPES_URL, si l'artiste en a un.

    `liens` remplace la table de liens_url(), pour le test anti-fuite.
    """
    liens = liens_url() if liens is None else liens
    liens = liens[liens["creation"].dt.year < t0]
    table = pd.DataFrame(index=recentes(sorties, t0)["artist"].unique())
    table["liens_url"] = liens.groupby("artist").size()
    for groupe, types in GROUPES_URL.items():
        table[groupe] = table.index.isin(liens.loc[liens["type_lien"].isin(types), "artist"])
    return table.fillna({"liens_url": 0})


if __name__ == "__main__":
    # Table des modèles, comme la partie 11 du notebook 03.
    sorties = sorties_du_projet()
    tables = []
    for t0 in COHORTES:
        tables.append(variables(sorties, t0))
        print(f"cohorte {t0} : {len(tables[-1])} artistes")
    table = pd.concat(tables).drop(columns=ECARTEES)
    table.reset_index(names="artist").to_parquet(FICHIER_VARIABLES, index=False)
    print(f"{FICHIER_VARIABLES} : {len(table)} lignes, {table.shape[1] - 2} variables")
