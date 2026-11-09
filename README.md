# Qui va signer chez Universal, Sony ou Warner ?

Projet de data science à Télécom Paris, par Iliana et Zakariya.

Universal, Sony et Warner sont les trois majors, les géants du disque. À elles trois, elles font environ deux tiers de la musique enregistrée vendue dans le monde. Pour trouver leurs futures stars, elles paient des A&R (pour « artistes et répertoire »), des chasseurs de talents qui écoutent des démos, vont aux concerts et décident qui signer.

Le problème, c'est le nombre. Rien que dans notre base, 82 372 artistes indépendants avaient sorti un disque dans les 3 ans avant 2021, et un A&R n'en écoute que quelques centaines par an. Celui qui repère le bon avant les autres le signe avant que les enchères montent.

Alors on lui fait une liste. Chaque année, on classe les artistes indépendants selon leurs chances de sortir un disque chez une major dans les 3 ans.

Testé sur 2021, ça marche. Parmi les 500 premiers de la liste, 74 sortent bien un disque en major dans les 3 ans. Au hasard, on en aurait eu 4. Pour trouver ces 74 artistes sans la liste, il aurait fallu en écouter plus de 8 000.

## Les données

On part de MusicBrainz, l'encyclopédie ouverte de la musique, remplie par des bénévoles, un peu le Wikipédia du disque. On télécharge toute la base (8,2 Go en deux archives) et on en garde 30 tables.

On a figé la version du 4 novembre 2026 (`20261104-002222`). La base bouge tous les jours et le serveur ne garde que les dernières versions. Avec une version plus récente, le code tourne pareil mais les chiffres changent un peu.

Trois mots reviennent partout. Une sortie, c'est un disque publié (album, EP ou single), souvent en plusieurs éditions selon le pays ou le format. Un label, c'est la marque sur le disque, comme Columbia, et il appartient souvent à plus gros que lui (Columbia est à Sony). Un lien relie deux fiches de la base, un label à sa maison mère, un artiste à son label, un musicien à son groupe.

| Thème | Tables |
|---|---|
| Artistes | `artist`, `artist_type`, `gender`, `area`, `iso_3166_1`, `l_area_area`, `artist_credit_name`, `l_artist_artist` |
| Sorties | `release`, `release_status`, `release_country`, `release_unknown_country`, `release_label`, `release_group`, `release_group_meta`, `release_group_primary_type`, `release_group_secondary_type`, `release_group_secondary_type_join`, `medium`, `medium_format` |
| Labels et liens | `label`, `label_type`, `l_label_label`, `l_artist_label`, `link`, `link_type` |
| Web et concerts (notebook 06) | `url`, `l_artist_url`, `event`, `l_artist_event` |

Et c'est sale. Du texte brut sans en-tête ni types, une sortie éparpillée sur une dizaine de tables, des dates souvent réduites à l'année, 56 % des labels sans type, des liens de propriété entre labels datés une fois sur huit. Et surtout, la base ne connaît que son état d'aujourd'hui, pas ce qu'elle savait il y a cinq ans.

## Ce qu'on a fait, et pourquoi

| Étape | Ce qu'on fait | Pourquoi |
|---|---|---|
| `src/donnees.py` | Télécharge la base, vérifie les SHA256, garde 30 tables en parquet | Une archive corrompue nous est déjà arrivée |
| `src/nettoyage.py` | Type chaque colonne d'après son nom | Tout arrive en texte, même les dates |
| `src/transformation.py`, le ménage | `charger_sorties`, `garde`, `retenir` et `artiste_principal` virent promos, compils, rééditions et Various Artists, et datent chaque sortie | Sinon on compte des signatures qui n'en sont pas |
| `src/transformation.py`, la table | Fabrique la cible et les variables dans `data/features.parquet` | Une seule table, une ligne par artiste et par T0 |
| `src/modelisation.py` | Prépare les variables, construit les modèles, calcule les mesures | Le même code pour 05 et 06 |
| `01_definition_cible` | Qui est en major, quelles sorties comptent, quelles fenêtres de temps | Chaque réglage vient d'un chiffre |
| `02_validation_cible` | La cible vérifiée à la main, face aux contrats, avec les dates des liens | Savoir si la cible dit vrai avant de prédire |
| `03_features` | 30 variables, sans rien du futur | La base connaît la fin de l'histoire |
| `04_exploration` | Distributions, redondances, dérive de 2014 à 2021 | Voir les données bouger avant de modéliser |
| `05_modelisation` | Découpage dans le temps, trois modèles, test sur 2021 | Juger sur le haut de la liste, comme un A&R |
| `06_variables_datees` | Concerts et liens URL datés comme les sorties | Une info qu'on croyait figée à aujourd'hui |

## Lancer le projet

Python 3.12, PowerShell, depuis la racine du dépôt.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.donnees
python -m src.transformation
```

Le téléchargement est le plus long. Si notre version a disparu du serveur, le script prend la plus récente et prévient. Ensuite, les notebooks dans l'ordre, de 01 à 06. Pour tout rejouer d'un coup (`PYTHONUTF8` évite une erreur d'encodage de Windows).

```powershell
cd notebooks
$env:PYTHONUTF8 = "1"
jupyter execute --inplace --timeout=1800 01_definition_cible.ipynb 02_validation_cible.ipynb 03_features.ipynb 04_exploration.ipynb 05_modelisation.ipynb 06_variables_datees.ipynb
```

## Nos choix

### C'est quoi, une major ?

Un disque porte le nom du label (Columbia), pas de la major (Sony), et 56 % des labels n'ont pas de type. Alors on part des maisons mères des trois groupes et on descend les liens label ownership et imprint. Ça donne 3 616 labels de majors. On ne suit ni label rename, qui ferait passer les labels rattachés à deux majors de 68 à 597, ni label distribution, car un distributeur ne possède pas ce qu'il distribue (on retire aussi les labels typés Distributor).

Mais les labels se vendent. Quand la base date un lien, le label ne compte en major que pendant ces années-là (Chrysalis n'est chez Warner que de 2013 à 2016). Une fin de lien à un an près d'un changement de nom ne compte pas, le label a été renommé, pas vendu (273 des 517 fins datées). Au total, un quart des sorties en major ne le sont plus.

### Quelles sorties comptent ?

On garde 3 808 518 sorties sur 5 903 902. Les officielles, en album, EP ou single, en première édition, datées par leur pays le plus ancien. Exit les promos (des copies), les compils, lives, remix et BO (du catalogue ou plusieurs artistes) et les rééditions (16 % des sorties de majors contre 6 % ailleurs, ressortir un vieil album n'est pas signer l'artiste). La date du premier disque donnée par la base compte aussi les promos et les bootlegs, alors on la recalcule, sinon 6 276 sorties officielles passeraient pour des rééditions. Une sortie compte en major si au moins la moitié des éditions de son disque y sont, ça écarte les licences dans un seul pays.

Dans « A feat. B », seul A compte. Various Artists et [unknown] sont exclus, et [no label] veut dire autoédité, pas inconnu.

### Qui on suit, et combien de temps ?

À chaque date T0, on suit les indépendants qui ont sorti quelque chose dans les 3 ans d'avant et qui sont déjà dans la base. Un artiste est positif si sa première sortie en major tombe dans les 3 ans qui suivent.

Indépendant veut dire jamais sorti chez un label de major, même avant que la major le rachète. Marduk sortait déjà chez Century Media avant son rachat par Sony en 2015, et un rachat n'est pas une signature.

```
   tout ce que la base savait à T0          T0          3 ans après
  --------------------------------------------|---------------------------
   les variables                              |   la cible
   (au moins une sortie dans les 3 ans        |   (une première sortie
    d'avant pour être suivi)                  |    en major ici ?)
```

3 ans avant, car 82 % des positifs avaient sorti un disque dans les 3 ans avant de passer en major. 3 ans après, car une signature met du temps à donner un disque. Sur 501 artistes avec un contrat daté chez une major, 47 % sortent en major l'année du contrat et 25 % un ou deux ans après. Vampire Weekend aurait signé chez Columbia fin 2016, premier album Columbia en 2019.

On entraîne sur T0 = 2014 à 2018 et on teste sur 2021. Pas plus récent, la base se remplit des années après coup, une sortie de 2015 n'y était qu'à 37 % un an après.

### Interdit de regarder le futur

Le gros piège. La base d'aujourd'hui connaît la fin de l'histoire. Environ la moitié des positifs n'avaient encore aucune sortie dans la base à T0, leur passé a été ajouté une fois connus. Vue d'aujourd'hui, la cohorte 2021 aurait 1 553 positifs au lieu de 719, et un modèle apprendrait juste « bien documenté, donc va signer ».

Donc on reconstitue ce que la base savait à T0. Les sorties n'ont pas de date de création, mais leurs id sont croissants. Une sortie existait donc au plus tard à la plus petite date de mise à jour des sorties d'id plus grand. Même astuce pour les labels des sorties, les formats, les membres des groupes, les liens entre labels et les liens URL. Même le statut en major se recalcule à T0. Vues d'aujourd'hui, 53 % des sorties en major des 3 ans avant 2014 ne l'étaient pas encore en 2014, leur label ou leur lien vers une major est arrivé après.

Pour vérifier, on injecte 10 000 fausses sorties en major, 5 000 faux labels Sony et 1 000 faux liens vers Sony, tous ajoutés après T0, et aucune variable ne bouge. Datés de la veille de T0, ils en font bouger 11.

### Les variables

30 variables en 8 familles, toutes calculées avec ce que la base savait à T0. La dernière colonne donne la variable la plus parlante, avec son AUC seule (0,5 si elle ne sert à rien) ou son taux de positifs.

| Famille | Dedans | Signal le plus net |
|---|---|---|
| activité | disques par type et plus anciens, ancienneté, années actives, récence, tendance | singles, AUC 0,61 |
| traction | éditions par disque, pays de sortie, part d'éditions physiques | pays de sortie, AUC 0,56 |
| documentation | délai d'ajout dans la base | AUC 0,46 et 0,52 |
| label | part de sorties avec label, nombre de labels, label tremplin | label tremplin, AUC 0,60 |
| proximité des majors | label distribué par une major, distributeur de major, édition isolée en major, invité sur un disque de major | édition isolée, 20 % de positifs contre 1 % |
| réseau | collaborateurs et membres liés, et combien sont déjà en major | part de collaborateurs en major, AUC 0,60 |
| profil | type d'artiste, pays, indicateurs de manquant | pays manquant, AUC 0,42 |
| marché | taux de passage en major du pays | AUC 0,53 |

Le label tremplin, c'est le meilleur taux de passage en major parmi les labels de l'artiste, mesuré sur les cohortes d'avant. Le sexe et l'âge sont virés, les fiches complétées après coup leur donnent un faux signal (83 % des positifs ont une fiche modifiée après T0, contre 48 % des négatifs). Les notes et les tags aussi, on ne les a qu'à l'état d'aujourd'hui.

Le notebook 06 ajoute une neuvième famille, les liens URL, datés comme les sorties. Les concerts se datent aussi, mais la base ne les connaît que depuis fin 2014, trop tard pour l'entraînement.

### Le modèle

Validation sur 2018 avec seulement 2014 et 2015, car 299 des 620 positifs de 2018 sont déjà positifs en 2017, c'est le même passage en major vu deux fois. Avec 1 % de positifs, dire toujours non donne 99 % d'exactitude et ne trouve personne. Alors on juge le classement, avec l'AUC PR et la part de positifs dans les 100 et 500 premiers, ce que voit un A&R qui en écoute 100 ou 500.

On retient une régression logistique sur les rangs des variables (`QuantileTransformer`, L2, C = 0,1). Les rangs évitent que les queues lourdes écrasent tout (AUC PR de 0,038 à 0,047). En 2018 elle bat la forêt et le boosting. L'écart avec le boosting n'est pas significatif et ils font jeu égal sur un second découpage (2014 vers 2017), alors on prend la plus simple.

## Résultats

Validation sur 2018, entraînement sur 2014 et 2015.

| Modèle | AUC PR | AUC ROC | précision@100 | précision@500 |
|---|---|---|---|---|
| hasard | 0,010 | 0,488 | 0,02 | 0,008 |
| label tremplin seul | 0,017 | 0,554 | 0,11 | 0,050 |
| régression logistique | 0,048 | 0,757 | 0,19 | 0,114 |
| forêt aléatoire | 0,036 | 0,752 | 0,09 | 0,094 |
| gradient boosting | 0,041 | 0,746 | 0,21 | 0,086 |

Test sur 2021, entraînement sur 2014 à 2018. 82 372 artistes, dont 719 positifs (0,87 %).

| Modèle | AUC PR | AUC ROC | précision@100 | précision@500 | rappel@500 |
|---|---|---|---|---|---|
| label tremplin seul | 0,018 | 0,615 | 0,09 | 0,060 | 0,042 |
| régression logistique (retenue) | 0,050 | 0,770 | 0,21 | 0,148 | 0,103 |
| forêt aléatoire | 0,062 | 0,777 | 0,26 | 0,162 | 0,113 |
| gradient boosting | 0,051 | 0,763 | 0,18 | 0,120 | 0,083 |
| moyenne logistique + forêt (idée d'après le test) | 0,060 | 0,783 | 0,19 | 0,166 | 0,115 |

La logistique fait près de 6 fois mieux que le hasard en AUC PR, et 15 % de ses 500 premiers passent en major, 17 fois le taux moyen.

Surprise, en 2021 la forêt passe nettement devant la logistique (le boosting fait jeu égal). Ce n'est pas 2021 qui change tout, avec les seules cohortes 2014 et 2015 la logistique reste devant. Ce sont les cohortes en plus qui aident les arbres (+0,021 d'AUC PR pour la forêt, +0,003 pour la logistique), et les artistes sortis dans plus de 10 pays, 47 en 2018 contre 1 478 en 2021, bien plus souvent passés en major (5,9 % contre 0,78 %).

On garde la logistique, choisie avant le test, pour ne pas tricher. Après le test, une idée. La moyenne logistique + forêt bat la logistique sur les deux validations, et 16,6 % de ses 500 premiers passent en major en 2021, un chiffre un peu flatteur puisque l'idée vient du test. Les liens URL datés du notebook 06 font gagner un peu les quatre modèles, mais le gain n'est net qu'une fois sur huit. Pour la cohorte suivante, on partirait sur cette moyenne.

## Limites

- Les pays de sortie ne se datent pas, alors que les sorties listées dans plus de 100 pays passent de 983 en 2013 à 5 710 en 2020. Les arbres sont peut-être un peu optimistes en 2021.
- Les labels sont de moins en moins renseignés (83 % des sorties de 2000 ont un label connu, 58 % en 2025), et une sortie en major sans label passe à la trappe.
- Il reste des faux positifs. Sur 20 positifs de 2021 tirés au hasard et vérifiés à la main, il en reste 16 après corrections, dont 8 justes. Les faux sont surtout des homonymes et des artistes déjà signés dont les premiers disques n'ont pas de label dans la base. Face aux contrats datés, la cible retrouve quand même 78 % des signatures.
- Les dates des liens ont leurs trous. Le label nommé « Universal » n'est relié à Universal que de 1998 à 2000, et 650 de ses sorties de 2014 à 2023 ne comptent plus en major. Aucune règle simple ne sépare ces trous des vraies ventes.
- Notre définition d'indépendant est prudente. Un artiste passé par un label racheté depuis par une major est écarté, même s'il signe ensuite ailleurs (In Flames, de Century Media à Epic). Dater aussi cette exclusion ajouterait 68 à 149 positifs par cohorte, mais en 2014, 58 sur 142 ne seraient que des rachats.
- La base a ses chouchous. Elle documente mieux les artistes connus et occidentaux (les États-Unis font un cinquième des cohortes), et le modèle ne voit pas les futurs signés pas encore documentés à T0.

## Sources

- Les données MusicBrainz, https://data.metabrainz.org/pub/musicbrainz/data/fullexport/
- Le schéma de la base, https://github.com/metabrainz/musicbrainz-server/blob/master/admin/sql/CreateTables.sql
- Le calcul de la date du premier disque, https://github.com/metabrainz/musicbrainz-server/blob/master/admin/sql/CreateFunctions.sql
- Les liens entre labels, https://musicbrainz.org/relationships/label-label
- Les contrats « recording contract », https://musicbrainz.org/relationship/b336d682-592f-4486-9f45-3d5d59895bdc
- La vente de Chrysalis par Warner, https://musicbusinessworldwide.com/warner-sells-chrysalis-records-back-to-chris-wright-and-blue-raincoat
- Vampire Weekend chez Columbia, https://stereogum.com/1911974/vampire-weekend-reportedly-sign-to-columbia-for-lp4/news
- Le poids des majors en 2024, https://www.midiaresearch.com/blog/recorded-music-market-2024-362-billion-up-65
