# Flight Disruption Risk Predictor

🔗 **Démo en ligne : [flight-disruption-risk.onrender.com](https://flight-disruption-risk.onrender.com)**
(hébergement gratuit Render — le service se met en veille après 15 min d'inactivité, ~1 minute pour se réveiller au premier appel)

Code source : [github.com/lilian-hurst/flight-disruption-risk](https://github.com/lilian-hurst/flight-disruption-risk)

Prévision à +3h du risque de perturbation météo pour un aéroport, combinée à la congestion aérienne en direct. Construit comme projet portfolio ciblant les stages **Amadeus** (Data Scientist / Data Engineer), et réutilisable pour **Thales** (dataviz dans une chaîne MLOps) et **Artefact/Bel Group/Fleury Michon/Sézane** (data engineering + ML de bout en bout).

## Ce que fait le projet

1. **Ingestion de données réelles, en direct** : trafic aérien via [OpenSky Network](https://opensky-network.org/) (positions/altitudes/vitesses d'avions réels, API publique sans clé) et météo via [Open-Meteo](https://open-meteo.com/) (historique + prévisions, API publique sans clé).
2. **Modèle XGBoost** entraîné sur ~3,5 mois de météo historique réelle (8 aéroports européens majeurs), qui prédit si les conditions seront à risque de perturbation **dans 3 heures**.
3. **API FastAPI** (`GET /risk/{icao}`) qui combine la prédiction du modèle avec la congestion aérienne en temps réel (nombre d'avions détectés, phase de vol) pour un aéroport donné.
4. **Conteneurisé avec Docker**, testé (build + run + requête réelle vérifiés dans cet environnement).

## Résultats du modèle (sur un split test réel, 20% des données)

| Métrique | Valeur |
|---|---|
| Accuracy | 0.83 |
| Précision | 0.67 |
| Rappel | 0.42 |
| F1 | 0.51 |
| ROC-AUC | 0.86 |
| Émissions CO2eq de l'entraînement | ~1,8 × 10⁻⁶ kg (suivi CodeCarbon) |

## Méthodologie et limites (important, à savoir avant un entretien)

**Pourquoi une prédiction à +3h plutôt qu'un vrai retard en minutes ?**
Les endpoints historiques d'OpenSky (`/flights/departure`, `/flights/arrival`) nécessitent un compte enregistré — vérifié dans cet environnement : un appel anonyme renvoie une erreur HTTP 403 *"You cannot access historical flights"*. Sans données historiques de retards réels, le projet utilise l'archive météo historique d'Open-Meteo (qui, elle, est en accès libre) et une règle documentée de "risque météo" (précipitations, rafales de vent, couverture nuageuse au-delà de seuils standards en aéronautique) comme **proxy de label**, pas comme des retards réels mesurés.

**Piège évité, à mentionner en entretien si on vous pousse sur la méthodologie** : une première version du modèle prédisait ce risque à partir des mêmes variables utilisées pour construire la règle — et obtenait logiquement 100% d'accuracy, ce qui est un signal classique de fuite de données (*data leakage*), pas un bon modèle. Le projet a été corrigé pour prédire le risque **3 heures dans le futur** à partir des conditions **actuelles** : c'est un vrai problème de prévision, et le modèle obtient des scores nettement plus modestes (83% accuracy, 0.86 ROC-AUC) — beaucoup plus crédibles et défendables.

**Comment brancher de vrais retards mesurés** : créer un compte OpenSky gratuit (accès aux endpoints historiques), ou utiliser un jeu de données de ponctualité (ex. DOT/BTS, Eurocontrol Network Manager), et remplacer `label_disruption_risk()` dans `src/build_dataset.py` par une jointure sur des retards réels.

**Limite connue sur le déploiement en ligne (Render)** : OpenSky Network est injoignable depuis les serveurs Render (timeout de connexion systématique, vérifié à plusieurs reprises), alors qu'il répond normalement en local et en Docker local — probablement un blocage réseau côté OpenSky visant certaines plages d'IP d'hébergeurs cloud. Résultat : sur la démo en ligne, la **prédiction météo (le cœur du projet) fonctionne parfaitement avec de vraies données**, mais le bloc "trafic aérien en direct" affiche un message d'indisponibilité au lieu des données OpenSky. Le code gère ce cas proprement (message clair, pas de crash) plutôt que de le masquer — voir `src/api.py`. En local ou dans le conteneur Docker, cette limite n'existe pas.

## Structure du projet

```
flight-delay-predictor/
├── src/
│   ├── airports.py        # Référentiel de 8 aéroports européens (ICAO, lat/lon)
│   ├── weather.py         # Client Open-Meteo (historique + temps réel)
│   ├── opensky.py         # Client OpenSky Network (trafic aérien en direct)
│   ├── build_dataset.py   # Construction du jeu d'entraînement (météo réelle + label proxy)
│   ├── train.py           # Entraînement XGBoost + métriques + suivi CodeCarbon
│   └── api.py              # API FastAPI de scoring en direct
├── tests/                  # 17 tests pytest (logique de features + API, réseau mocké)
├── models/                  # Modèle entraîné, encodeur, métriques (versionnés)
├── data/                    # Jeu de données d'entraînement (régénérable, non versionné)
├── requirements.txt
└── Dockerfile
```

## Interface graphique

Un dashboard web (`static/index.html`, HTML/CSS/JS vanilla, sans framework — servi directement par FastAPI sur `/`) permet de :
- choisir un aéroport et voir son risque de perturbation à +3h, la météo en direct et le trafic aérien en direct
- comparer tous les aéroports suivis en un coup d'œil

Aucune dépendance supplémentaire : la même image Docker sert à la fois l'API et l'interface.

## Déploiement (gratuit)

Le projet est prêt à déployer sur [Hugging Face Spaces](https://huggingface.co/spaces) (hébergement Docker gratuit, sans carte bancaire, standard dans la communauté data/ML — idéal pour un lien à mettre sur un CV).

```bash
# 1. Créer un compte gratuit sur huggingface.co (si pas déjà fait)
# 2. Créer un token d'accès (write) sur https://huggingface.co/settings/tokens
# 3. Lancer le déploiement :
HF_TOKEN=hf_xxx python3 deploy/deploy_hf_space.py
```

Voir `deploy/deploy_hf_space.py` pour le détail (création du Space, upload du code/modèle/Dockerfile, pas des données brutes régénérables).

## Utilisation

### En local
```bash
pip install -r requirements.txt

# Reconstruire le jeu de données et le modèle (optionnel : un modèle déjà entraîné est fourni)
python3 src/build_dataset.py --start 2026-06-01 --end 2026-09-17
python3 src/train.py

# Lancer les tests
pytest tests/ -v

# Lancer l'API
uvicorn src.api:app --reload
```

### Avec Docker (testé et fonctionnel dans cet environnement)
```bash
docker build -t flight-delay-predictor .
docker run -p 8000:8000 flight-delay-predictor
```

### Exemple de requête
```bash
curl http://localhost:8000/risk/LFMN
```
```json
{
  "airport": {"icao": "LFMN", "name": "Nice Côte d'Azur", "iata": "NCE", "lat": 43.6584, "lon": 7.2159},
  "forecast_horizon_hours": 3,
  "disruption_risk_score": 0.07,
  "disruption_risk_level": "faible",
  "current_weather": {"temperature_2m": 20.2, "windspeed_10m": 7.3, "precipitation": 0.0, "cloudcover": 47, ...},
  "live_traffic_congestion": {"n_aircraft": 5, "n_on_ground": 0, "n_low_altitude": 1, "avg_velocity": 207.2, ...},
  "queried_at": "2026-09-18T04:39:56Z"
}
```
*(exemple réel obtenu en interrogeant l'API pendant le développement — OpenSky a bien renvoyé 5 avions réellement présents près de Nice à cet instant)*

Aéroports disponibles : `LFPG` (Paris CDG), `LFPO` (Paris Orly), `LFMN` (Nice), `LFLL` (Lyon), `EDDF` (Frankfurt), `EHAM` (Amsterdam), `EGLL` (London Heathrow), `LEMD` (Madrid).

## Ce qui a été vérifié en conditions réelles dans cet environnement
- Appels réels à l'API OpenSky Network (`/states/all`) : avions réellement détectés (ex. 33 avions autour de CDG, 27 autour de Heathrow au moment du test)
- Appels réels à l'archive historique Open-Meteo (8 aéroports × ~3,5 mois → 20 904 lignes)
- Entraînement XGBoost réel avec suivi CodeCarbon réel
- Build Docker réel (image construite avec succès), avec l'interface graphique incluse
- Conteneur Docker lancé et interrogé avec succès (requête réelle aboutie, réponse correcte, page `/` servie)
- 17 tests pytest, tous passants
- **Déploiement réel sur Render vérifié en ligne** : `/health`, `/` (interface) et `/risk/{icao}` testés depuis l'extérieur après mise en ligne, prédiction ML confirmée fonctionnelle avec de vraies données météo (le bloc trafic OpenSky est indisponible spécifiquement depuis Render — voir section méthodologie)

## Pistes d'extension (voir `projets_thales_dassault_amadeus.md` pour le contexte complet)
- Dashboard Streamlit/Dash branché sur l'historique des prédictions (pour la piste Thales — visualisation dans une chaîne MLOps)
- Déploiement Kubernetes (manifestes à écrire, pas de cluster testé ici faute de `kind`/`minikube` installés)
- Tracking d'expériences MLflow pour comparer plusieurs versions du modèle
- Remplacement du label proxy par de vrais retards mesurés (voir section méthodologie)
