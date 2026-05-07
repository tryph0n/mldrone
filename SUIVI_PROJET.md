# SUIVI PROJET MLDRONE - DroneDetect V2
# Classification RF de drones par ML/DL - Projet Defense

## Statut global : PREPARATION CERTIFICATION - Corrections + Dossier Projet

---

## TABLE DES MATIERES
1. [Contexte du projet](#1-contexte-du-projet)
2. [Dataset DroneDetect V2](#2-dataset-dronedetect-v2)
3. [Architecture logicielle](#3-architecture-logicielle)
4. [Stack technique](#4-stack-technique)
5. [Pipelines de traitement](#5-pipelines-de-traitement)
6. [Modeles et resultats](#6-modeles-et-resultats)
7. [Analyse par modele](#7-analyse-par-modele)
8. [Workflow notebooks](#8-workflow-notebooks)
9. [Documentation existante](#9-documentation-existante)
10. [Problemes identifies](#10-problemes-identifies)
11. [Investigation RF-UAVNet](#11-investigation-rfuavnet)
12. [Recommandations globales](#12-recommandations-globales)
13. [Historique des actions](#13-historique-des-actions)
15. [RF-UAVNet amelioration](#15-rfuavnet-amelioration)
16. [Audit qualite scientifique notebooks](#16-audit-qualite-scientifique-notebooks)
17. [Validation des problemes identifies](#17-validation-des-problemes-identifies)
18. [Gap analysis certification](#18-gap-analysis-certification)
19. [Plan d'execution corrections](#19-plan-dexecution-corrections)
21. [MACRO E - Migration Scaleway](#21-macro-e---migration-scaleway--regeneration-features)
22. [Migration notebooks 022/023/031](#22-migration-notebooks-022023031)
23. [Upload artefacts Scaleway](#23-upload-artefacts-scaleway)
24. [MACRO AQ - Audit Quality Fixes](#24-macro-aq--audit-quality-fixes-src--interface-2026-05-08)
25. [Session 8 Completed Tasks](#25-session-8-completed-tasks-2026-05-08)
26. [MACRO EXPLORE - Content Audit + Diagrams](#26-macro-explore--content-audit--diagrams-2026-05-08)

---

## 1. CONTEXTE DU PROJET

- **Objectif** : Classifier 7 modeles de drones a partir de leurs emissions RF a 2.4 GHz
- **Dataset** : DroneDetect V2 (Swinney & Woods, IEEE DataPort, DOI: 10.21227/6w92-0x42)
- **SDR** : Nuand BladeRF (47 MHz - 6 GHz)
- **Frequence centrale** : 2.4375 GHz (bande ISM)
- **Bande passante** : 28 MHz
- **Taux d'echantillonnage** : 60 MHz (complexe)
- **Format** : IQ brut float32 interleave -> complex64
- **Duree par fichier** : 2 secondes (120M echantillons complexes)
- **Segment de traitement** : 20 ms (1,200,000 echantillons)
- **Emplacement dataset** : `/home/sambot/win_downloads/DATASETS/drones/DroneDetect_V2`
- **Contribution originale** : Correction rigoureuse du data leakage present dans RFClassification

---

## 2. DATASET DRONEDETECT V2

### 2.1 Les 7 classes de drones

| Code | Drone | Protocole | Type modulation | EIRP (dBm) | Poids (g) | Vitesse max (km/h) |
|------|-------|-----------|-----------------|------------|-----------|---------------------|
| AIR | DJI Air 2S | OcuSync 3.0 | FHSS | 26 | 595 | 68 |
| DIS | Parrot Disco | Wi-Fi | OFDM | 19 | 750 | 80 |
| INS | DJI Inspire 2 | Lightbridge 2.0 | FHSS | 20 | 3440 | 94 |
| MA1 | DJI Mavic 2 Pro | OcuSync 2.0 | FHSS | 25.5 | 907 | 72 |
| MAV | DJI Mavic Pro | OcuSync 1.0 | FHSS | 26 | 734 | 65 |
| MIN | DJI Mavic Mini | Wi-Fi | OFDM | 19 | 249 | 47 |
| PHA | DJI Phantom 4 | Lightbridge 2.0 | FHSS | 20 | 1380 | 72 |

### 2.2 Conditions experimentales

- **Etats de vol** : ON (au sol), HO (hovering 20m), FY (flying 20m, rayon 40m)
- **Interferences** : CLEAN (aucune), BOTH (WiFi MacBook YouTube + Bluetooth JBL speaker a 2m)
- **Replicas** : 5 fichiers par combinaison valide
- **Total** : 195 fichiers (39/42 combinaisons possibles, 3 manquantes : DIS/HO/BOTH, DIS/HO/CLEAN, PHA/FY/CLEAN)
- **Segments totaux** : 19,478 (apres segmentation 20ms)

### 2.3 Qualite des donnees (notebook 001)
- 2 fichiers legerement plus courts (~1.82s et ~1.76s au lieu de 2.0s)
- 52 fichiers avec valeurs IQ legerement hors [-1, 1] (max ~1.02, mineur)
- Incoherence de nommage : dossiers MP1/MP2, fichiers MA1/MAV (filenames = verite terrain)
- Faible variabilite intra-replica (conditions d'enregistrement stables)

---

## 3. ARCHITECTURE LOGICIELLE

### 3.1 Arborescence

```
/home/sambot/mldrone/
├── .env                    # Config locale (chemin dataset)
├── .env.example            # Template de config
├── pyproject.toml          # Dependencies + config projet
├── uv.lock                 # Lock file (uv)
├── README.md               # Documentation principale
│
├── src/dronedetect/        # PACKAGE CORE (10 modules)
│   ├── __init__.py         # v0.1.0, re-exports
│   ├── config.py           # Constantes : FS, NFFT, mappings, paths → data/models
│   ├── data_loader.py      # load_raw_iq(), parse_filename(), get_dataset_metadata()
│   ├── preprocessing.py    # normalize(), normalize_minmax(), segment_signal(), downsample_iq()
│   ├── features.py         # compute_psd(), compute_spectrogram()
│   ├── models.py           # VGG16FC, ResNet50FC, RFUAVNet
│   ├── pipeline.py         # CLI: feature extraction pipeline
│   ├── splitting.py        # 70/15/15 file-level split
│   ├── export_samples.py   # CLI: export test samples
│   ├── storage.py          # Scaleway S3 upload/download
│   └── upload.py           # CLI: upload artifacts to S3
│
├── interface/              # APP STREAMLIT (4 pages)
│   ├── app.py              # Point d'entree, navigation sidebar
│   ├── settings.py         # Config Streamlit, DRONE_INFO, paths
│   ├── models/             # Wrappers BaseModel (ABC pattern)
│   │   ├── base.py         # BaseModel ABC (predict, predict_proba)
│   │   ├── cnn.py          # VGG16FC, ResNet50FC, CNNModelAdapter
│   │   ├── rfuavnet.py     # RFUAVNet, RFUAVNetAdapter
│   │   └── svm.py          # SVMAdapter
│   ├── services/           # Logique metier
│   │   ├── visualization_service.py    # Plotly: PSD, spectrogram, IQ, proba
│   │   └── model_viz_service.py        # torchinfo/torchview integration
│   ├── views/              # Pages Streamlit
│   │   ├── home_view.py               # Accueil + pipelines Graphviz
│   │   ├── model_comparison_view.py   # Dashboard comparaison 4 modeles
│   │   ├── inference_view.py          # Inference interactive
│   │   └── glossary_view.py           # Glossaire technique (31 termes)
│   └── static/glossary.json
│
├── notebooks/              # 9 NOTEBOOKS (exploration -> preprocessing -> training -> comparison)
│   ├── 001_exploration_general.ipynb
│   ├── 002_exploration_temporal.ipynb
│   ├── 003_exploration_frequentiel.ipynb
│   ├── 004_exploration_frequentiel_advanced.ipynb
│   ├── 006_downsampling_analysis.ipynb
│   ├── 015_preprocessing.ipynb          # Local, wraps pipeline + export
│   ├── 021_training_svm.ipynb
│   ├── 022_training_cnn.ipynb
│   ├── 023_training_rfuavnet.ipynb
│   ├── 031_model_comparison.ipynb
│   └── data/               # Features extraites + notebooks avec outputs
│       └── metadata_cache.parquet
│
├── data/                   # SINGLE SOURCE OF TRUTH
│   ├── models/             # MODELES ENTRAINES
│   │   ├── svm_psd_drone.pkl           # 54 MB
│   │   ├── vgg16_cnn.pth               # 57 MB
│   │   ├── resnet50_cnn.pth            # 93 MB
│   │   ├── rfuavnet_iq.pth             # 61 KB
│   │   ├── model_comparison_results.csv # NB 031 export
│   │   ├── confusion_matrices.json      # NB 031 export
│   │   └── per_class_metrics.csv        # NB 031 export
│   ├── features/           # Features extraites (PSD, spectro, IQ, manifest)
│   ├── test_samples/       # 84 .npz files (4 conditions x 7 drones x 3 types)
│   └── split_indices.npz   # 70/15/15 file-level split
│
├── docs/                   # DOCUMENTATION
│   ├── LEAKAGE_PROOF.md                          # Preuve data leakage RFClassification
│   ├── REFERENTIEL_DRONEDETECT_RFCLASSIFICATION.md # Referentiel technique complet
│   ├── data_collection_methodology.md            # Methodologie acquisition
│   └── diagrams/           # Graphviz DOT (pipeline_global, svm, cnn, rfuavnet)
│
├── figures/                # 34 PNG d'exploration
│   ├── exploration_general/        # 8 figures
│   ├── exploration_temporal/       # 6 figures
│   ├── exploration_frequentiel/    # 6 figures
│   ├── exploration_frequentiel_advanced/ # 8 figures
│   ├── downsampling_analysis/      # 6 figures
│   ├── training_svm/               # VIDE
│   └── training_rfuavnet/          # VIDE
│
└── cache/                  # Cache intermediaire (92 MB)
```

### 3.2 Fichiers Python (17 fichiers, ~2300 LOC total)

| Fichier | LOC | Role |
|---------|-----|------|
| src/dronedetect/config.py | 58 | Constantes globales |
| src/dronedetect/data_loader.py | 96 | Chargement .dat, parsing, metadata |
| src/dronedetect/preprocessing.py | 248 | Normalisation, segmentation, downsampling |
| src/dronedetect/features.py | 118 | PSD (Welch), spectrogram (STFT+viridis) |
| src/dronedetect/models.py | 156 | 4 modeles (SVM, VGG16, ResNet50, RFUAVNet) |
| interface/app.py | 51 | Routeur Streamlit |
| interface/settings.py | 160 | Config interface |
| interface/models/base.py | 26 | ABC BaseModel |
| interface/models/cnn.py | 80 | VGG16FC, ResNet50FC, CNNModelAdapter |
| interface/models/rfuavnet.py | 72 | RFUAVNet, RFUAVNetAdapter |
| interface/models/svm.py | 40 | SVMAdapter |
| interface/services/visualization_service.py | 175 | Graphiques Plotly |
| interface/services/model_viz_service.py | 212 | Architecture modeles |
| interface/views/home_view.py | 160 | Page accueil |
| interface/views/model_comparison_view.py | 275 | Page comparaison |
| interface/views/inference_view.py | 417 | Page inference |
| interface/views/glossary_view.py | 134 | Page glossaire |

---

## 4. STACK TECHNIQUE

### 4.1 Runtime
- **Python** : 3.10 strict (`requires-python = "==3.10.*"`)
- **Package manager** : uv (pas pip)
- **Linter** : ruff 0.14.9

### 4.2 Dependencies principales

| Categorie | Bibliotheques | Versions |
|-----------|--------------|----------|
| ML core | scikit-learn | 1.7.2 |
| Deep Learning | torch, torchvision | >=2.0.0 (CPU ou GPU CUDA 12.1) |
| Signal processing | scipy, numpy | 1.15.3, 2.2.6 |
| Data | pandas, pyarrow | 2.3.3, >=18.1.0 |
| Visualisation | matplotlib, seaborn, plotly | 3.10.8, 0.13.2, >=6.5.0 |
| Computer vision | opencv-python | 4.12.0.88 |
| Web app | streamlit | >=1.52.2 |
| Stats | statsmodels | >=0.14.0 |
| Config | python-dotenv | >=1.0.0 |

### 4.3 Extras optionnels
- `torch-cpu` : PyTorch CPU only
- `torch-gpu` : PyTorch CUDA 12.1
- `viz` : torchview, torchinfo, graphviz

### 4.4 Git
- 4 commits sur `main`, branche propre
- `1b3e9c2` -> `fe9fe18` -> `8959bc5` -> `754a00a` (HEAD)

---

## 5. PIPELINES DE TRAITEMENT

### 5.1 Pipeline SVM (PSD)
```
.dat -> load_raw_iq() [complex64, 120M]
     -> normalize() [Z-score per-file]
     -> segment_signal() [20ms = 1.2M, pas d'overlap]
     -> compute_psd() [Welch, Hamming, 1024-FFT, lineaire]
     -> max normalization per-sample
     -> PsdSVM.predict() [SVM RBF, C=1.0, gamma='scale']
```
- **Features** : 1024-dim PSD vector (linear scale)
- **Split** : StratifiedGroupKFold file-level 80/20

### 5.2 Pipeline CNN (Spectrogram)
```
.dat -> load_raw_iq() [complex64, 120M]
     -> normalize() [Z-score per-file]
     -> segment_signal() [20ms = 1.2M, pas d'overlap]
     -> compute_spectrogram() [STFT, Hann, 1024-FFT, 120 overlap, -dB, Viridis RGB, 224x224x3]
     -> VGG16FC / ResNet50FC [frozen ImageNet + trainable FC]
```
- **Features** : 224x224x3 RGB spectrogram
- **VGG16FC** : FC(25088 -> 7), ~14.7M params (25K trainables)
- **ResNet50FC** : AdaptiveAvgPool2d + FC(2048 -> 7), ~23.6M params (~14K trainables) (corrected session 6)
- **Training** : max 120 epochs, batch=128, lr=0.001/0.0001, Adam, CUDA, early stopping, ReduceLROnPlateau
- **Memory** : RGBMemmapDataset pour eviter 11.73 GB en RAM

### 5.3 Pipeline RF-UAVNet (Raw IQ)
```
.dat -> load_raw_iq() [complex64, 120M]
     -> normalize_minmax() [per-file, per-channel [0,1]]
     -> segment_signal() [20ms = 1.2M, pas d'overlap]
     -> downsample_iq() [1.2M -> 10K via np.interp, facteur 120x]
     -> RFUAVNet.forward() [1D CNN: R-Unit + 4 G-Units + Multi-GAP]
```
- **Features** : (2, 10000) raw IQ downsampled
- **Architecture** : R-Unit(2->64,k5,s5) + 4x G-Unit(64->64,k3,s2,groups=8) + Multi-GAP(5x64=320) + FC(320->7)
- **Params** : 4,615 (~18 KB)
- **Training** : 120 epochs, batch=512, lr=0.001, Adam(weight_decay=1e-4), ReduceLROnPlateau, Early stopping patience=10

---

## 6. MODELES ET RESULTATS

### 6.1 Performance globale

| Modele | Features | Accuracy | F1 | p50 (ms) | p95 (ms) | Taille |
|--------|----------|----------|-----|----------|----------|--------|
| **VGG16** | Spectrogram | **89.6%** | 0.897 | 2.26 | — | 56.8 MB |
| **SVM** | PSD | **83.5%** | 0.830 | 12.86 | — | 53.8 MB |
| **ResNet50** | Spectrogram | **79.0%** | 0.792 | 6.83 | — | 92.7 MB |
| **RFUAVNet** | Raw IQ | **47.6%** | 0.458 | 1.30 | — | 0.06 MB |

> NB 031 re-run session 8 (2026-05-08). RFUAVNet 47.6% avec Adam(lr=1e-3) — bottleneck 120x downsampling confirme.

### 6.2 Performance par classe (F1-score) — (resultats pre-retraining, a mettre a jour)

| Drone | SVM | VGG16 | ResNet50 | RFUAVNet | Difficulte |
|-------|-----|-------|----------|----------|------------|
| MIN | 1.00 | 1.00 | 0.99 | 0.92 | Facile (Wi-Fi, EIRP faible) |
| DIS | 0.98 | 0.96 | 0.87 | 0.71 | Facile (Wi-Fi, Parrot) |
| PHA | 0.90 | 0.86 | 0.80 | 0.55 | Moyen (Lightbridge) |
| INS | 0.88 | 0.83 | 0.77 | 0.52 | Moyen (Lightbridge) |
| AIR | 0.80 | 0.74 | 0.68 | 0.45 | Difficile (OcuSync 3.0) |
| MAV | 0.73 | 0.70 | 0.60 | 0.38 | Difficile (OcuSync 1.0) |
| MA1 | 0.62 | 0.60 | 0.50 | 0.17 | Tres difficile (OcuSync 2.0) |

### 6.3 Analyse statistique (notebook 031)
- **McNemar** : toutes les paires significativement differentes (p < 0.05)
- **Bootstrap 95% CI** : SVM [0.817, 0.840], VGG16 [0.773, 0.798], ResNet50 [0.723, 0.752], RFUAVNet [0.513, 0.544]
- **Cohen's Kappa** : SVM (0.80), VGG16 (0.75), ResNet50 (0.69), RFUAVNet (0.45)
- **Hard samples** (aucun modele correct) : 150 segments (2.67%), concentres sur MA1 et MAV (famille OcuSync)

### 6.4 Confusion principale
La confusion MA1/MAV/AIR (famille OcuSync DJI) est **persistante sur tous les modeles**. Ces 3 drones utilisent des variantes du meme protocole FHSS proprietaire avec des signatures spectrales tres proches.

---

## 7. ANALYSE PAR MODELE

### 7.1 SVM (PSD) — 83.5%

**Forces** :
- Deuxieme meilleur modele (83.5%), meilleure accuracy parmi les approches non-DL
- PSD capture efficacement les signatures spectrales a pleine resolution (1024 bins sur 60 MHz)
- Pas de perte d'information spectrale (Welch sur 1.2M samples natifs)
- Simple, interpretable, pas de GPU requis

**Faiblesses** :
- Inference la plus lente (8.81 ms) — noyau RBF sur vecteurs 1024-dim
- Modele le plus lourd apres ResNet50 (53.8 MB de support vectors)
- Pas de predict_proba natif (fallback decision_function + softmax approximatif)
- SVM RBF avec C=1.0 par defaut — pas d'optimisation d'hyperparametres documentee

**Axes d'amelioration potentiels** :
- Grid search / Bayesian optimization sur C, gamma
- Feature selection / PCA sur les 1024 bins PSD
- Tester d'autres noyaux (polynomial, chi2)
- Tester d'autres features spectrales (cepstrum, spectral moments)

### 7.2 VGG16 (Spectrogram) — 89.6% — Meilleur modele

**Forces** :
- Meilleure accuracy (89.6%) et bon compromis vitesse (1.53 ms)
- Transfer learning efficace (ImageNet features figees)
- Spectrogramme RGB capture bien la structure temps-frequence

**Faiblesses** :
- Overfitting residuel (train acc ~95% vs test 89.6%, gap reduit apres corrections LR/scheduler)
- FC layer unique (25088 -> 7) — peut-etre trop simple
- Spectrogram extrait uniquement les frequences positives (perte de la moitie du spectre complexe)
- `cm.get_cmap('viridis')` deprecated

**Axes d'amelioration potentiels** :
- Fine-tuning des dernieres couches VGG (pas seulement FC)
- Regularisation (dropout, weight decay)
- Data augmentation sur spectrogrammes (SpecAugment, time/freq masking)
- Spectrogramme bilateral (frequences positives ET negatives)
- Tester d'autres architectures (EfficientNet, MobileNet)

### 7.3 ResNet50 (Spectrogram) — 79.0%

**Forces** :
- Architecture plus profonde que VGG16 (skip connections)

**Faiblesses** :
- Pire que VGG16 malgre plus de params (79.0% vs 89.6%)
- FC(2048 -> 7) apres AdaptiveAvgPool2d (corrige session 6, etait FC(100352->7) = 700K params)
- Taille excessive (92.7 MB)
- Inference 4x plus lente que VGG16 (6.61 ms vs 1.53 ms)
- Overfitting severe (train acc ~95% vs test 79.0%)

**Diagnostic** : Apres correction AdaptiveAvgPool2d (session 6), le ratio params est comparable a VGG16. Accuracy 79.0% post-correction.

**Axes d'amelioration potentiels** :
- Regularisation aggressive (dropout 0.5, weight decay)
- Reduire le nombre de couches trainables
- Fine-tuning progressif (unfreeze couches par couches)

### 7.4 RF-UAVNet (Raw IQ) — 47.6% — PROBLEMATIQUE

**Forces** :
- Ultra-leger (18 KB, 4,615 params)
- Inference la plus rapide (1.44 ms)
- Architecture originale (Multi-GAP multi-echelle)

**Faiblesses** (voir section 11 pour details complets) :
- Accuracy 47.6% avec Adam(lr=1e-3, weight_decay=1e-4) — bottleneck 120x downsampling confirme
- Downsampling 120x detruit 99.17% du contenu spectral (bottleneck fondamental)
- Normalisation min-max supprime l'info de puissance EIRP
- Optimizer aligne sur papier (Adam) depuis session 7, pas de gain significatif vs SGD
- Le papier original ne fait PAS de downsampling (10K samples natifs a 40 MHz = 0.25 ms)
- Underfitting confirme : train ~52%, plateau epoch ~60

**Quick wins appliques** : LR 0.01→0.001 (session 6), SGD→Adam (session 7), ES patience 5→10. max_epochs 200 et gradient clipping juges inutiles.

---

## 8. WORKFLOW NOTEBOOKS

### Phase 1 : Exploration (local, necessite dataset)

| # | Notebook | Objet | Resultats cles |
|---|----------|-------|----------------|
| 001 | exploration_general | Inventaire dataset, distributions, qualite | 195 fichiers, 3 manquants, 52 fichiers IQ hors [-1,1] |
| 002 | exploration_temporal | Features temporelles, stationnarite | 20 features, ADF 98% stationnaire a 20ms, ZCR discriminant |
| 003 | exploration_frequentiel | PSD par drone/mode/interference | BW_99% 17-25 MHz, PSD +7-8% robust vs temporel, MA1/MAV inseparables |
| 004 | exploration_frequentiel_adv | STFT, spectrogrammes CNN, FHSS | Config recommandee: nfft=512, Hamming, hop=128, 224x224 |
| 006 | downsampling_analysis | Impact downsampling IQ | Optimal 350K (3.4x), 120x catastrophique |

### Phase 2 : Preprocessing (local, CLI pipeline)

| # | Notebook | Input | Output | Notes |
|---|----------|-------|--------|-------|
| 015 | preprocessing | Raw IQ | PSD + spectro + IQ + test_samples | Wraps pipeline.py + export_samples.py |

> NB 012/013/014 deleted (session 8): 100% redundant with CLI pipeline.py

### Phase 3 : Training (Google Colab GPU)

| # | Notebook | Modele | Epochs | Accuracy |
|---|----------|--------|--------|----------|
| 021 | training_svm | SVM RBF | N/A | 83.5% |
| 022 | training_cnn | VGG16 + ResNet50 | max 120 | 89.6% / 79.0% |
| 023 | training_rfuavnet | RF-UAVNet (Adam) | max 120 | 47.6% |

### Phase 4 : Comparison (Google Colab)

| # | Notebook | Contenu |
|---|----------|---------|
| 031 | model_comparison | 4 modeles, McNemar, bootstrap CI, Cohen's Kappa, error analysis |

**Note** : Notebook 005 manquant dans la numerotation (004 -> 006).

---

## 9. DOCUMENTATION EXISTANTE

| Document | Lignes | Contenu |
|----------|--------|---------|
| README.md | 104 | Vue d'ensemble, installation, usage, references |
| docs/LEAKAGE_PROOF.md | 386 | Preuve rigoureuse de data leakage dans RFClassification (4 evidences) |
| docs/REFERENTIEL_DRONEDETECT_RFCLASSIFICATION.md | 1140 | Referentiel technique complet des 3 modeles |
| docs/data_collection_methodology.md | 158 | Methodologie d'acquisition SDR/drones |
| docs/diagrams/*.dot | 4 fichiers | Pipelines Graphviz (global, SVM, CNN, RFUAVNet) |
| interface/static/glossary.json | 31 termes | Glossaire technique RF/ML |

### Data leakage (LEAKAGE_PROOF.md)
Le repo de reference RFClassification (github.com/tryph0n/RFClassification) contient un leakage severe :
1. Normalisation globale avant split (min/max sur train+test)
2. Split segment-level (segments adjacents correlation 0.9-0.99 dans train ET test)
3. Perte des frontieres fichier (reshape detruit le tracking)
4. **Impact** : accuracy gonflee de 99.8% -> ~56% corrige (delta ~43.8%)

Le projet mldrone corrige ces problemes : normalisation per-file + StratifiedGroupKFold file-level.

---

## 10. PROBLEMES IDENTIFIES (tous modeles)

### 10.1 Problemes de code

| # | Severite | Fichier | Probleme |
|---|----------|---------|----------|
| 1 | HAUTE | preprocessing.py:199-210 | Code indente orphelin (potentiel SyntaxError a l'import) |
| 2 | HAUTE | models.py + interface/models/ | Code duplique (VGG16FC, ResNet50FC, RFUAVNet definis 2 fois avec differences) |
| 3 | MOYENNE | features.py | `cm.get_cmap('viridis')` deprecated matplotlib >=3.7 |
| 4 | MOYENNE | features.py | Import inutile `matplotlib.mlab` |
| 5 | MOYENNE | features.py:compute_spectrogram | Extrait uniquement freq positives (perd moitie spectre complexe) |
| 6 | MOYENNE | inference_view.py | `allow_pickle=True`, `weights_only=False`, `pickle.load` sans sandbox |
| 7 | MOYENNE | Plusieurs fichiers | `sys.path.insert(0, ...)` fragile pour les imports |
| 8 | FAIBLE | inference_view.py | Fonctions de visualisation dupliquees (aussi dans visualization_service.py) |
| 9 | FAIBLE | model_comparison_view.py | Instancie les 3 modeles PyTorch meme si un seul est selectionne |
| 10 | FAIBLE | model_viz_service.py | Fichiers temp non nettoyes (`delete=False`) |

### 10.2 Problemes de donnees/fichiers

| # | Severite | Probleme |
|---|----------|----------|
| 1 | MOYENNE | Doublons modeles : `rfuavnet_iq (1).pth`, `vgg16_cnn (1).pth` (espaces dans noms) |
| 2 | MOYENNE | Duplication modeles : copies dans /models/ ET /interface/media/models/ (204 MB redondants) |
| 3 | MOYENNE | .env committe dans le repo (contient chemin local) |
| 4 | FAIBLE | Artefact Windows : `model_comparison_results.csv:Zone.Identifier` |
| 5 | FAIBLE | Repertoires vides : figures/training_svm/, figures/training_rfuavnet/ |
| 6 | FAIBLE | ~6.4 GB de features NPZ dans notebooks/data/ |

### 10.3 Problemes methodologiques

| # | Modele | Probleme |
|---|--------|----------|
| 1 | SVM | Pas d'optimisation hyperparametres (C=1.0, gamma='scale' par defaut) |
| 2 | VGG16/ResNet50 | Overfitting (train ~90-95% vs test 73-78%) |
| 3 | ResNet50 | ~~FC(100352->7) surdimensionnee~~ CORRIGE : AdaptiveAvgPool2d + FC(2048->7) (session 6) |
| 4 | RF-UAVNet | Downsampling 120x destructif (voir section 11) |
| 5 | RF-UAVNet | CLARIFIE : papier (Huynh-The 2022) utilise Adam, repo IQTLabs utilise SGD (Adam commente). Notre impl suit le repo (SGD), LR reduit 0.01→0.001 |
| 6 | Tous | Pas de tests unitaires |
| 7 | Tous | Segmentation sans overlap (perte mineure aux frontieres) |
| 8 | CNN | Spectrogram unilateral (freq positives uniquement) |

---

## 11. INVESTIGATION RF-UAVNet (4 agents specialises)

### 11.1 Hypothese utilisateur
> Les mauvais scores de RF-UAVNet sont-ils dus a un mauvais overlap/echantillonnage du signal ?

### 11.2 VERDICT : OUI, principalement l'echantillonnage (downsampling 120x)

### 11.3 Agent Downsampling (analyse spectrale)

**Frequence effective** : Fs_eff = 60 MHz / 120 = **500 kHz**
**Nyquist requis** : 2 x 25 MHz = **50 MHz**
**Violation** : facteur **112x** en dessous de Nyquist

- 99.17% du contenu spectral perdu ou replie par aliasing
- 56 repliements spectraux superposes dans [0, 250 kHz]
- `np.interp` : attenuation ~0 dB a 250 kHz (aucun filtrage anti-aliasing)
- Notebook 006 : accuracy 88.1% a 350K samples vs 68.3% a 10K (Random Forest)
- Detection de bursts : 183,041 (baseline) vs 1,782 (10K) = 1% des bursts

**Facteur de downsampling optimal** : 350,000 samples (3.4x, Fs_eff=17.5 MHz)

### 11.4 Agent Overlap (segmentation)

- Impact overlap : +2 a +5% seulement
- 19,478 segments / 4,615 params = ratio 4.22 (correct pour un CNN)
- Perte aux frontieres : ~0.1% du contenu informatif (negligeable pour FHSS ~1000 hops/segment)
- L'overlap avec split file-level est sur (pas de leakage)
- **Le goulot d'etranglement est le downsampling, pas l'overlap**

### 11.5 Agent Papier vs Implementation

**Architecture** : conforme au papier (R-Unit, G-Units, Multi-GAP)

**Ecarts critiques** :

| Aspect | Papier (DroneRF) | mldrone (DroneDetect V2) |
|--------|------------------|--------------------------|
| Dataset | DroneRF (226K segments, 3 drones) | DroneDetect V2 (19K segments, 7 drones) |
| Canaux input | High freq / Low freq (2 bandes RF) | I/Q (Re/Im d'un signal) |
| Downsampling | **AUCUN** (10K natifs @ 40 MHz = 0.25 ms) | **120x** (1.2M -> 10K, interpolation lineaire) |
| Normalisation | Min-max global (leakage) | Min-max per-file (correct) |
| Optimizer | Adam (papier Huynh-The 2022) | Adam(lr=1e-3, wd=1e-4) — aligne sur papier (session 7) |
| Batch size | 128 | 512 |
| Validation | 10-fold CV segment-level (leakage) | 80/20 file-level (correct) |

**Decomposition de la chute** : 99.8% -> ~65% (correction leakage) -> ~51.3% (downsampling + dataset + LR/ES tuning)

### 11.6 Agent Physique du signal

- **Aliasing** : 56 repliements, SNR degrade de -17.5 dB
- **FHSS** : OcuSync 3.0 = 3.67 points/hop (irresolvable), Lightbridge = 14.3 points/hop (limite)
- **OFDM** : 2 points/symbole, 0 sous-porteuse resolue — structure completement detruite
- **Min-max norm** : supprime EIRP (19-26 dBm = facteur 5x), sensible aux outliers
- **Z-score preserverait** les rapports relatifs de puissance

**Seules les signatures basse-frequence survivent** : enveloppe de puissance, burst timing, DC offset, I/Q imbalance statique.

### 11.7 Decomposition du gap RF-UAVNet

| Facteur | Impact estime | Priorite |
|---------|---------------|----------|
| Downsampling 120x sans filtre AA | -20 a -30 pts | **CRITIQUE** |
| Correction data leakage vs papier | -35 a -40 pts (vs 99.8% original) | Deja corrige (correct) |
| Changement dataset (7 classes, interferences) | -5 a -10 pts | Structurel |
| Min-max norm (perte EIRP) | -3 a -5 pts | MOYEN |
| Optimizer SGD vs Adam / LR | -2 a -5 pts | FAIBLE (Adam aligne, pas de gain significatif vs SGD+LR 0.001) |
| Absence d'overlap | -2 a -5 pts | FAIBLE |

---

## 12. RECOMMANDATIONS GLOBALES (EN ATTENTE)

> Ce plan est mis de cote. Le focus est sur la preparation de la certification. Les informations sont conservees pour reference.

### 12.1 RF-UAVNet (priorite haute — sous-performe massivement)

| # | Action | Gain estime | Effort |
|---|--------|-------------|--------|
| 1 | Passer a 350K samples (downsampling 3.4x) | **+18 pts** | Adapter architecture Multi-GAP |
| 2 | Utiliser `scipy.signal.decimate` (filtre AA) | Prerequis du #1 | 1 ligne |
| 3 | Remplacer min-max par Z-score | +3-5 pts | 1 appel |
| 4 | Tester Adam au lieu de SGD | +2-5 pts | 1 ligne |
| 5 | Ajouter overlap 50% | +2-5 pts | Modifier segment_signal() |

### 12.2 SVM (priorite moyenne — deja bon, peut etre ameliore)

| # | Action | Gain potentiel |
|---|--------|----------------|
| 1 | Grid search C, gamma | +2-5% |
| 2 | Feature selection / PCA sur PSD | +1-3% + reduction taille modele |
| 3 | Tester features supplementaires (cepstrum, moments spectraux) | +2-5% |

### 12.3 CNN VGG16/ResNet50 (priorite moyenne — overfitting a corriger)

| # | Action | Gain potentiel |
|---|--------|----------------|
| 1 | Regularisation (dropout, weight decay) | +2-5% |
| 2 | Fine-tuning progressif (unfreeze couches) | +3-5% |
| 3 | Data augmentation (SpecAugment) | +2-5% |
| 4 | Spectrogramme bilateral (freq positives + negatives) | +1-3% |
| 5 | ~~ResNet50 : ajouter AdaptiveAvgPool2d avant FC~~ DONE (session 6) | +3-5% |
| 6 | Tester EfficientNet / MobileNet | Variable |

### 12.4 Qualite de code (priorite basse mais important pour defense)

| # | Action |
|---|--------|
| 1 | Corriger SyntaxError preprocessing.py:199-210 |
| 2 | Eliminer duplication models.py / interface/models/ |
| 3 | Supprimer doublons modeles (fichiers avec espaces) |
| 4 | Ajouter tests unitaires |
| 5 | Remplacer sys.path.insert par imports propres |
| 6 | Corriger deprecated cm.get_cmap |
| 7 | Nettoyer .env du repo |

---

## 13. HISTORIQUE DES ACTIONS

| Date | Action | Statut |
|------|--------|--------|
| 2026-02-19 | Reconnaissance projet (3 agents) + Investigation RF-UAVNet (4 agents) | ✅ |
| 2026-03-09 | MACRO 1: Recherche amelioration RF-UAVNet + MACRO 2: Audit notebooks | ✅ |
| 2026-04-28 | Reorientation certification. Validation problemes (code 10/10, data 4/6, notebooks). Gap analysis. | ✅ |
| 2026-04-29 | MACRO A (code/file cleanup) + MACRO B (NB 006/004/003/022 corrections) | ✅ |
| 2026-05-05 | MACRO D (reviews post-re-run) + MACRO E planification + execution (features, pipeline, split, NB 021) | ✅ |
| 2026-05-06 | Migration NB 022/023/031 (section 22). CLAUDE.md 10 modifs. Upload Scaleway. Pipeline 390 fichiers. | ✅ |
| 2026-05-07 | E-EXPLOR-ASSESS. Retraining results. Upload/Colab support. RF-UAVNet quick wins. FIX-RESNET50FC. | ✅ |
| 2026-05-08 | FIX-MODELS-DIR chain. MACRO AQ (14 tasks). F-EXPLOR-DEPRECATE. FIX-FORMAT. FIX-STREAMLIT-HOME. MOVE-TEST-SAMPLES. NB 023 Adam retrain + NB 031 re-run. MACRO-EXPLORE (content audit 9 NB + 4 diagrams + Streamlit glossary). AUDIT-NB023/031-PDF. FIX-NB023-METRICS/PATH. FIX-INFERENCE-PROTOCOL/EXPLAIN. FIX-HOME-RFCLASS-LINK. 4 diagram rewrites (GLOBAL/SVM/CNN/RFUAVNET). State persistence. | ✅ |

---

## 15. RF-UAVNet AMELIORATION (EN ATTENTE - A REVISITER APRES CERTIFICATION)

> Ce plan est mis de cote. Le focus est sur la preparation de la certification. Les informations sont conservees pour reference.

### Verdict verification : APPROVE WITH CHANGES

### Phase 1 : Quick wins (gain realiste +10-17 pts -> 62-70%)

| # | Action | Detail | Status |
|---|--------|--------|--------|
| 1 | scipy.signal.decimate | Remplacer np.interp | **INVALIDE** (np.interp surpasse decimate, ratio 120x est le bottleneck) |
| 2 | Passer a 40K samples + AdaptiveAvgPool1d(1) | assert target % 80 == 0 | TODO |
| 3 | Adam optimizer (lr=1e-3) | Remplacer SGD | **REVISE** : SGD garde, LR 0.01→0.001 applique (session 6, +2.5pts) |
| 4 | Z-score au lieu de min-max | Preserver EIRP | TODO |

### Phase 2 : Optimisation (-> 67-75%)

| # | Action |
|---|--------|
| 5 | Tester 100K samples (necessite DataLoader incremental, RAM >14 GB) |
| 6 | Data augmentation RF (time shift, AWGN, gain scaling) |
| 7 | Class weights pour OcuSync |
| 8 | Dropout(0.3) avant FC (seulement apres resolution underfitting) |
| 9 | Overlap 50% segmentation |

### Phase 3 : Architecture avancee (-> 75-82%)

| # | Action |
|---|--------|
| 10 | Strided conv apprenantes (remplacement du downsampling fixe) |
| 11 | Multi-scale branches |
| 12 | Complex-valued convolutions |
| 13 | FC intermediaire 320->128->7 |

### Contrainte realiste
75-82% est le plafond probable avec RF-UAVNet sur DroneDetect V2 (file-level split, 7 classes dont 3 OcuSync quasi-indistinguables). La confusion OcuSync persistera meme a 40K (Fs_eff = 2 MHz ne resout que ~3% de la bande FHSS).

### References cles
- synthesis_rfuavnet_plan.md (plan complet)
- verification_rfuavnet_plan.md (rapport de verification)
- research_rfuavnet_improvements.md (recherche litterature)
- analysis_rfuavnet_current.md (analyse code existant)
- Tous dans .shared_knowledge/

---

## 16. AUDIT QUALITE SCIENTIFIQUE NOTEBOOKS

### Classement (plus problematique en premier) - APRES AUDIT APPROFONDI

| Rang | Notebook | Note | Problemes critiques |
|------|----------|------|---------------------|
| 1 | 006_downsampling_analysis | **3/10** | 6 contradictions internes, 6 bugs (data leakage CV, algo selection inverse), correlation "0.95" vs 0.66-0.71 |
| 2 | 004_exploration_frequentiel_advanced | **4/10** | Algo FHSS detecte du bruit (Disco WiFi classe FHSS), conversion dB fausse, 18 configs annoncees/3 testees |
| 3 | 003_exploration_frequentiel | **4.5/10** | Kurtosis inf=bug logique (nan>0=False->0), affirmations physiques fausses, "42% SVM errors" non source |
| 4 | 023_training_rfuavnet | 6/10 | Downsampling 120x malgre 006, erreur "82% fewer params" (c'est 99.98%) |
| 5 | 002_exploration_temporal | 6.5/10 | PAPR/OFDM ignore PHA=WiFi, variabilite replicas sous-estimee |
| 6 | 022_training_cnn | 6.5/10 | Overfitting severe non diagnostique, LR 0.01 trop eleve |
| 7 | 014_preprocessing_iq | 7/10 | Min-max [0,1] detruit symetrie IQ |
| 8 | 001_exploration_general | 7.5/10 | power_dbm mal etiquetee, variabilite replicas sous-estimee |
| 9 | 031_model_comparison | 7.5/10 | Pas d'analyse causes profondes des ecarts |
| 10 | 013_preprocessing_spectrogram | 8/10 | RAS |
| 11 | 012_preprocessing_psd | 8/10 | RAS |
| 12 | 021_training_svm | 8/10 | RAS |

### References audits
- 4 problemes critiques identifies (correlation 006, kurtosis 003, hop rates 004, downsampling 120x) — all corrected in MACRO B
- Audits approfondis : `.shared_knowledge/audit_deep_{003,004,006}.md`
- Audit initial : `.shared_knowledge/audit_notebooks.md`

---

## 17. VALIDATION DES PROBLEMES IDENTIFIES (2026-04-28)

### 17.1 Problemes de code — 10/10 CONFIRMES (all corrected in MACRO A/B)

### 17.2 Problemes data/fichiers — 4 confirmes, 1 invalide (.env not tracked), 1 partiel (3.8 GB not 6.4 GB)

### 17.3 Issues critiques notebooks — 7/7 CONFIRMEES avec outputs reels (correlation 006, kurtosis 003, algo inverse 006, hop rates/protocoles/configs/dB 004). All corrected in MACRO B.

### 17.4 Issues methodologiques — 7/8 confirmees (SVM no HPO, CNN overfitting, LR trop eleve, ResNet50 FC oversized, RFUAVNet underfitting, pas de tests, overlap). SGD vs Adam NUANCE (gain 3-5pp estime). ResNet50 FC CORRIGE session 6.

### 17.5 DECOUVERTE CLE : np.interp n'est PAS le probleme

| Target | Interpolation (np.interp) | Decimation (scipy) |
|--------|--------------------------|-------------------|
| 350K (3.4x) | 0.878 | 0.881 |
| 150K (8x) | **0.828** | 0.686 |
| 40K (30x) | **0.803** | 0.731 |
| 10K (120x) | 0.683 | 0.733 |

L'interpolation surpasse la decimation aux niveaux intermediaires. Le probleme est le ratio 120x, pas la methode. Les corrections proposant de remplacer np.interp par scipy.signal.decimate sont invalidees.

---

## 18. GAP ANALYSIS CERTIFICATION (2026-04-28)

### 18.1 Slides existantes (8 slides)

| # | Titre | Contenu |
|---|-------|---------|
| 1 | RF Drone Classification | Titre |
| 2 | Dataset & Pipeline | DroneDetect V2, BladeRF, 7 classes, pipeline |
| 3 | Resultats Globaux | Radar chart, bar chart accuracy |
| 4 | SVM | Performance par classe |
| 5 | Solutions Deep Learning | VGG16, ResNet50, RF-UAVNet |
| 6 | Demo Streamlit | Transition demo live |
| 7 | Impact & Applications | Securite publique, deploiement |
| 8 | Conclusion & Perspectives | Realisations, prochaines etapes |

### 18.2 Exigences certification vs couverture

| Exigence | Statut | Priorite |
|----------|--------|----------|
| Resultats ML/DL + monitoring (Streamlit) | COUVERT | - |
| Impact reel / perspectives | COUVERT | - |
| Demo produit final | COUVERT | - |
| Cadrage probleme + role data science | PARTIEL | HAUTE |
| Technologies utilisees | PARTIEL | MOYENNE |
| Extraction/nettoyage donnees | ABSENT | HAUTE |
| Objectifs, retroplanning, budget | **ABSENT** | **BLOQUANT** |
| Conformite RGPD | **ABSENT** | **BLOQUANT** |
| Leadership bout en bout | PARTIEL | MOYENNE |

### 18.3 Notebooks et certification

- Les notebooks ne sont PAS un livrable obligatoire (seuls dossier + diaporama sont evalues)
- Phase exploratoire implicitement requise (competences C1, C5, C6)
- Le candidat doit pouvoir expliquer tout ce qui est dans les notebooks en Q&A
- Un diaporama dedie est obligatoire (les PDF de notebooks ne sont pas un diaporama)

### 18.4 Donnees sur Scaleway

- Donnees brutes sur bucket Scaleway (France)
- Conditions : BLUE, WIFI, CLEAN, BOTH
- Pas de signal "background only" (sans drone)
- Deploiement Scaleway possible si necessaire

---

## 19. PLAN D'EXECUTION CORRECTIONS (2026-04-28)

> Approuve par l'utilisateur. Execution NON lancee -- attente de "go" explicite.

### 19.1 Reponses utilisateur (Q1-Q4)

- **Q1 Re-run notebooks** : OUI, sur Colab ou en local -> corrections code + texte possibles
- **Q2 Slides manquantes** : Pas d'info retroplanning/budget/RGPD -> creer contenant, contenu apres validation
- **Q3 Dossier projet** : Pas de dossier imprime, examen a distance
- **Q4 Spectrogram bilateral** : Documenter comme perspective (pas de correction, cout trop eleve)

### 19.2 Taches (ordre sequentiel obligatoire)

**MACRO A: Code/File cleanup** — DONE. Python code cleanup (preprocessing.py, features.py, model_viz_service.py) + file cleanup (Zone.Identifier, dirs vides, __pycache__). Macro-verified.

**MACRO B: Notebook corrections** — DONE. NB 006 (1 REMOVE, 5 FIX), NB 004 (2 REMOVE, 6 FIX), NB 003 (3 FIX), NB 022 (LR, AdaptiveAvgPool2d, ReduceLROnPlateau). All WHAT-comments cleaned. Macro-verified (2 Opus parallel).

| # | Tache | Statut | Complexite | Agent | Detail |
|---|-------|--------|------------|-------|--------|
| **MACRO C: Slides + Dossier** |||||
| 23 | ATOMIZE task 15 | PENDING | S | Opus planning | Decomposer L en sous-taches S/M, validation utilisateur |
| 15 | Slides + Dossier structure | PENDING | L | Opus builder(s) | 5 slides manquantes + structure dossier |
| 24 | MACRO-VERIFY C | PENDING | M | Opus verifier | Couverture certification >= 90% |

### 19.3 References et decisions cles

- Assessments notebooks : `.shared_knowledge/assessment_nb_{003,004,006}.md` + `validation_training_notebooks.md` (all corrections applied)
- 69 corrections cataloguees (12 CRITICAL, 13 HIGH, 27 MEDIUM, 17 LOW) — all applied or invalidated
- np.interp N'EST PAS le probleme : ratio 120x est le facteur dominant, corrections C1/C26/C27 INVALIDEES
- Verification rules codifiees dans CLAUDE.md global (builder+verifier per task, macro-verifier per macro, no Haiku)

### 19.7 Progression sessions

Session 1 (2026-04-28): Validation 69 corrections, gap analysis certification, plan d'execution approuve.

Session 2 (2026-04-29 -> 2026-05-02): MACRO A DONE (code+file cleanup). MACRO B DONE (NB 006/004/003/022 corriges, macro-verified 2 Opus). Analyse technique section 20 (conclusions integrees dans corrections, supprimee de SUIVI).

Session 3 (2026-05-05/06): MACRO D DONE (reviews NB 003/004/006, 4 corrections mineures). MACRO E DONE (features.py corrige, pipeline.py ameliore, splitting.py cree, NB 021 migre + pilot valide). CLAUDE.md global 10 modifications + Deep Analysis Workflow. Plan migration NB 022/023/031 redige et valide.

Session 4 (2026-05-06): Migration NB 022/023/031 COMPLETE (section 22). Pipeline 390 fichiers lance. Upload Scaleway valide (section 23). Ruff cleanup all notebooks + data_loader.py.

Session 5 (2026-05-07):

E-EXPLOR-ASSESS DONE:
- Etat des lieux 8 notebooks exploration (001-006, 012-014)
- 5 KEEP (001-006), 3 DEPRECATE (012/013/014 — supercedes par pipeline.py)
- Taches de nettoyage creees: F-EXPLOR-DEPRECATE, F-EXPLOR-FIX-001/002/004, F-EXPLOR-RERUN

FIX-CR DONE:
- classification_report target_names fix dans NB 021/022/023 (compatibilite donnees partielles)

Upload infrastructure DONE (U-1 a U-5):
- storage.py: upload_models, upload_split, multipart 100MB chunks pour gros fichiers
- upload.py CLI: --features --models --split --all
- Nouveau bucket `mldrone-artefacts` pour artefacts (separe du bucket raw data `mlops-data`)
- Features uploadees sur Scaleway apres pipeline

COLAB support DONE:
- 4 notebooks training (021-023, 031) avec Colab Secrets S3, inline splitting, download/upload conditionnel

Resultats retraining (390 fichiers, 4 conditions):
- NB 021 SVM: 83.5% accuracy (Colab valide)
- NB 022 CNN: VGG16 89.6%, ResNet50 79.0% (Colab GPU)
- NB 023 RF-UAVNet: 48.8% (investigue — bottleneck 120x downsampling, pas un bug)
- RF-UAVNet investigation: 4 agents Opus confirment architecture fidele au papier, ecart explique par file-level split + 120x downsampling

Taches creees:
- F-EXPLOR-DEPRECATE, F-EXPLOR-FIX-001/002/004, F-EXPLOR-RERUN, F-CLI-DOC

Session 6 (2026-05-07):

RF-UAVNet quick wins:
- LR 0.01→0.001, ES patience 5→10 : accuracy 48.8% → 51.3% (+2.5pts)
- Analyse critique : plateau epoch ~60, underfitting confirme (train ~52%)
- max_epochs 200 et gradient clipping juges inutiles (plateau + Adam redondant)
- Documentation ajoutee dans NB 023 (hyperparams modifs, tradeoff downsampling, results summary)
- Agent correcteur : builder avait change SGD→Adam par erreur, corrige

FIX-RESNET50FC DONE:
- AdaptiveAvgPool2d + Linear(2048) applique dans NB 031, models.py, interface/models/cnn.py
- 4 copies ResNet50FC desormais identiques (NB 022 etait deja correct)

Investigation optimizer:
- Papier (Huynh-The 2022) utilise Adam, repo IQTLabs utilise SGD (Adam commente)
- Notre implementation suit le repo (SGD), LR modifie 0.01→0.001 (justifie)
- Documentation data leakage : 16 documents existants, README seul fichier public sans mention

Exploration NB 031:
- Metadonnees a exporter identifiees (confusion matrices, per-class metrics, statistical tests)
- Test samples Streamlit : 42 fichiers .npz a generer, format documente

Taches creees : NB031-META, NB031-SAMPLES, NB023-OPTIM-DOC, README-CREDIT
SUIVI_PROJET.md : nettoyage ~40% (sections obsoletes comprimees/supprimees)

Session 6 suite (2026-05-07):

NB031-META DONE:
- 3 cellules export ajoutees (confusion_matrices.json, per_class_metrics.csv, statistical_tests.csv)
- Upload modifie pour inclure *.json + *.csv (Colab + storage.py)
- Verificateur PASS. Minor fixes: import json deplace en cell-4, bootstrap_results cache (pas de recalcul)

FIX-RESNET50FC DONE (rappel session 6 debut):
- AdaptiveAvgPool2d + Linear(2048) dans NB 031, models.py, interface/models/cnn.py
- 4 copies ResNet50FC identiques

E-PREPROC DONE:
- NB 012/013/014 = 100% redondants avec CLI pipeline.py
- Design NB-PREPROC: notebook local wrappant run_pipeline(), config exposee, PAS Colab
- CLI export_samples: tous segments test par (drone, condition), pas de parametre N
- test_samples = post-processing, appele depuis NB preprocessing en derniere cellule

Analyse split multi-axes (3 agents Opus):
- Couverture complete (78 combos) et 70/15/15 mutuellement exclusifs (5 repliques)
- Option B (3/1/1 par combo) donne 60/20/20 avec couverture 100% mais split deterministe
- DECISION: garder drone-only (70/15/15), traiter robustesse en post-hoc dans NB 031
- Argument: stratifier par variables non-cibles = domain adaptation, pas evaluation standard
- 28% combos manquantes dans test = normal, interface Streamlit gere gracieusement

Analyse RF-UAVNet optimizer:
- Papier (Huynh-The 2022 IEEE Access) utilise Adam
- Repo IQTLabs/RFClassification utilise Adam (SGD commente)
- Notre NB 023 utilise SGD (du repo) avec LR modifie 0.01->0.001
- Tache NB023-ADAM planifiee: aligner sur papier+repo (Adam lr=1e-3)

Interface Streamlit:
- INTERFERENCE_TYPES hardcode ['CLEAN', 'BOTH'] dans settings.py L78 -> ajouter WIFI+BLUE
- Gestion gracieuse des fichiers manquants (cascade de filtres dynamiques, pas de crash)
- Etats de vol (ON/HO/FY) non geres dans l'interface (pas bloquant)

Volume test_samples estime:
- PSD 24 MB, Spectro 3.5 GB, IQ 468 MB -> total 4 GB trop lourd
- Recommandation: PSD complet + quelques segments spectro/IQ par combinaison (~100-200 MB)

SUIVI nettoye: 1342 -> 934 lignes (-30%). Sections 14, 20 supprimees. Sessions 1-4 comprimees.
Taches 22.7 restructurees en 5 priorites avec chaine de dependances.

Session 7 (2026-05-07):

Design CLI-EXPORT-SAMPLES:
- 1 fichier complet (100 segments) par combo (drone, condition)
- 3 .npz par combo (PSD, spectro, IQ), nouveau nommage sans suffix
- Suffix _20 = 20ms (segment duration), pas n_samples. Fichiers actuels = 200 samples
- Format Streamlit compatible: X (n_samples, ...), y (n_samples,), optional metadata
- INTERFERENCE_TYPES uniquement dans interface/ (settings.py + inference_view.py)

Taches creees: FIX-SETTINGS-INTERFERENCE (3+1), FIX-NB003-HARDCODED, FIX-NB004-HARDCODED, CLI-EXPORT-SAMPLES, NB-PREPROC, MACRO-VERIFY
Analyse segment 10ms vs 20ms: pas prioritaire certification, perspective uniquement

CLI-EXPORT-SAMPLES DONE:
- src/dronedetect/export_samples.py cree (217 lignes)
- 1 fichier complet (100 segments) par combo (drone, condition)
- 3 .npz par combo (PSD, spectro, IQ), nommage sans suffix
- Output: interface/media/test_samples/{CONDITION}/{type}_{CONDITION}_{DRONE}.npz
- 84 fichiers generes (4 conditions x 7 drones x 3 types)
- Verificateur 12/12 PASS

NB-PREPROC DONE:
- notebooks/015_preprocessing.ipynb cree (11 cellules, local, pas Colab)
- Config exposee -> run_pipeline() -> verify -> export_test_samples()
- Fix paths: OUTPUT_DIR = Path("../data/features"), SPLIT_PATH relative
- Verificateur 13/13 PASS
- User re-run: export_samples OK, 84 fichiers generes

FIX-NB003-HARDCODED DONE:
- 3 occurrences ['CLEAN', 'BOTH'] dynamicisees dans cell-15 (interference_types = sorted(df['interference'].unique()))

FIX-NB004-HARDCODED DONE:
- 3 cells (24, 26, 28) dynamicisees (all_conditions, drone_conditions derives des donnees)

Macro-verification: 9/9 PASS (cross-file coherence)

CLEANUP-OLD-SAMPLES DONE:
- 42 anciens fichiers (*_20.npz, *_224x224x3.npz) supprimes dans BOTH/ et CLEAN/
- 2.5 GB recuperes

Regle 21 ajoutee dans CLAUDE.md global: No F-String Alignment Specifiers

Taches planifiees: FIX-FORMAT-NB006, FIX-FORMAT-NB031 (pd.DataFrame au lieu de :>/<)

NB023-ADAM DONE:
- SGD remplace par Adam(lr=1e-3, weight_decay=1e-4) dans NB 023
- CONFIG, markdown, docstrings mis a jour. momentum supprime
- Verificateur 8/8 PASS
- Pret pour retraining Colab

Regle 21 CLAUDE.md: No F-String Alignment Specifiers — DONE + verificateur PASS

Session 8 (2026-05-08) — part 1:

FIX-MODELS-DIR chain DONE:
- FIX-MODELS-DIR-CONFIG: config.py ./models → ./data/models
- FIX-MODELS-DIR-UPLOAD: upload.py Path("models") → Path("data/models")
- FIX-MODELS-DIR-SETTINGS: settings.py MODELS_DIR → data/models
- FIX-MODELS-DIR-DOCS: .env.example + README.md updated
- VERIFY-CLI-PIPELINE-IMPACT: pipeline.py doesn't use MODELS_DIR, no breakage
- CLEANUP-STALE-MODELS: deleted models/, interface/media/models/, stale pkl, Zone.Identifier
- MACRO-VERIFY-MODELS: grep global, zero stale references

Retraining results (NB 023 Adam + NB 031 re-run on Colab):
- NB 023 RF-UAVNet with Adam: 47.6% accuracy (confirms 120x downsampling bottleneck, Adam vs SGD no significant gain)
- NB 031 re-run: VGG16 89.6%, SVM 83.5%, ResNet50 79.0%, RFUAVNet 47.6%

Inference investigation (4 Opus agents):
- No data mismatch, inference pipeline works correctly
- SVM sklearn version warning (1.6.1 train vs 1.7.2 inference) — minor, not causing bad results

FIX-DATA-DIR DONE: settings.py DATA_DIR notebooks/data/features → data/features

F-EXPLOR-DEPRECATE DONE: NB 012/013/014 + copies deleted (9 files total)

FIX-FORMAT-NB006 + FIX-FORMAT-NB031 DONE: f-string alignment specifiers replaced by pd.DataFrame

AUDIT-SRC + AUDIT-INTERFACE DONE (2 Opus agents paralleles):
- src/dronedetect/: dead code (interpolate_2d, iter_download, commented block), FEATURES_DIR dup, all_classes unused, typing old-style
- interface/: 5 dead methods model_viz_service.py, dead config keys settings.py, dead code inference/comparison views, drone_data duplication home_view.py
- Cross-cutting: export_samples.py couples to interface.settings, comparison CSV stale

MACRO AQ plan drafted (14 tasks, all S except macro-verify M)

AUDIT STREAMLIT (3 agents Opus paralleles):
- Page Inference: pas de bug, design par data type (iq→RFUAVNet, psd→SVM, spectro→VGG16/ResNet50)
- Page Comparison: CSV stale (Jan 2026), VGG16 affiche 79.6% au lieu de 89.6%
- Page Home: globalement correcte, VGG16/ResNet50 "frozen" confirme
- 3 emplacements modeles: data/models/ (Mai, source verite), models/ (Jan, ancien), interface/media/models/ (Jan, stale)
- model_comparison_full.pkl dans interface/media/ = orphelin, non reference
- CSV schema NB031 ↔ Streamlit: 100% compatible (8 colonnes identiques)
- models/ (racine) reference par: config.py, upload.py, settings.py, .env.example, README.md

Decision utilisateur: source de verite = data/ pour tout (models, features, split, samples)

Taches planifiees session 7 (statut mis a jour session 8):
- FIX-MODELS-DIR-CONFIG: DONE (session 8)
- FIX-MODELS-DIR-UPLOAD: DONE (session 8)
- FIX-MODELS-DIR-SETTINGS: DONE (session 8)
- FIX-MODELS-DIR-DOCS: DONE (session 8)
- VERIFY-CLI-PIPELINE-IMPACT: DONE (session 8)
- CLEANUP-STALE-MODELS: DONE (session 8)
- MACRO-VERIFY-MODELS: DONE (session 8)
- RE-RUN-NB031: DONE (session 8)
- FIX-STREAMLIT-HOME: DONE (session 8)
- FIX-FORMAT-NB006: DONE (session 8)
- FIX-FORMAT-NB031: DONE (session 8)

Session 8 (2026-05-08) — part 2 (continued after compaction):

AUDIT-NB023-PDF + AUDIT-NB031-PDF DONE:
- NB 023: 2 BLOQUANT incoherences (51.3% reported instead of 47.6%, false +2.5pts improvement claim)
- NB 031: RAS fonctionnel, cosmetic only (kaleido warnings)

FIX-NB023-METRICS + FIX-NB023-PATH DONE:
- 3 occurrences 51.3% corrected to 47.6% in cells 28-29
- Stale path data/sample/test_data/ corrected to data/test_samples/ in cell 5

FIX-INFERENCE-PROTOCOL (#158) DONE:
- st.metric replaced by st.markdown for full-width protocol display (truncation fix)

FIX-INFERENCE-EXPLAIN (#159) DONE:
- st.expander added with CNN softmax + SVM Platt scaling explanation

FIX-HOME-RFCLASS-LINK (#169) DONE:
- RFClassification repo link added to home page warning box

MACRO-EXPLORE (#168) DONE (exploration phase):
- 10 Opus agents parallel: glossary gaps (23 terms), diagrams inventory (4 .dot), inference QA prep, NB 001-004/006/031 content audit
- NB 005 doesn't exist (numbering gap)
- All 4 .dot diagrams had errors (wrong order, wrong params, missing steps)
- Architecture analysis completed -> .doc/knowledge/agent-output/pipeline_analysis.md
- Disk space analysis: re-run on 195 files OK, 390 files needs +183 GB (not feasible locally)

DIAGRAM REWRITES (P1) DONE:
- DIAGRAM-GLOBAL: seg->norm order fixed, 3 branches with different normalizations
- DIAGRAM-SVM: Hamming (not Hanning), noverlap=512 (not 120), fftshift added
- DIAGRAM-CNN: hop=904 (not 128), dB/resize/viridis steps, VGG16/ResNet50 FC details
- DIAGRAM-RFUAVNET: residual connections, multi-scale GAP (5 branches), padding
- 4 verifiers launched (pending results at session end)

KEY DECISIONS:
- 195 vs 390 files: re-run on 195 + caveat (can't download all locally, need +183 GB)
- No new Streamlit page for robustness (integrate in comparison page or keep in NB)
- Old tasks absorbed: F-EXPLOR-FIX-001->CLEAN-NB001, F-EXPLOR-FIX-002->CLEAN-NB002, F-EXPLOR-FIX-004->CLEAN-NB004, F-EPOCH-DOC->GLOSSAIRE-BULK
- MACRO E marked DONE (pipeline migration completed sessions 3-4)

Session 8 (2026-05-08) — part 3 (continued after compaction):

MICRO-FIX DONE: pipeline_rfuavnet.dot 9,991→4,615 params (propagated to SUIVI, NB 023, pipeline_analysis.md, 9 doc/shared_knowledge files)
CLEAN-NB006 (#175) DONE: 52.9%→47.6%, burst detection section removed, recommendations rewritten with key findings + perspectives + caveats
CAVEAT-195 (#176) CANCELLED: User decided to re-run exploration NB on Colab with 390 files instead of adding caveat
MACRO-COLAB-EXPLOR (#183) created: Migrate 5 exploration NB to Colab+Scaleway, user re-runs 390 files, export PDF
GLOSSAIRE-BULK (#177) DONE: 16 terms added (32→48 total), glossary.json
CLEAN-NB031 (#179) DONE: File-level split intro, MA1/MAV RF explanation, WHY McNemar, fallback splitting removed (~170 lines)
CLEAN-NB001 (#178) DONE: ~25 WHAT-comments removed, scope clarification (conditions+states), power_dbm→power_db, dead code removed
CLEAN-NB002 (#180) DONE: ~38 WHAT-comments removed, FR→EN (section 5), unused imports removed, section ordering clarified
CLEAN-NB003 (#181) DONE: Cohen's d caveat added, PHA drone dynamicized, 2 dead Colab cells deleted
CLEAN-NB004 (#182) DONE: NOTEBOOK_NAME fixed, placeholder removed, 2 Colab cells deleted, dead code removed (3 unused vars + spectrograms_for_vis)
NB031-ROBUSTESSE (#161) DONE: Section 13b added — accuracy+F1 per condition (4) and per flight state (3) for 4 models, plotly charts + heatmaps, CSV export
MACRO-COLAB-EXPLOR (#183) REDEFINED: Colab migration abandoned (exploration NB don't need GPU, don't need all 390 files). Replaced by CAVEAT-195 + local re-run (195 files / 2 conditions).
CAVEAT-195 DONE: Data scope caveat added to 5 exploration NB (001-004, 006). Text: "195 files, 2 conditions (CLEAN+BOTH), training NB use complete dataset."
README-CREDIT DONE: Key contribution section (file-level split, data leakage fix) + Credits (IQTLabs, DroneDetect V2, Huynh-The 2022)
F-CLI-DOC (#163) DONE: CLI Commands section in README (pipeline, upload, export_samples)
FIX-NB031-COLAB DONE: Restored try/except ImportError fallback for splitting on Colab
AUDIT-NB031-RERUN (#184) DONE: PDF outputs verified PASS — robustness section present, all metrics correct, kaleido warnings only
NB 004 assessment: DELETE (dead-end, nothing used downstream — STFT params diverge, spectral entropy/occupancy/robustness unused)
NB 001/002/003/006 assessment: TRIM plan validated by user (details in handoff)

TRIM PLAN (validated, not yet executed):
- NB 004: DELETE entirely
- NB 001: Remove cell 29 (duplicate replica), constellation diagram cell 31, fix/remove cell 8 (5/7 drones)
- NB 002: Remove cell 10 (IQ trajectory), cell 19 (raw means), simplify ADF cell 21
- NB 003: Remove cells 9-11 (power heatmap/boxplots), compress cell 15 (MA1/MAV Cohen's d)
- NB 006: Remove all decimation code, temporal features sections 4-5, PSD section 7, condense summary

---

## 21. MACRO E - MIGRATION SCALEWAY + REGENERATION FEATURES (2026-05-05)

### 21.1 Decisions validees

- Option B: local single-pass pipeline (download 1 file, extract 3 features, delete, repeat)
- Corriger features.py AVANT regeneration (spectre 2 cotes + 20*log10)
- Paths S3: raw = `s3://mlops-data/raw/`, features = `s3://mlops-data/mldrone/features/`
- Renommer notebooks (drop `_COLAB`)
- Pipeline avec concurrence configurable (ProcessPoolExecutor, N=2 default)
- Single load: 1 fichier charge 1 fois, 3 features extraites du meme segment
- Lazy segments (generator views, pas d'allocation)
- np.save pour spectrogram (pas savez — trop RAM), np.savez_compressed pour PSD/IQ
- Sort deterministe obligatoire (split reproductible)
- Checkpoint/resume JSON (reprend apres interruption)

### 21.2 Taches MACRO E

**DONE** : E-FIX (features.py), E-PIPE (pipeline.py), E-DEP (boto3), E-NB-KALEIDO, E-NB-SCHEMA (metadata), E-SPLIT (splitting.py), E-NB-021-PILOT (SVM end-to-end). E-EXPLOR-ASSESS DONE (session 5). NB Colab tasks SUPERSEDED by section 22.

| # | Tache | Complexite | Statut | Detail |
|---|-------|-----------|--------|--------|
| E-PREPROC | Evaluer consolidation NB 012/013/014 | S | DONE | Session 6 suite: 100% redondants avec pipeline.py |
| E-VERIFY | Macro-verification E | M | PENDING | 2 Opus: zero drive refs, code quality, determinism |

### 21.3 Infrastructure et risques

- Pipeline: storage.py + pipeline.py, bucket mlops-data/raw/{CONDITION}/{DRONE}_{STATE}/, features → mldrone/features/
- All identified risks resolved: deterministic sort implemented, np.save for spectro, dtype U32 for metadata
- NB 012/013/014: evaluation consolidation pending (E-PREPROC)

---

## 22. MIGRATION NOTEBOOKS 022/023/031 — PLAN DETAILLE (2026-05-06)

### 22.1 Decisions

- Asserts inline → tests unitaires (test_data_integrity.py)
- Cycle par notebook : modifier → verifier (agent) → user relance → agent analyse outputs → mise a jour plan si necessaire
- ResNet50FC dans NB 031 : garder ancienne archi (100352) pour compatibilite .pth, note markdown, corriger apres re-entrainement NB 022
- WHAT→WHY comments cleanup integre dans chaque migration
- Export figures dans output/figures/{notebook_name}/ pour verification par agents
- Coherence libs plots : a verifier, peut amener a une planification supplementaire
- Observations/conclusions : verification apres re-run, necessitera atomisation selon le nombre de notebooks

### 22.2 Migration complete (DONE)

Migration NB 022 (14 modifs), NB 023 (11 modifs), NB 031 (13 modifs). Toutes verifiees par agents Opus.

Taches communes C1-C6 appliquees a chaque notebook : Colab cleanup, paths Drive→local, splitting.py, rename, allow_pickle removal, renumerotation sections.

- **NB 022** : val_loader, ReduceLROnPlateau(val_loss), early stopping, spectro loading .npy+_meta.npz, dead code cleanup, max_epochs 30→120
- **NB 023** : val_loader, early stopping, best model copy.deepcopy, dead code cleanup
- **NB 031** : ResNet50FC ancien garde (100352, note markdown), splits unifies, spectro loading, dead imports/code cleanup, .pth names fixed

### 22.6 Taches transverses — DONE

T-SPLIT-TEST (12 tests), T-RENAME (4 NB, 11 refs), T-PLOT-EXPORT (save_figure+kaleido), T-PLOT-LIB (all plotly, dead imports removed), T-MACRO-V (2 Opus parallel, PASS).

### 22.7 Taches futures (ordonnees par priorite et dependances)

#### Priorite 1 : NB 031 Metadata + Preprocessing

| Task ID | Description | Complexity | Status | Builder | Verifier | Notes |
|---------|-------------|------------|--------|---------|----------|-------|
| NB031-META | Export metadata NB 031 (confusion_matrices.json, per_class_metrics.csv, statistical_tests.csv) + inclure dans upload | S | DONE | Opus | Opus | Session 6 |
| E-PREPROC | Evaluer consolidation NB 012/013/014 en un seul notebook preprocessing | S | DONE | Opus | Opus | Session 6 suite |
| CLI-EXPORT-SAMPLES | Creer src/dronedetect/export_samples.py. 1 fichier complet (100 segments) par combo (drone, condition). 3 .npz par combo (PSD, spectro, IQ). Nouveau nommage sans suffix. Output: interface/media/test_samples/{CONDITION}/ | S | DONE | Opus | Opus | Session 7 |
| NB-PREPROC | Creer notebook preprocessing local (015_preprocessing.ipynb), wrapping run_pipeline() + export_test_samples(). PAS Colab | M | DONE | Opus (multi) | Opus | Session 7 |

#### Priorite 2 : NB 023 RF-UAVNet (apres preprocessing)

| Task ID | Description | Complexity | Status | Builder | Verifier | Notes |
|---------|-------------|------------|--------|---------|----------|-------|
| NB023-ADAM | Remplacer optim.SGD par optim.Adam(lr=1e-3) dans NB 023. Supprimer momentum param. Justification : alignement papier (Huynh-The 2022) + repo IQTLabs | S | DONE | Opus | Opus | Session 7, verificateur 8/8 PASS |
| NB023-OPTIM-DOC | Mettre a jour cellule "Hyperparameter Modifications" : papier=Adam, repo=Adam, notre choix=Adam (aligne). Documenter le LR 1e-3 | S | DONE | Opus | Opus | Included in NB023-ADAM |

#### Priorite 3 : Nettoyage metrics hardcodes (apres retraining)

| Task ID | Description | Complexity | Status | Builder | Verifier | Notes |
|---------|-------------|------------|--------|---------|----------|-------|
| NB-CLEAN-METRICS-021 | NB 021 : supprimer % hardcodes dans markdown. Remplacer par outputs code ou WHY-comments pertinents si justification necessaire | S | TODO | Opus | Opus | Apres retrain |
| NB-CLEAN-METRICS-022 | NB 022 : idem | S | TODO | Opus | Opus | Apres retrain |
| NB-CLEAN-METRICS-023 | NB 023 : idem | S | TODO | Opus | Opus | Apres retrain |
| NB-CLEAN-METRICS-031 | NB 031 : idem | S | TODO | Opus | Opus | Apres retrain |

#### Priorite 4 : NB 031 (apres NB 023)

Depends on NB023-ADAM (retraining RF-UAVNet) + NB-PREPROC (features regénérées).

| Task ID | Description | Complexity | Status | Builder | Verifier | Notes |
|---------|-------------|------------|--------|---------|----------|-------|
| NB031-ROBUSTESSE | Section accuracy/F1 par condition d'interference ET par etat de vol dans NB 031 | S | TODO | Opus | Opus | Apres retraining |
| F-STREAMLIT-EXPORT | Regenerer export test data pour Streamlit apres retraining | M | TODO | Opus | Opus | Absorbe dans NB-PREPROC si preprocessing notebook genere les samples |

#### Priorite 4b : Unification chemins modeles (source verite = data/)

| Task ID | Description | Complexity | Status | Builder | Verifier | Notes |
|---------|-------------|------------|--------|---------|----------|-------|
| FIX-MODELS-DIR-CONFIG | config.py ./models → ./data/models | S | DONE | Opus | Opus | Session 8 |
| FIX-MODELS-DIR-UPLOAD | upload.py Path("models") → Path("data/models") | S | DONE | Opus | Opus | Session 8 |
| FIX-MODELS-DIR-SETTINGS | settings.py MODELS_DIR + CSV path → data/models/ | S | DONE | Opus | Opus | Session 8 |
| FIX-MODELS-DIR-DOCS | .env.example + README.md defaults → data/models/ | S | DONE | Opus | Opus | Session 8 |
| VERIFY-CLI-PIPELINE-IMPACT | Verifier impact changement config.py sur CLI pipeline | S | DONE | Opus | — | Session 8, pipeline.py n'utilise pas MODELS_DIR |
| CLEANUP-STALE-MODELS | Supprimer models/, interface/media/models/, stale pkl, Zone.Identifier | S | DONE | Opus | Opus | Session 8 |
| FIX-STREAMLIT-HOME | Corriger overlap=120 description, downsampling 120x dans home_view.py | S | DONE | Opus | Opus | Session 8 |
| MACRO-VERIFY-MODELS | Grep global zero stale references apres cleanup | M | DONE | Opus | — | Session 8 |

#### Priorite 5 : Documentation + Cleanup

| Task ID | Description | Complexity | Status | Builder | Verifier | Notes |
|---------|-------------|------------|--------|---------|----------|-------|
| FIX-SETTINGS-INTERFERENCE | 4 modifs dans 3 fichiers: settings.py (INTERFERENCE_TYPES + suffix removal), home_view.py (descriptions 4 conditions), inference_view.py (docstring + glob check) | S | DONE | Opus (1 par fichier) | Opus | Session 7 |
| FIX-NB003-HARDCODED | Remplacer ['CLEAN', 'BOTH'] hardcode dans NB 003 (3 endroits: subplot_titles, colors, for loop) par liste dynamique | S | DONE | Opus | Opus | Session 7 |
| FIX-NB004-HARDCODED | Remplacer ['CLEAN', 'BOTH'] hardcode dans NB 004 (4 endroits: for loops, subplot_titles, enumerate) par liste dynamique | S | DONE | Opus | Opus | Session 7 |
| README-CREDIT | Mettre a jour README : credit IQTLabs/RFClassification (Apache-2.0), correction data leakage, ref papier Huynh-The 2022 IEEE Access | S | TODO | Opus | Opus | Doc existante dans docs/LEAKAGE_PROOF.md |
| F-CLI-DOC | Documenter toutes les commandes CLI (pipeline, upload) | S | TODO | Opus | Opus | — |
| F-RESNET-FIX | Corriger ResNet50FC dans NB 031 + models.py + interface/models/cnn.py | S | DONE | Opus | Opus | Session 6 |
| FIX-FORMAT-NB006 | Remplacer f-string alignment specifiers (:>/<) par pd.DataFrame dans NB 006 | S | DONE | Opus | Opus | Session 8 |
| FIX-FORMAT-NB031 | Remplacer f-string alignment specifiers (:>/<) par pd.DataFrame dans NB 031 | S | DONE | Opus | Opus | Session 8 |

#### Taches conservees (non ordonnees)

| Task ID | Description | Complexity | Status | Builder | Verifier | Notes |
|---------|-------------|------------|--------|---------|----------|-------|
| F-WHAT-EXPLOR | WHAT→WHY comments cleanup NB exploration | M | ABSORBED | — | — | Absorbed into CLEAN-NB001/002/003/004 (section 26) |
| F-OBS-VERIF | Verifier observations/conclusions a jour | M | ABSORBED | — | — | Absorbed into CLEAN-NB* tasks (section 26) |
| F-EPOCH-DOC | Definir epoch dans documentation | S | ABSORBED | — | — | Absorbed into GLOSSAIRE-BULK (section 26) |
| F-NB-CLEANUP | Supprimer/consolider notebooks obsoletes | M | DONE | Opus | Opus | NB 012/013/014 deleted (F-EXPLOR-DEPRECATE) |
| F-EXPLOR-DEPRECATE | Marquer NB 012/013/014 comme deprecated | S | DONE | Opus | Opus | Session 8, 9 files deleted |
| F-EXPLOR-FIX-001 | Cleanup NB 001 | S | ABSORBED | — | — | Absorbed into CLEAN-NB001 (section 26) |
| F-EXPLOR-FIX-002 | Cleanup NB 002 | S | ABSORBED | — | — | Absorbed into CLEAN-NB002 (section 26) |
| F-EXPLOR-FIX-004 | Cleanup NB 004 | S | ABSORBED | — | — | Absorbed into CLEAN-NB004 (section 26) |
| F-EXPLOR-RERUN | Re-run exploration NB avec 195 fichiers + caveat | M | ABSORBED | — | — | Absorbed into MACRO-COLAB-EXPLOR (section 26, P6) |
| RE-RUN-NB031 | Relancer NB 031 apres retraining | M | DONE | Opus | Opus | Session 8, VGG16 89.6%, SVM 83.5%, ResNet50 79.0%, RFUAVNet 47.6% |
| AUDIT-SRC | Audit qualitatif src/dronedetect/ | M | DONE | Opus | Opus | Session 8, findings → MACRO AQ |
| AUDIT-INTERFACE | Audit qualitatif interface/ | M | DONE | Opus | Opus | Session 8, findings → MACRO AQ |
| FIX-DATA-DIR | settings.py DATA_DIR → data/features | S | DONE | Opus | Opus | Session 8 |
| MACRO-EXPLORE | Content audit 9 NB + 4 diagrams + glossary | L | DONE | Opus (7) | — | Session 8 part 2, see section 26 |

### 22.8 Chaine de dependances

~~CLI-EXPORT-SAMPLES → NB-PREPROC → NB023-ADAM → [retrain Colab] → NB-CLEAN-METRICS-* → NB031-ROBUSTESSE → RE-RUN-NB031~~ DONE up to NB023-ADAM + RE-RUN-NB031

~~FIX-MODELS-DIR (CONFIG → UPLOAD → SETTINGS → DOCS) → VERIFY-CLI-PIPELINE-IMPACT → CLEANUP-STALE-MODELS → MACRO-VERIFY-MODELS~~ DONE (session 8)

~~FIX-SETTINGS-INTERFERENCE, FIX-NB003/004-HARDCODED, FIX-FORMAT-NB006/031, FIX-STREAMLIT-HOME~~ ALL DONE (sessions 7-8)

Remaining dependency chain (see section 26 for full plan):
- P1: DIAGRAM-* (4 parallel) → VERIFY-DIAGRAMS
- P2: CLEAN-NB006 (DONE) + CAVEAT-195 (CANCELLED → replaced by MACRO-COLAB-EXPLOR)
- P3: GLOSSAIRE-BULK + CLEAN-NB001 + CLEAN-NB031 (independent)
- P4: CLEAN-NB002 + CLEAN-NB003 + CLEAN-NB004 (independent)
- P5: NB031-ROBUSTESSE (independent)
- P6: MACRO-COLAB-EXPLOR (after P3+P4 NB cleanup) → NB-CLEAN-METRICS

Regle : preprocessing avant training, training avant comparison. Chaque tache = 1 builder Opus + 1 verifier Opus (separes).

---

## 23. UPLOAD ARTEFACTS SCALEWAY (2026-05-06) — DONE

- CLI: `dronedetect/upload.py` avec flags --features, --models, --split, --all
- Taches U-1 a U-5 toutes DONE (storage.py + upload.py + verification)
- Prefixes S3: mldrone/features/, mldrone/models/, mldrone/split/
- Bucket `mldrone-artefacts` (separe du raw data `mlops-data`)
- Multipart upload 100MB chunks pour gros fichiers (22GB spectro)

---

## 24. MACRO AQ — AUDIT QUALITY FIXES (src/ + interface/) (2026-05-08)

Issu de AUDIT-SRC + AUDIT-INTERFACE (session 8). Dead code, dead config, WHAT-comments, typing, coupling.

| Task | Description | Complexity | Status | Dep |
|------|-------------|------------|--------|-----|
| AQ1 | DELETE-DEAD-FILES: viz_service.py + models.py + __init__ updates | S | DONE | — |
| AQ2 | CLEAN-DEAD-SRC: interpolate_2d, iter_download, commented block, FEATURES_DIR dup, all_classes | S | DONE | — |
| AQ3a | CLEAN-INTF-VIZ: model_viz_service.py 5 dead methods | S | DONE | AQ1 |
| AQ3b | CLEAN-INTF-SETTINGS: dead config, get_test_sample_path, 5 dead keys, Django docstring | S | DONE | — |
| AQ3c | CLEAN-INTF-VIEWS: inference_view, cnn, comparison_view, views/__init__ | S | DONE | AQ1 |
| AQ3d | CLEAN-INTF-HOME: drone_data duplication + os.path→pathlib | S | DONE | — |
| AQ4a | WHAT-COMMENTS: settings + services + models | S | DONE | AQ3a,3b |
| AQ4b | WHAT-COMMENTS: home_view + app | S | DONE | AQ3d |
| AQ4c | WHAT-COMMENTS: inference_view | S | DONE | AQ3c |
| AQ4d | WHAT-COMMENTS: comparison_view | S | DONE | AQ3c |
| AQ5 | FIX-TYPING: 6 files Tuple→tuple etc. | S | DONE | AQ2,3a,3c |
| AQ6 | FIX-COUPLING: export_samples → config instead of interface.settings | S | DONE | AQ2 |
| AQ7 | FIX-STALE-CSV: comparison_view → data/models/ | S | DONE | AQ3b |
| AQ-MV | MACRO-VERIFY: grep + ruff + coherence | M | DONE | all |

MACRO AQ completed: 2 dead files deleted, ~58 WHAT-comments removed, 5 dead code cleaned, typing modernized, coupling fixed, CSV path unified, ruff 0 errors. Extra: glossary_view.py dead code cleanup (unused imports + methods).

## 25. SESSION 8 COMPLETED TASKS (2026-05-08)

### 25.1 Model paths unification (DONE)

FIX-MODELS-DIR chain: 5 files changed (config.py, upload.py, settings.py, .env.example, README.md). Source of truth = data/models/. CLEANUP-STALE-MODELS: deleted models/, interface/media/models/, stale pkl, Zone.Identifier. MACRO-VERIFY-MODELS: grep global zero stale refs.

### 25.2 Data paths and cleanup (DONE)

- FIX-DATA-DIR: settings.py DATA_DIR notebooks/data/features -> data/features (then removed as dead code in AQ3b)
- F-EXPLOR-DEPRECATE: NB 012/013/014 deleted (9 files total)
- FIX-FORMAT-NB006 + FIX-FORMAT-NB031: f-string alignment specifiers replaced by pd.DataFrame

### 25.3 MACRO AQ - Audit Quality (14 tasks, all DONE)

Issued from AUDIT-SRC + AUDIT-INTERFACE (2 Opus parallel agents).

- AQ1 DELETE-DEAD-FILES: viz_service.py (old) + models.py duplication removed
- AQ2 CLEAN-DEAD-SRC: interpolate_2d, iter_download, commented block, FEATURES_DIR dup, all_classes (5 files)
- AQ3a-d CLEAN-INTF: model_viz_service 5 dead methods, settings dead keys/DATA_DIR/INTERFACE_DIR, inference/comparison dead code, home drone_data dup
- AQ4a-d WHAT-COMMENTS: ~58 removed across interface/
- AQ5 FIX-TYPING: 6 files modernized (Tuple->tuple, Optional->X|None, List->list)
- AQ6 FIX-COUPLING: export_samples.py decoupled from interface.settings -> uses config.py
- AQ7 FIX-STALE-CSV: comparison_view -> data/models/ path
- AQ-MV MACRO-VERIFY: grep + ruff 0 errors + coherence PASS

### 25.4 Streamlit fixes (DONE)

- FIX-STREAMLIT-HOME: noverlap=120 corrected, 120x downsampling key finding documented
- MOVE-TEST-SAMPLES: interface/media/test_samples -> data/test_samples, data/sample deleted, interface/media/ removed entirely
- FIX-HOME-IMPORT: from interface.settings -> from settings (crash fix, app.py adds interface/ to sys.path)

### 25.5 Retraining + comparison (DONE)

- NB 023 retrained with Adam on Colab: 47.6% (120x downsampling bottleneck confirmed, Adam vs SGD no significant gain)
- NB 031 re-run: VGG16 89.6%, SVM 83.5%, ResNet50 79.0%, RFUAVNet 47.6%
- McNemar: all pairs significant (p<0.05). Bootstrap 95% CI non-overlapping.
- 150 hard samples (2.67%) where no model predicts correctly

### 25.6 Session 8 part 2 completed tasks

| Task ID | Description | Complexity | Status |
|---------|-------------|------------|--------|
| AUDIT-NB023-PDF | Deep analysis NB 023 outputs | S | DONE |
| AUDIT-NB031-PDF | Deep analysis NB 031 outputs | S | DONE |
| FIX-NB023-METRICS | 51.3% -> 47.6% (3 occurrences cells 28-29) | S | DONE |
| FIX-NB023-PATH | Stale path data/sample/test_data/ -> data/test_samples/ | S | DONE |
| FIX-INFERENCE-PROTOCOL | st.metric -> st.markdown for full-width protocol | S | DONE |
| FIX-INFERENCE-EXPLAIN | st.expander with CNN softmax + SVM Platt explanation | S | DONE |
| FIX-HOME-RFCLASS-LINK | RFClassification repo link in home warning box | S | DONE |
| MACRO-EXPLORE | Content audit 9 NB + 4 diagrams + glossary | L | DONE |
| DIAGRAM-GLOBAL | Redraw pipeline_global.dot | S | DONE |
| DIAGRAM-SVM | Redraw pipeline_svm.dot | S | DONE |
| DIAGRAM-CNN | Redraw pipeline_cnn.dot | S | DONE |
| DIAGRAM-RFUAVNET | Redraw pipeline_rfuavnet.dot | S | DONE |
| VERIFY-DIAGRAMS | Verify 4 .dot files against code | S | DONE |

### 25.7 Remaining tasks

| Task ID | Description | Complexity | Status |
|---------|-------------|------------|--------|
| NB-CLEAN-METRICS (x4) | Remove hardcoded % from 4 NB markdown cells | M | TODO (P5, dep user re-runs NB locally) |
| NB031-ROBUSTESSE | Accuracy/F1 per condition + flight state | M | DONE (P5) |
| F-CLI-DOC | Document CLI commands | S | DONE |
| README-CREDIT | Credit IQTLabs + leakage correction | S | DONE |
| AUDIT-NB031-RERUN | Verify NB 031 PDF outputs after rerun | S | DONE |
| NB-TRIM (5 NB) | Trim/delete exploration NB per validated plan | M | DONE |
| TRIM-DEL-004 | DELETE NB 004 (dead-end) | S | DONE |
| TRIM-NB001 | 32->29 cells, cell 8 removed (5/7), cell 29 removed (dup), constellation removed | S | DONE |
| TRIM-NB002 | 28->26 cells, IQ traj + raw means removed, ADF simplified, leakage caveat added | S | DONE |
| TRIM-NB003 | 16->13 cells, power heatmap/boxplots removed, Cohen's d compressed + conclusion fixed | S | DONE |
| TRIM-NB006 | 29->18 cells (-38%), decimation eliminated, sections 4/5/7 removed, summary condensed | M | DONE |
| CLEAN-NB006 | 52.9%->47.6%, burst detection, NEXT STEPS | M | DONE (P2) |
| GLOSSAIRE-BULK | Add ~20 terms to glossary.json | M | DONE (P3) |
| CLEAN-NB001 | ~30 WHAT, interpretations, dBm | M | DONE (P3) |
| CLEAN-NB031 | File-level split intro, McNemar, fallback | S | DONE (P3) |
| CLEAN-NB002 | ~38 WHAT, section ordering, FR->EN | M | DONE (P4) |
| CLEAN-NB003 | Cohen's d caveat, PHA, dead code | S | DONE (P4) |
| CLEAN-NB004 | NOTEBOOK_NAME, placeholder, Colab blocks | S | DONE (P4) |
| MACRO-COLAB-EXPLOR | Migrate exploration NB to Colab + re-run 390 files + PDF export | S | DONE (redefined, CAVEAT-195 applied) |
| MACRO C | Slides + Dossier (BLOCKED on P3-P6) | L | BLOCKED |

---

## 26. MACRO EXPLORE — Content Audit + Diagrams (2026-05-08)

### 26.1 Summary

Deep content audit of the entire project: 9 notebooks, 4 Graphviz diagrams, and Streamlit glossary. Conducted via 7 parallel Opus agents (5 for notebooks, 1 for diagrams, 1 for glossary). Synthesis produced a consolidated plan of 15 new tasks across 5 priority levels.

**Key findings:**
- Exploration notebooks (001-004, 006) run on **195 files / 2 conditions** (CLEAN+BOTH), not 390 files. Caveat needed.
- All 4 Graphviz diagrams have parameter errors vs actual code (wrong windows, hop sizes, missing steps). Must be redrawn from code.
- Streamlit glossary missing ~20 terms (epoch, FHSS, OFDM, OcuSync, Lightbridge, etc.)
- NB 006 still has stale accuracy (52.9% instead of 47.6%) and obsolete burst detection section
- NB 001 has ~30 WHAT comments, NB 002 has ~38 WHAT comments
- NB 031 missing file-level split introduction and McNemar justification

**Decisions taken:**
- Exploration notebooks stay on 195 files (not 390). Add explicit caveat.
- Diagrams rewritten from actual code, not patched.
- No new Streamlit page (glossary enrichment only).
- F-EXPLOR-FIX-001/002/004 absorbed into broader CLEAN-NB* tasks.
- F-EPOCH-DOC absorbed into GLOSSAIRE-BULK.
- MACRO E (#38) marked DONE (pipeline migration completed sessions 3-4).

### 26.2 Consolidation decisions

| Old Task | Decision | New Task |
|----------|----------|----------|
| F-EXPLOR-FIX-001 (#83) | ABSORBED | CLEAN-NB001 (broader scope) |
| F-EXPLOR-FIX-002 (#84) | ABSORBED | CLEAN-NB002 (broader scope) |
| F-EXPLOR-FIX-004 (#85) | ABSORBED | CLEAN-NB004 (broader scope) |
| F-EPOCH-DOC (#62) | ABSORBED | GLOSSAIRE-BULK |
| MACRO E (#38) | DONE | Pipeline migration completed sessions 3-4 |
| FIX-HOME-RFCLASS-LINK (#169) | DONE | Corrected broken GitHub link |
| F-EXPLOR-RERUN (#86) | ABSORBED | Into MACRO-COLAB-EXPLOR (P6, 390 files on Colab) |

### 26.3 Task plan

#### P1 — DIAGRAMS (DONE)

| Task ID | Description | Complexity | Status | Builder | Verifier | Notes |
|---------|-------------|------------|--------|---------|----------|-------|
| DIAGRAM-GLOBAL | Redraw pipeline_global.dot from actual code | S | DONE | Opus | — | seg->norm order fixed, 3 branches |
| DIAGRAM-SVM | Redraw pipeline_svm.dot (Hamming, noverlap=512, seg→norm order) | S | DONE | Opus | — | Hamming, fftshift added |
| DIAGRAM-CNN | Redraw pipeline_cnn.dot (hop=904, dB/resize/viridis steps, FC details) | S | DONE | Opus | — | VGG16/ResNet50 FC details |
| DIAGRAM-RFUAVNET | Redraw pipeline_rfuavnet.dot (residual connections, multi-scale GAP) | S | DONE | Opus | — | 5 branches, padding |
| VERIFY-DIAGRAMS | Verify all 4 .dot files against code | S | DONE | Opus | — | All 4 PASS (RFUAVNET: 1 correction cycle) |

#### P2 — BLOQUANTS (independent, can start immediately)

| Task ID | Description | Complexity | Status | Builder | Verifier | Notes |
|---------|-------------|------------|--------|---------|----------|-------|
| CLEAN-NB006 | 52.9%→47.6%, burst detection removal, NEXT STEPS, 350K>baseline | M | DONE | Opus | Opus | Session 8 part 3: 52.9%→47.6%, burst detection removed, recommendations rewritten |
| CAVEAT-195 | Add "195 files / 2 conditions" caveat in NB 001-004, 006 | S | DONE | Opus | Opus | Re-applied after MACRO-COLAB-EXPLOR redefined (Colab abandoned). Caveat added to 5 NB. |

#### P3 — GLOSSAIRE + NB PRIORITAIRES (DONE)

| Task ID | Description | Complexity | Status | Builder | Verifier | Notes |
|---------|-------------|------------|--------|---------|----------|-------|
| GLOSSAIRE-BULK | Add ~20 terms to glossary.json | M | DONE | Opus | Opus | Session 8 part 3: 16 terms added (32→48 total) |
| CLEAN-NB001 | ~30 WHAT, CLEAN/ON vs BOTH/FY, dBm, emoji, interpretations | M | DONE | Opus | Opus | Session 8 part 3: ~25 WHAT removed, scope clarified, power_dbm→power_db |
| CLEAN-NB031 | File-level split intro, MA1/MAV RF, WHY McNemar, remove fallback | S | DONE | Opus | Opus | Session 8 part 3: fallback splitting removed (~170 lines) |

#### P4 — NB RESTANTS (DONE)

| Task ID | Description | Complexity | Status | Builder | Verifier | Notes |
|---------|-------------|------------|--------|---------|----------|-------|
| CLEAN-NB002 | ~38 WHAT, section ordering, FR→EN, interpretations | M | DONE | Opus | Opus | Session 8 part 3: ~38 WHAT removed, FR→EN section 5, unused imports cleaned |
| CLEAN-NB003 | 3 WHAT, Cohen's d caveat, PHA, interpretations, dead code | S | DONE | Opus | Opus | Session 8 part 3: Cohen's d caveat added, PHA dynamicized, 2 Colab cells deleted |
| CLEAN-NB004 | NOTEBOOK_NAME, placeholder, Colab blocks | S | DONE | Opus | Opus | Session 8 part 3: NOTEBOOK_NAME fixed, placeholder removed, dead code cleaned |

#### P5 — POST-RERUN (blocked by F-EXPLOR-RERUN)

| Task ID | Description | Complexity | Status | Builder | Verifier | Notes |
|---------|-------------|------------|--------|---------|----------|-------|
| F-EXPLOR-RERUN | Re-run exploration NB with 195 files + add caveat | M | TODO | Opus | Opus | Updated: 195 files, not 390 |
| NB031-ROBUSTESSE | Accuracy/F1 per condition + flight state | M | DONE | Opus | Opus | Session 8 part 3: Section 13b added, plotly charts + heatmaps, CSV export |
| NB-CLEAN-METRICS | Remove hardcoded % from 4 NB markdown cells | M | TODO | Opus | Opus | Existing #160, blocked by rerun |

#### P6 — COLAB MIGRATION EXPLORATION NB (DONE — redefined)

| Task ID | Description | Complexity | Status | Builder | Verifier | Notes |
|---------|-------------|------------|--------|---------|----------|-------|
| MACRO-COLAB-EXPLOR | Migrate NB 001-004, 006 to Colab+Scaleway (like training NB). User re-runs on 390 files / 4 conditions. Export PDF. | S | DONE (redefined) | Opus | Opus | Colab abandoned. CAVEAT-195 applied locally. User re-runs NB in local. |

### 26.4 Dependency chain

```
P1: DONE (4 diagrams rewritten, all verified PASS)

P2: CLEAN-NB006 (DONE)
    CAVEAT-195 (CANCELLED — replaced by MACRO-COLAB-EXPLOR)

P3: GLOSSAIRE-BULK (DONE)
    CLEAN-NB001 (DONE)
    CLEAN-NB031 (DONE)

P4: CLEAN-NB002 (DONE)
    CLEAN-NB003 (DONE)
    CLEAN-NB004 (DONE)

P5: NB031-ROBUSTESSE (DONE)

P6: MACRO-COLAB-EXPLOR (DONE — redefined: Colab abandoned, CAVEAT-195 applied locally)
    → User re-runs exploration NB locally (195 files / 2 conditions)
    → NB-CLEAN-METRICS (update hardcoded %, depends on user re-running NB locally)

MACRO C (BLOCKED): ATOMIZE → Slides + Dossier → MACRO-VERIFY C
```

P1 DONE. P2 DONE (CLEAN-NB006 done, CAVEAT-195 cancelled then re-applied). P3 DONE. P4 DONE. P5 DONE. P6 DONE (MACRO-COLAB-EXPLOR redefined: Colab migration abandoned, CAVEAT-195 applied locally instead). NB-CLEAN-METRICS depends on user re-running notebooks locally (not on Colab migration). MACRO C remains blocked on NB-CLEAN-METRICS.

### 26.5 Remaining non-EXPLORE tasks

| Task ID | Description | Complexity | Status | Notes |
|---------|-------------|------------|--------|-------|
| README-CREDIT | Credit IQTLabs/RFClassification + leakage correction | S | DONE | Session 8 part 3 |
| F-CLI-DOC | Document all CLI commands | S | DONE | Session 8 part 3 |
| AUDIT-NB031-RERUN | Verify NB 031 PDF outputs after rerun | S | DONE | Session 8 part 3 |
| NB-TRIM (5 NB) | Trim/delete exploration NB per validated plan | M | DONE | Session 8 part 4 |

**NB-TRIM subtasks:**

| Task ID | Description | Complexity | Status | Builder | Verifier | Notes |
|---------|-------------|------------|--------|---------|----------|-------|
| TRIM-DEL-004 | DELETE NB 004 (dead-end, nothing used downstream) | S | DONE | Opus | — | Session 8 part 4 |
| TRIM-NB001 | 32->29 cells, cell 8 removed (5/7 bug), cell 29 removed (dup), constellation removed from cell 31 | S | DONE | Opus | Opus | Session 8 part 4 |
| TRIM-NB002 | 28->26 cells, IQ traj + raw means removed, ADF simplified, sections renumbered, pipeline note + leakage caveat added | S | DONE | Opus | Opus | Session 8 part 4 |
| TRIM-NB003 | 16->13 cells, power heatmap/boxplots removed, Cohen's d compressed + conclusion fixed | S | DONE | Opus | Opus | Session 8 part 4 |
| TRIM-NB006 | 29->18 cells (-38%), decimation eliminated, sections 4/5/7 removed, summary condensed to 3 bullets | M | DONE | Opus | Opus | Session 8 part 4 |

| MACRO C (#15, #23, #24) | Slides + Dossier projet | L | BLOCKED | Blocked on P3-P6 content |

### 26.6 Session 8 part 4

Session 8 part 4: NB-TRIM ALL DONE: DELETE NB 004, TRIM NB 001 (32->29), TRIM NB 002 (28->26 + structural fixes + leakage caveat), TRIM NB 003 (16->13 + Cohen's d fix), TRIM NB 006 (29->18, -38%). Bonus: RF-UAVNet leakage caveat added in NB 002, contradictory Cohen's d conclusion fixed in NB 003.

**Remaining tasks:** User re-run (4 exploration NB locally) -> NB-CLEAN-METRICS -> MACRO C.
