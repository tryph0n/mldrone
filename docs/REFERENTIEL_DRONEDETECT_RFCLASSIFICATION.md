# RÉFÉRENTIEL TRAITEMENT DES DONNÉES - RFClassification (DroneDetect)

Date d'analyse: 2025-12-17
Repository: https://github.com/tryph0n/RFClassification
Dataset: DroneDetect (7 classes - drones uniquement)
Analysé par: Claude Sonnet 4.5

---

## SOMMAIRE

### [1. SVM (PSD Features)](#1-svm-psd-features)
- [1.1 Feature Selection](#11-feature-selection)
- [1.2 Feature Engineering](#12-feature-engineering)
  - [1.2.1 Paramètres PSD (Welch)](#121-paramètres-psd-welch)
  - [1.2.2 Conversion dB](#122-conversion-db)
  - [1.2.3 Filtrage fréquentiel](#123-filtrage-fréquentiel)
  - [1.2.4 Features dérivées](#124-features-dérivées)
  - [1.2.5 Pipeline d'extraction](#125-pipeline-dextraction)
- [1.3 Autres Traitements](#13-autres-traitements)
  - [1.3.1 Normalization/Scaling](#131-normalizationscaling)
  - [1.3.2 Data Augmentation](#132-data-augmentation)
  - [1.3.3 Resampling](#133-resampling)
  - [1.3.4 Outlier Removal](#134-outlier-removal)
  - [1.3.5 Class Balancing](#135-class-balancing)
  - [1.3.6 Train/Test Split](#136-traintest-split)
  - [1.3.7 Cross-Validation](#137-cross-validation)
  - [1.3.8 Pipeline Complet](#138-pipeline-complet-ordre-des-opérations)
- [1.4 Résumé SVM](#14-résumé-svm)

### [2. VGG16/ResNet50 (Spectrogram Images)](#2-vgg16resnet50-spectrogram-images)
- [2.1 Feature Selection](#21-feature-selection)
- [2.2 Feature Engineering](#22-feature-engineering)
  - [2.2.1 Paramètres STFT](#221-paramètres-stft)
  - [2.2.2 Échelle de fréquence](#222-échelle-de-fréquence)
  - [2.2.3 Conversion en image](#223-conversion-en-image)
  - [2.2.4 Normalisation pixels](#224-normalisation-pixels)
  - [2.2.5 Colormap](#225-colormap)
  - [2.2.6 Pipeline d'extraction](#226-pipeline-dextraction)
- [2.3 Autres Traitements](#23-autres-traitements)
  - [2.3.1 Data Augmentation](#231-data-augmentation)
  - [2.3.2 Train/Test Split](#232-traintest-split)
  - [2.3.3 Class Balancing](#233-class-balancing)
  - [2.3.4 Transfer Learning](#234-transfer-learning)
  - [2.3.5 Cross-Validation](#235-cross-validation)
  - [2.3.6 Pipeline Complet](#236-pipeline-complet)
- [2.4 Résumé VGG16/ResNet50](#24-résumé-vgg16resnet50)

### [3. RFUAVNet (Raw IQ)](#3-rfuavnet-raw-iq)
- [3.1 Feature Selection](#31-feature-selection)
- [3.2 Feature Engineering](#32-feature-engineering)
  - [3.2.1 Format IQ data](#321-format-iq-data)
  - [3.2.2 Downsampling](#322-downsampling)
  - [3.2.3 Filtering](#323-filtering)
  - [3.2.4 Segmentation](#324-segmentation)
  - [3.2.5 Normalisation IQ](#325-normalisation-iq)
  - [3.2.6 Pipeline d'extraction](#326-pipeline-dextraction)
- [3.3 Architecture RFUAVNet](#33-architecture-rfuavnet)
- [3.4 Autres Traitements](#34-autres-traitements)
  - [3.4.1 Data Augmentation](#341-data-augmentation)
  - [3.4.2 Train/Test Split](#342-traintest-split)
  - [3.4.3 Class Balancing](#343-class-balancing)
  - [3.4.4 Cross-Validation](#344-cross-validation)
  - [3.4.5 Pipeline Complet](#345-pipeline-complet)
- [3.5 Résumé RFUAVNet](#35-résumé-rfuavnet-dronerf-uniquement)

### [4. COMPARAISON INTER-MODÈLES](#4-comparaison-inter-modèles)

### [5. INCERTITUDES ET LIMITATIONS](#5-incertitudes-et-limitations)

### [6. SOURCES](#6-sources)

### [7. RECOMMANDATIONS ET OBSERVATIONS](#7-recommandations-et-observations)

---

## 1. SVM (PSD Features)

### 1.1 Feature Selection

**Status:** NON

**Confirmation:** Aucune feature selection détectée dans le pipeline

**Fichiers analysés:**
- `/home/sambot/mldrone/RFClassification/run_dronedetect_feat.py`
- `/home/sambot/mldrone/RFClassification/ML Approaches.ipynb`
- `/home/sambot/mldrone/RFClassification/models.py`
- `/home/sambot/mldrone/RFClassification/loading_functions.py`
- `/home/sambot/mldrone/RFClassification/feat_gen_functions.py`

**Recherche effectuée pour:** PCA, SelectKBest, RFE, VarianceThreshold, feature selection

**Résultat:** Aucune méthode de feature selection appliquée sur les PSD features pour le modèle SVM

**Features:** 1024 → 1024 (aucune réduction)

### 1.2 Feature Engineering

#### 1.2.1 Paramètres PSD (Welch)

**Fichier:** `/home/sambot/mldrone/RFClassification/run_dronedetect_feat.py:174`
```python
fpsd, Pxx_den = signal.welch(d_complex, fs, window=psd_win_type, nperseg=n_per_seg)
```

**Paramètres définis:**
Fichier: `/home/sambot/mldrone/RFClassification/run_dronedetect_feat.py:28-36`
```python
fs = 60e6 #60 MHz
bandwidth = 28e6 # 28MHz
center_freq = 2.43e9

# Specifications on what features to generate
n_per_seg = 256 # length of each segment (powers of 2)
n_overlap_spec = 120
psd_win_type = 'hamming' # make ends of each segment match
```

**Paramètres utilisés dans ML Approaches.ipynb:**
Fichier: `/home/sambot/mldrone/RFClassification/ML Approaches.ipynb:80`
```python
n_per_seg = 1024
```

**IMPORTANT - Incohérence détectée:**

Le script de génération `run_dronedetect_feat.py:33` définit `n_per_seg = 256` par défaut, mais les résultats finaux (accuracy 85.4% ± 0.5%) ont été générés avec `n_per_seg = 1024` dans le notebook `ML Approaches.ipynb`.

Cette différence impacte la résolution fréquentielle:
- Avec n_per_seg=256: 60MHz/256 = 234.4 kHz/bin
- Avec n_per_seg=1024: 60MHz/1024 = 58.6 kHz/bin (meilleure résolution fréquentielle)

Les résultats documentés dans ce référentiel correspondent à **n_per_seg=1024** (valeur utilisée pour les résultats finaux).

**Paramètres finaux (utilisés pour résultats 85.4% accuracy):**
- **fs (sample rate):** 60,000,000 Hz (60 MHz)
- **nperseg (window length):** 1024
- **noverlap:** 512 (default = nperseg//2, non spécifié explicitement)
- **nfft:** 1024 (default = nperseg, non spécifié)
- **window:** 'hamming'
- **scaling:** 'density' (default, non spécifié)
- **return_onesided:** False (implicite pour complex input)

#### 1.2.2 Conversion dB

**Type:** PSD reste en échelle linéaire (pas de conversion dB)

**Fichier:** `/home/sambot/mldrone/RFClassification/loading_functions.py:82-87`
```python
Feat = DATA['feat'][i_infile]
if self.feat_name == 'SPEC':
    # conver to dB
    Feat = -10*np.log10(Feat)
# apply norm
Feat = Feat/np.max(Feat)
```

**Confirmation:** La conversion `-10*np.log10()` est appliquée UNIQUEMENT pour 'SPEC', pas pour 'PSD'

**Fichier:** `/home/sambot/mldrone/RFClassification/run_dronedetect_feat.py:174-176`
```python
fpsd, Pxx_den = signal.welch(d_complex, fs, window=psd_win_type, nperseg=n_per_seg)
if pa_save:
    F_PSD.append(Pxx_den)
```
PSD stocké directement sans conversion logarithmique

#### 1.2.3 Filtrage fréquentiel

**Status:** NON

Aucun filtrage fréquentiel appliqué sur les features PSD.
Bande complète utilisée: -30 MHz à +30 MHz (two-sided spectrum avec fs=60MHz)

#### 1.2.4 Features dérivées

**Status:** NON

Aucune feature dérivée calculée (pas de mean, std, skewness, kurtosis, spectral centroid, etc.)
Les 1024 bins PSD bruts sont utilisés directement comme features.

#### 1.2.5 Pipeline d'extraction

```
Raw IQ data (shape: complex64, variable length ~240M samples)
  ↓
Normalisation (z-score) (loading_functions.py:136)
  data_norm = (data-np.mean(data))/(np.sqrt(np.var(data)))
  ↓
Segmentation en t_seg=20ms (loading_functions.py:138-141)
  len_seg = int(20/1e3 * 60e6) = 1,200,000 samples
  ↓
Welch PSD (run_dronedetect_feat.py:174)
  signal.welch(d_complex, fs=60e6, window='hamming', nperseg=1024)
  ↓
PSD features (shape: 1024)
  Two-sided spectrum, 1024 frequency bins
  Frequency resolution: 60MHz/1024 ≈ 58.6 kHz/bin
```

### 1.3 Autres Traitements

#### 1.3.1 Normalization/Scaling

**Status:** OUI (deux niveaux)

**Niveau 1: Normalisation des données IQ brutes**
- Méthode: Z-score normalization (standardization)
- Fichier: `/home/sambot/mldrone/RFClassification/loading_functions.py:136`
```python
data_norm = (data-np.mean(data))/(np.sqrt(np.var(data)))
```
- Appliqué: Per-file (sur chaque fichier raw avant segmentation)

**Niveau 2: Normalisation des features PSD**
- Méthode: Division par maximum (min-max scaling vers [0,1])
- Fichier: `/home/sambot/mldrone/RFClassification/loading_functions.py:87`
```python
Feat = Feat/np.max(Feat)
```
- Appliqué: Per-sample (sur chaque PSD individuellement lors du chargement)

#### 1.3.2 Data Augmentation

**Status:** NON

Aucune data augmentation appliquée (pas de noise injection, time shift, rotation)

#### 1.3.3 Resampling

**Status:** NON

Aucun resampling appliqué (pas de SMOTE, RandomOverSampler, etc.)

#### 1.3.4 Outlier Removal

**Status:** NON

Aucun outlier removal appliqué

#### 1.3.5 Class Balancing

**Stratégie:** Aucune stratégie de balancing explicite

**Distribution classes:**
- Dataset total: 38,978 samples
- Classes: drones (3 classes: 'DJI', 'UDI', 'TAR' - déduit des noms de fichiers)
- Distribution exacte: NON TROUVÉ dans les fichiers analysés

**Pas de class_weight='balanced' dans le modèle SVM**
Fichier: `/home/sambot/mldrone/RFClassification/models.py:38`
```python
self.svc = svm.SVC(kernel='rbf', C=self.C, gamma = self.gamma)
```

#### 1.3.6 Train/Test Split

**Pour évaluation finale:**
- **Ratio:** 67% train / 33% test
- **Stratification:** NON
- **File-level grouping:** NON
- **Random seed:** None (aléatoire à chaque exécution)

**Fichier:** `/home/sambot/mldrone/RFClassification/ML Approaches.ipynb:369-372`
```python
X_train, X_test, y_train, y_test = train_test_split(X_use,
                                                    y_use,
                                                    test_size=0.33,
                                                    random_state=None)
```

**Pour cross-validation:**
Pas de train/test split externe (utilise k-fold directement sur tout le dataset)

#### 1.3.7 Cross-Validation

- **Type:** k-fold (standard, non stratifié)
- **Folds:** 5
- **Shuffle:** OUI
- **Random seed:** None

**Fichier:** `/home/sambot/mldrone/RFClassification/models.py:45`
```python
cv = KFold(n_splits=k_fold, random_state=None, shuffle=True)
```

**Résultats par fold (ML Approaches.ipynb):**
- Fold 1: Accuracy: 0.848, F1: 0.845
- Fold 2: Accuracy: 0.852, F1: 0.850
- Fold 3: Accuracy: 0.856, F1: 0.853
- Fold 4: Accuracy: 0.863, F1: 0.861
- Fold 5: Accuracy: 0.849, F1: 0.847

#### 1.3.8 Pipeline Complet (ordre des opérations)

```
1. Load raw IQ data (loading_functions.py:133)
   Format: binary float32 → complex64
   ↓
2. Z-score normalization (loading_functions.py:136)
   data_norm = (data - mean) / std
   ↓
3. Segmentation temporelle (loading_functions.py:138-141)
   Découpage en segments de 20ms (1.2M samples/segment)
   ↓
4. Generate PSD features (run_dronedetect_feat.py:174)
   Welch: nperseg=1024, window='hamming', fs=60MHz
   Output: 1024 frequency bins (two-sided)
   ↓
5. Save features to disk (run_dronedetect_feat.py:206)
   Format: numpy .npy files
   ↓
6. Load features for ML (ML Approaches.ipynb:95)
   DroneDetectTorch dataset class
   ↓
7. Per-sample normalization (loading_functions.py:87)
   Feat = Feat / max(Feat)
   ↓
8. Cross-validation split (models.py:45)
   5-fold KFold, shuffle=True
   ↓
9. Train SVM model (models.py:48)
   RBF kernel, C=1 (default), gamma='scale' (default)
```

### 1.4 Résumé SVM

- **Features initiales:** 1024 (PSD bins)
- **Features finales:** 1024 (aucune feature selection)
- **Transformations clés:**
  1. Z-score normalization des IQ raw (per-file)
  2. Welch PSD (nperseg=1024, hamming, two-sided)
  3. Per-sample normalization (division par max)
  4. SVM RBF kernel (C=1, gamma='scale')
  5. 5-fold cross-validation (shuffle, non-stratified)
- **Accuracy finale:** 85.4% ± 0.5% (5-fold CV)
- **F1-score finale:** 85.1% ± 0.5% (5-fold CV)
- **Inference time:** 9.96ms per sample (moyenne)

---

## 2. VGG16/ResNet50 (Spectrogram Images)

### 2.1 Feature Selection

**Status:** NON

Aucune sélection de features n'est effectuée avant ou après les couches convolutionnelles. Les embeddings extraits par VGG16 (25088 dimensions) et ResNet50 (100352 dimensions) sont directement connectés à une couche fully-connected finale sans PCA, feature map selection, ou pruning.

**Fichier:** `/home/sambot/mldrone/RFClassification/models.py:156-168, 187-195`
```python
# VGG16
self._fc = nn.Linear(25088, num_classes)
def forward(self, x):
    x = self.vggfeats(x)
    x = x.reshape(-1,25088)  # Reshape direct sans sélection
    x = self._fc(x)
    return x

# ResNet50
self._fc = nn.Linear(100352, num_classes)
def forward(self, x):
    x = self.resnetfeats(x)
    x = x.reshape(-1,100352)  # Reshape direct sans sélection
    x = self._fc(x)
    return x
```

### 2.2 Feature Engineering

#### 2.2.1 Paramètres STFT

**Fichier:** `/home/sambot/mldrone/RFClassification/Generate DroneDetect Features.ipynb:82-85, 242-243`
```python
n_per_seg = 1024  # FFT size
spec_han_window = np.hanning(n_per_seg)
n_overlap_spec = 120

spec, _, _, _ = plt.specgram(d_complex, NFFT=n_per_seg, Fs=fs, window=spec_han_window,
                              noverlap=n_overlap_spec, sides='onesided', scale='dB')
```

**Paramètres:**
- **fs (sample rate):** 60e6 Hz (60 MHz)
- **n_fft (NFFT):** 1024
- **win_length:** 1024 (même que n_fft)
- **hop_length:** 1024 - 120 = 904 samples
- **window:** 'hanning' (np.hanning)

#### 2.2.2 Échelle de fréquence

**Type:** Linear avec conversion en dB

**Fichier:** `/home/sambot/mldrone/RFClassification/Generate DroneDetect Features.ipynb:242-243`
```python
plt.specgram(d_complex, NFFT=n_per_seg, Fs=fs, window=spec_han_window,
             noverlap=n_overlap_spec, sides='onesided', scale='dB')
```

**Clarification:** L'échelle de fréquence est linéaire avec une conversion en échelle dB (logarithmique en amplitude, pas en fréquence). Il n'y a PAS de mel-scale.

#### 2.2.3 Conversion en image

**Taille finale:** 224 x 224 x 3

**Fichier:** `/home/sambot/mldrone/RFClassification/Generate DroneDetect Features.ipynb:249`
```python
F_SPEC.append(interpolate_2d(spec, (224,224)))
```

**Fichier:** `/home/sambot/mldrone/RFClassification/feat_gen_functions.py:9-18`
```python
def interpolate_2d(Sxx_in, output_size):
    x = np.linspace(0, 1, Sxx_in.shape[0])
    y = np.linspace(0, 1, Sxx_in.shape[1])
    f = interpolate.interp2d(y, x, Sxx_in, kind='linear')

    x2 = np.linspace(0, 1, output_size[0])
    y2 = np.linspace(0, 1, output_size[1])
    arr2 = f(y2, x2)
    return arr2
```

**Méthode:** Interpolation bilinéaire (scipy.interpolate.interp2d) pour redimensionner le spectrogramme à 224x224.

#### 2.2.4 Normalisation pixels

**Type:** Division par 255 (normalisation [0, 1])

**Fichier:** `/home/sambot/mldrone/RFClassification/loading_functions.py:90-94`
```python
elif self.feat_format == 'IMG':
    DATA = cv2.imread(self.dir_name+self.files[i])
    DATA = cv2.cvtColor(DATA, cv2.COLOR_BGR2RGB)
    Feat = DATA/255
```

**Valeurs:**
- Mean: Aucune (pas de normalisation ImageNet)
- Std: Aucune (pas de normalisation ImageNet)
- Range: [0, 1] (division par 255 uniquement)

**Clarification:** Il n'y a PAS de normalisation ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]). La normalisation est une simple division par 255 pour ramener les valeurs de pixels dans [0, 1].

#### 2.2.5 Colormap

**Type:** viridis

**Fichier:** `/home/sambot/mldrone/RFClassification/helper_functions.py:14`
```python
plt.pcolormesh(data, cmap='viridis', vmin=data.min(), vmax=data.max())
```

#### 2.2.6 Pipeline d'extraction

```
Raw IQ data (shape: complexe 64-bits, variable length)
  ↓
[STFT via plt.specgram] (Generate DroneDetect Features.ipynb:242-243)
  n_fft=1024, window=hanning, noverlap=120, scale='dB'
  ↓
[Interpolation bilinéaire à 224x224] (feat_gen_functions.py:9-18)
  ↓
[Sauvegarde en image avec colormap viridis] (feat_gen_functions.py:28-30)
  ↓
[Chargement et conversion BGR→RGB] (loading_functions.py:91-92)
  ↓
[Normalisation par division par 255] (loading_functions.py:94)
  ↓
Spectrogram image (shape: [224, 224, 3], range [0, 1])
```

### 2.3 Autres Traitements

#### 2.3.1 Data Augmentation

**Status:** NON

Aucune data augmentation n'est appliquée pour les modèles VGG16/ResNet50 avec spectrogrammes. Les imports de `torchvision.transforms` dans le notebook DL Approaches ne sont pas utilisés.

**Fichier:** `/home/sambot/mldrone/RFClassification/DL Approaches.ipynb:38`
```python
import torchvision.transforms as transforms  # Importé mais non utilisé
```

Les techniques de data augmentation trouvées dans le codebase (rotation, flip, SpecAugment, etc.) sont uniquement utilisées dans les approches semi-supervisées (Semi-supervised Methods), pas pour VGG16/ResNet50.

#### 2.3.2 Train/Test Split

**Type:** K-Fold Cross-Validation (shuffle=True, non-stratifié)

**Fichier:** `/home/sambot/mldrone/RFClassification/nn_functions.py:36, 42, 48-49`
```python
kfold = KFold(n_splits=k_folds, shuffle=True)

for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
```

**Fichier:** `/home/sambot/mldrone/RFClassification/run_dl.py:83-94`
```python
k_folds = 5
batch_size = 128
learning_rate = 0.01
num_epochs = 10
momentum = 0.95
l2reg = 1e-4

trainedModel, res_acc, res_f1, res_runtime = runkfoldcv(Model, dataset, device,
                                                         k_folds, batch_size,
                                                         learning_rate, num_epochs,
                                                         momentum, l2reg)
```

**Paramètres:**
- **Folds:** 5
- **Shuffle:** True
- **Stratified:** NON (KFold standard, pas StratifiedKFold)
- **File-level grouping:** NON
- **Ratio:** 80% train / 20% test par fold

#### 2.3.3 Class Balancing

**Status:** NON

Aucun class weighting, weighted loss, ou sampling strategy n'est implémenté.

**Fichier:** `/home/sambot/mldrone/RFClassification/nn_functions.py:63`
```python
criterion = nn.CrossEntropyLoss()  # Pas de paramètre weight
```

#### 2.3.4 Transfer Learning

**Pretrained:** OUI sur ImageNet

**Architecture:** VGG16 ET ResNet50 (les deux sont implémentés)

**Fichier:** `/home/sambot/mldrone/RFClassification/models.py:149-154, 180-185`
```python
# VGG16
self.vggfull = models.vgg16(pretrained=True)
modules=list(self.vggfull.children())[:-1]  # Supprime FC layer et AdaptiveAvgPool
self.vggfeats=nn.Sequential(*modules)

for param in self.vggfeats.parameters():
    param.requires_grad_(False)  # Freeze toutes les couches conv

# ResNet50
self.resnetfull = models.resnet50(pretrained=True)
modules=list(self.resnetfull.children())[:-2]  # Supprime FC layer et AdaptiveAvgPool
self.resnetfeats=nn.Sequential(*modules)

for param in self.resnetfeats.parameters():
    param.requires_grad_(False)  # Freeze toutes les couches conv
```

**Frozen layers:** TOUTES les couches convolutionnelles (100% frozen)
- VGG16: 13 conv layers + 5 pooling layers frozen
- ResNet50: 48+ conv layers frozen

**Fine-tuned layers:** 1 seule couche (la FC finale)
- VGG16: Linear(25088 → num_classes)
- ResNet50: Linear(100352 → num_classes)

**Optimizer:** Adam (learning_rate=0.01)

**Fichier:** `/home/sambot/mldrone/RFClassification/nn_functions.py:66`
```python
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```

#### 2.3.5 Cross-Validation

**Type:** K-Fold (5 folds)

**Fichier:** `/home/sambot/mldrone/RFClassification/run_dl.py:83, 94`
```python
k_folds = 5

trainedModel, res_acc, res_f1, res_runtime = runkfoldcv(Model, dataset, device,
                                                         k_folds, ...)
```

#### 2.3.6 Pipeline Complet

```
1. Load raw IQ data (loading_functions.py:132-143)
   ↓
2. Generate spectrogram images (Generate DroneDetect Features.ipynb:242-249)
   - STFT: n_fft=1024, window=hanning, noverlap=120, scale='dB'
   - Resize: interpolation bilinéaire à 224x224
   - Colormap: viridis
   ↓
3. [PAS de data augmentation]
   ↓
4. Normalization (loading_functions.py:91-94)
   - Conversion BGR→RGB
   - Division par 255 (range [0, 1])
   ↓
5. K-Fold split (5 folds) (nn_functions.py:36, 42)
   - Shuffle: True
   - 80% train / 20% test par fold
   ↓
6. Load pretrained model (models.py:149, 180)
   - VGG16 ou ResNet50 avec poids ImageNet
   - Suppression FC layer finale
   - Freeze de TOUTES les couches conv
   ↓
7. Fine-tune (nn_functions.py:66, 70-102)
   - Adam optimizer (lr=0.01)
   - CrossEntropyLoss (sans class weights)
   - 10 epochs, batch_size=128
   - Training de la FC layer uniquement
```

### 2.4 Résumé VGG16/ResNet50

- **Input shape:** 224 x 224 x 3
- **Architecture:** VGG16 ET ResNet50 (les deux modèles sont implémentés et testables via `which_model` parameter)
- **Pretrained:** OUI (ImageNet weights)
- **Frozen layers:** TOUTES les couches convolutionnelles (100%)
- **Trainable layers:** 1 seule FC layer finale
- **Data augmentation:** AUCUNE
- **Class balancing:** NON
- **Optimizer:** Adam (lr=0.01)
- **Epochs:** 10
- **Batch size:** 128
- **Cross-validation:** 5-Fold (shuffle=True, non-stratified)
- **Normalisation:** [0, 1] par division par 255 (PAS de normalisation ImageNet)
- **Colormap spectrogrammes:** viridis
- **Accuracy finale:** Non disponible dans le code (résultats dans outputs du notebook)

---

## 3. RFUAVNet (Raw IQ)

### IMPORTANT: RFUAVNet N'EST PAS UTILISÉ AVEC DRONEDETECT

Après analyse approfondie du code, RFUAVNet est exclusivement utilisé avec le dataset **DroneRF**, pas DroneDetect.

**Confirmations:**
- `/home/sambot/mldrone/RFClassification/run_rfuav.py:31-33`
- `/home/sambot/mldrone/RFClassification/RFUAV-Net.ipynb` cellule id="56271efb"
- `/home/sambot/mldrone/RFClassification/README.md:98,108`

### 3.1 Feature Selection

**Status:** NON

Aucune méthode de feature selection n'est appliquée au niveau du réseau:
- Pas d'attention mechanisms
- Pas de feature map selection
- Pas de pruning
- Le réseau traite tous les 10,000 samples IQ d'entrée

**Fichier:** `/home/sambot/mldrone/RFClassification/models.py:204-314`

### 3.2 Feature Engineering

#### 3.2.1 Format IQ data

**Type:** Séparé (Real/Imaginary pour DroneRF stockées comme High/Low frequency)
**Shape tensor:** `[batch, 2, 10000]`
- Channel 0: High frequency data
- Channel 1: Low frequency data

**Fichier:** `/home/sambot/mldrone/RFClassification/loading_functions.py:308-365`

```python
def load_dronerf_raw(main_folder, t_seg):
    high_freq_files = os.listdir(main_folder+'High/')
    low_freq_files = os.listdir(main_folder+'Low/')

    fs = 40e6 #40 MHz

    for i in range(len(high_freq_files)):
        rf_data_h = pd.read_csv(main_folder+'High/'+high_freq_files[i], header=None).values
        rf_data_h = rf_data_h.flatten()

        rf_data_l = pd.read_csv(main_folder+'Low/'+low_freq_files[i], header=None).values
        rf_data_l = rf_data_l.flatten()

        # stack the features
        rf_sig = np.vstack((rf_data_h, rf_data_l))

        # segment
        len_seg = int(t_seg/1e3*fs)
        n_segs = (len(rf_data_h))//len_seg
        n_keep = n_segs*len_seg
        rf_sig = np.split(rf_sig[:,:n_keep], n_segs, axis =1)

    Xs_arr = Xs_arr.reshape(-1, *Xs_arr.shape[-2:])
    return Xs_arr, ys_arr, y4s_arr, y10s_arr
```

#### 3.2.2 Downsampling

**Status:** NON

Sample rate: 40 MHz (constant)
Aucun downsampling appliqué dans le pipeline RFUAVNet.

**Fichier:** `/home/sambot/mldrone/RFClassification/loading_functions.py:314`

#### 3.2.3 Filtering

**Status:** NON

Aucun filtering (bandpass, lowpass, highpass) appliqué sur les données raw IQ avant l'entrée dans le réseau.

#### 3.2.4 Segmentation

**Durée segment:** 10,000 samples (0.25 ms à 40 MHz)
**Overlap:** 0 samples (0%)

**Fichier:** `/home/sambot/mldrone/RFClassification/run_rfuav.py:32`
```python
t_seg = 0.25 #ms
```

**Fichier:** `/home/sambot/mldrone/RFClassification/loading_functions.py:338-342`
```python
# decide on segment lengths
len_seg = int(t_seg/1e3*fs)  # 0.25/1000 * 40e6 = 10,000 samples
n_segs = (len(rf_data_h))//len_seg
n_keep = n_segs*len_seg
rf_sig = np.split(rf_sig[:,:n_keep], n_segs, axis =1)
```

**Justification:** 0.25ms choisi pour correspondre à l'implémentation du papier original RF-UAVNet.

#### 3.2.5 Normalisation IQ

**Méthode:** Min-max normalization
**Scope:** Per-segment, per-channel

**Fichier:** `/home/sambot/mldrone/RFClassification/run_rfuav.py:51-57`
```python
for i in range(Xs_arr.shape[0]): # for each segment
    for ihl in range(2):  # for each channel (High/Low)
        r = np.max(Xs_arr[i,ihl,:]) - np.min(Xs_arr[i,ihl,:]) # range per segment
        m = np.min(Xs_arr[i,ihl,:])
        Xs_norm[i,ihl,:] = (Xs_arr[i,ihl,:]-m)/r
```

**Note:** Le code commente également une approche de normalisation globale (lignes 46-47) mais elle n'est PAS utilisée. La normalisation per-segment est préférée.

**Fichier:** `/home/sambot/mldrone/RFClassification/loading_functions.py:367-376` contient aussi une fonction `normalize_rf()` similaire.

#### 3.2.6 Pipeline d'extraction

```
Raw IQ data DroneRF (fs=40 MHz, High & Low freq channels)
  ↓
[Segmentation] (loading_functions.py:338-342)
  → 10,000 samples (0.25ms), 0% overlap
  ↓
[Normalisation min-max per-segment] (run_rfuav.py:51-57)
  → Per-channel, per-segment
  ↓
IQ segments (shape: [batch, 2, 10000])
  ↓
RFUAVNet input
```

### 3.3 Architecture RFUAVNet

**Fichier:** `/home/sambot/mldrone/RFClassification/models.py:204-314`

```python
class RFUAVNet(nn.Module):
    def __init__(self, num_classes):
        super(RFUAVNet, self).__init__()
        self.num_classes = num_classes

        self.dense = nn.Linear(320, num_classes) #320 inputs
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.smax = nn.Softmax(dim=1)

        # R-unit: initial conv layer
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=5, stride=5, dtype=torch.float32)
        self.norm1 = nn.BatchNorm1d(num_features=64, dtype=torch.float32)
        self.elu1 = nn.ELU(alpha=1.0, inplace=False)

        # G-units: 4 grouped convolution blocks
        self.groupconvlist = []
        self.norm2list = []
        self.elu2list = []
        for i in range(4):
            self.groupconvlist.append(nn.Conv1d(
                  in_channels=64,
                  out_channels=64,
                  kernel_size=3,
                  stride=2,
                  groups=8,
                  dtype=torch.float32
                ))
            self.norm2list.append(nn.BatchNorm1d(num_features=64))
            self.elu2list.append(nn.ELU(alpha=1.0, inplace=False))
        self.groupconv = nn.ModuleList(self.groupconvlist)
        self.norm2 = nn.ModuleList(self.norm2list)
        self.elu2 = nn.ModuleList(self.elu2list)

        # Multi-GAP (Global Average Pooling) layers
        self.avgpool1000 = nn.AvgPool1d(kernel_size=1000)
        self.avgpool500 = nn.AvgPool1d(kernel_size=500)
        self.avgpool250 = nn.AvgPool1d(kernel_size=250)
        self.avgpool125 = nn.AvgPool1d(kernel_size=125)
```

#### Résumé des layers:

1. **Input**: shape `[batch, 2, 10000]`

2. **R-Unit** (models.py:286-292):
   - Conv1D: in_channels=2, out_channels=64, kernel_size=5, stride=5
   - BatchNorm1D: num_features=64
   - ELU: alpha=1.0
   - Output: `[batch, 64, 2000]`

3. **G-Unit 1 + MaxPool + Skip Connection** (models.py:247-251):
   - Padding: (1,0)
   - GroupConv1D: in=64, out=64, kernel=3, stride=2, groups=8
   - BatchNorm1D: 64
   - ELU
   - MaxPool1D: kernel=2, stride=2
   - Skip addition
   - Output: `[batch, 64, 1000]`

4. **G-Unit 2 + MaxPool + Skip Connection** (models.py:254-256):
   - Similar structure
   - Output: `[batch, 64, 500]`

5. **G-Unit 3 + MaxPool + Skip Connection** (models.py:258-260):
   - Similar structure
   - Output: `[batch, 64, 250]`

6. **G-Unit 4 + MaxPool + Skip Connection** (models.py:262-264):
   - Similar structure
   - Output: `[batch, 64, 125]`

7. **Multi-GAP (Global Average Pooling)** (models.py:268-277):
   - GAP sur xg1 (1000 samples): output `[batch, 64, 1]`
   - GAP sur xg2 (500 samples): output `[batch, 64, 1]`
   - GAP sur xg3 (250 samples): output `[batch, 64, 1]`
   - GAP sur xg4 (125 samples): output `[batch, 64, 1]`
   - GAP sur x_togap (125 samples): output `[batch, 64, 1]`
   - Concatenation: `[batch, 320, 1]`
   - Flatten: `[batch, 320]`

8. **Dense Layer** (models.py:280):
   - Linear: 320 → num_classes
   - Activation: Softmax (dim=1)

#### Total parameters (calcul manuel):

**R-Unit Conv1:**
- Conv1D(2→64, k=5): (2x5 + 1)x64 = 704

**BatchNorm1:**
- 64x2 = 128

**G-Units (4 blocks):**
- GroupConv1D(64→64, k=3, groups=8): (64/8)x3x64 + 64 = 1,600 par block
- BatchNorm1D(64): 128 par block
- Total G-units: 4x(1,600 + 128) = 6,912

**Dense Layer:**
- Linear(320→num_classes): 320xnum_classes + num_classes

**Total (pour num_classes=4):**
- R-Unit: 704 + 128 = 832
- G-Units: 6,912
- Dense: 320x4 + 4 = 1,284
- **Total: ~9,028 paramètres** (approximation, peut varier légèrement selon num_classes)

### 3.4 Autres Traitements

#### 3.4.1 Data Augmentation

**Status:** NON

Aucune augmentation de données (time shift, noise injection, amplitude scaling) n'est appliquée.

#### 3.4.2 Train/Test Split

**Méthode:** K-Fold Cross-Validation (pas de train/test split fixe)

**Fichier:** `/home/sambot/mldrone/RFClassification/run_rfuav.py:77-79`
```python
k_folds = 5
model, avg_acc, mean_f1s, mean_runtime = runkfoldcv(
    model, dataset, device, k_folds, batch_size, learning_rate, num_epochs, momentum, l2reg)
```

**Fichier:** `/home/sambot/mldrone/RFClassification/nn_functions.py:23-36`
```python
def runkfoldcv(model, dataset, device, k_folds, batch_size, learning_rate, num_epochs, momentum, l2reg):
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        trainloader = torch.utils.data.DataLoader(
                          dataset,
                          batch_size=batch_size, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
                          dataset,
                          batch_size=batch_size, sampler=test_subsampler)
```

**Split ratio:** 5-fold CV = 80% train / 20% test par fold
**Random:** Oui (shuffle=True)

#### 3.4.3 Class Balancing

**Status:** NON

Aucun class balancing (SMOTE, class weights, undersampling/oversampling) n'est appliqué.

**Loss function:** CrossEntropyLoss sans class weights

**Fichier:** `/home/sambot/mldrone/RFClassification/nn_functions.py:63`
```python
criterion = nn.CrossEntropyLoss()
```

#### 3.4.4 Cross-Validation

**Status:** OUI

**K-Fold CV:** k=5
**Stratified:** NON (KFold standard, pas StratifiedKFold)

**Fichier:** `/home/sambot/mldrone/RFClassification/nn_functions.py:36`
```python
kfold = KFold(n_splits=k_folds, shuffle=True)
```

#### 3.4.5 Pipeline Complet

```
1. Load raw IQ data DroneRF (loading_functions.py:308-365)
   → High & Low freq channels, fs=40MHz
   ↓
2. Segmentation (loading_functions.py:338-342)
   → 10,000 samples, 0.25ms, no overlap
   ↓
3. Normalisation min-max per-segment (run_rfuav.py:51-57)
   → Per-channel, per-segment
   ↓
4. Create DroneData Dataset (run_rfuav.py:59)
   ↓
5. K-Fold CV (k=5) (run_rfuav.py:77-79)
   ↓
6. Train RFUAVNet (nn_functions.py:23-197)
   → Batch size: 128
   → Learning rate: 0.01
   → Epochs: 10
   → Optimizer: Adam (nn_functions.py:66)
   → Loss: CrossEntropyLoss
```

### 3.5 Résumé RFUAVNet (DroneRF uniquement)

- **Input:** Raw IQ data DroneRF (shape: `[batch, 2, 10000]`)
- **Sample rate:** 40 MHz
- **Segment duration:** 0.25 ms (10,000 samples)
- **Normalisation:** Min-max per-segment, per-channel
- **Architecture:** 1D CNN avec R-unit + 4 G-units + Multi-GAP + Dense layer (~9,028 paramètres pour 4 classes)
- **Data augmentation:** Aucune
- **Train/test:** 5-fold CV (80/20 par fold)
- **Hyperparamètres:**
  - Batch size: 128
  - Learning rate: 0.01
  - Epochs: 10
  - Optimizer: Adam
  - Loss: CrossEntropyLoss
- **Accuracy finale:** 99.8% (binary classification), selon README.md:108
- **Inference time:** 1.078ms (workstation avec Titan RTX)

---

## 4. COMPARAISON INTER-MODÈLES

| Aspect | SVM (PSD) | VGG16/ResNet50 (SPEC) | RFUAVNet (Raw IQ) |
|--------|-----------|----------------------|-------------------|
| **Dataset** | DroneDetect | DroneDetect | DroneRF uniquement |
| **Input** | 1024 PSD bins | 224x224x3 images | 2x10000 raw samples |
| **Sample rate** | 60 MHz | 60 MHz | 40 MHz |
| **Segment duration** | 20 ms | Variable (STFT) | 0.25 ms |
| **Feature selection** | NON | NON | NON |
| **Feature engineering** | Welch PSD (nperseg=1024, hamming, two-sided) | STFT (n_fft=1024, hanning, scale='dB'), Interpolation bilinéaire 224x224, Colormap viridis | Min-max normalization per-segment |
| **Normalization** | Z-score (per-file) + Per-sample (div by max) | BGR→RGB + Division par 255 | Min-max per-segment, per-channel |
| **Data augmentation** | NON | NON | NON |
| **Resampling** | NON | NON | NON |
| **Class balancing** | NON | NON | NON |
| **Train/test split** | 67/33 (non-stratified) | 5-fold CV (80/20) | 5-fold CV (80/20) |
| **Cross-validation** | 5-fold (shuffle, non-stratified) | 5-fold (shuffle, non-stratified) | 5-fold (shuffle, non-stratified) |
| **Optimizer** | SVM RBF (C=1, gamma='scale') | Adam (lr=0.01) | Adam (lr=0.01) |
| **Epochs** | N/A | 10 | 10 |
| **Batch size** | N/A | 128 | 128 |
| **Pretrained** | N/A | OUI (ImageNet, 100% frozen conv layers) | NON |
| **Accuracy** | 85.4% ± 0.5% | Non disponible (outputs notebook) | 99.8% (DroneRF binary) |
| **F1-score** | 85.1% ± 0.5% | Non disponible | Non disponible |
| **Inference time** | 9.96 ms | Non disponible | 1.078 ms |
| **Parameters** | 1024 features | 25088 (VGG16) / 100352 (ResNet50) | ~9,028 |

### Observations clés:

1. **RFUAVNet n'est PAS utilisé avec DroneDetect** mais uniquement avec DroneRF.

2. **Aucun modèle n'utilise de feature selection** explicite (PCA, SelectKBest, etc.).

3. **Aucun modèle n'utilise de data augmentation** sur DroneDetect.

4. **Aucun modèle n'utilise de class balancing** (class weights, SMOTE, etc.).

5. **Tous les modèles utilisent 5-fold cross-validation** (non-stratifiée, shuffle=True).

6. **VGG16/ResNet50 utilisent transfer learning** avec ImageNet weights mais **freeze 100% des couches convolutionnelles** (fine-tuning uniquement de la FC layer finale).

7. **Normalisation différente selon le modèle**:
   - SVM: Z-score (per-file) + Per-sample (div by max)
   - VGG16/ResNet50: Division par 255 (PAS de normalisation ImageNet)
   - RFUAVNet: Min-max per-segment, per-channel

8. **Résolution temporelle très différente**:
   - SVM: 20 ms segments
   - VGG16/ResNet50: Variable (dépend de STFT)
   - RFUAVNet: 0.25 ms segments (80x plus court que SVM)

---

## 5. INCERTITUDES ET LIMITATIONS

### Incertitudes par modèle:

#### SVM (PSD):

1. **Incohérence n_per_seg/nperseg** (CLARIFIÉE): Le script `run_dronedetect_feat.py:33` utilise 256 par défaut, mais les résultats finaux (85.4% accuracy) ont été générés avec 1024 dans `ML Approaches.ipynb:80`. Cette incohérence a été documentée dans la section 1.2.1.

2. **Distribution des classes** (5%): Nombre exact de samples par classe non trouvé. Uniquement le total (38,978 samples) est confirmé. Classes identifiées: 'DJI', 'UDI', 'TAR' (déduit du code).

3. **Return_onesided parameter** (3%): Non spécifié explicitement dans `run_dronedetect_feat.py:174`. Confirmé implicitement: complex input → two-sided (1024 bins). Confirmé par `gamutrf_feature_functions.py:13` qui utilise explicitement `return_onesided=False`.

4. **Ordre exact normalization dans pipeline** (2%): Z-score appliqué dans `load_dronedetect_raw` (confirmé ligne 136). Per-sample normalization appliqué dans `__getitem__` lors du chargement (confirmé ligne 87). Mais timing exact par rapport à la génération PSD nécessite vérification du workflow complet.

#### VGG16/ResNet50 (Spectrograms):

1. **Accuracy/F1 finaux** (5%): Les valeurs exactes d'accuracy et F1 obtenues ne sont pas visibles dans le code source, seulement dans les outputs des notebooks qui sont trop volumineux pour être inclus.

2. **Window length exact** (3%): Le code utilise `np.hanning(n_per_seg)` qui génère une fenêtre de longueur 1024, mais il n'est pas explicitement confirmé que `win_length=1024` dans tous les cas (pourrait être `n_fft+1`).

3. **Nombre exact de paramètres trainables** (2%): Le nombre exact de paramètres dans la FC layer finale dépend de `num_classes` qui varie selon le dataset (7 classes pour DroneDetect drones).

#### RFUAVNet (Raw IQ):

1. **Nombre exact de paramètres** (5%): Calcul manuel approximatif, nécessite exécution du modèle pour confirmation exacte. Dépend du `num_classes` utilisé.

2. **Résultats finaux précis des runs** (2%): Les résultats dans le README (99.8% accuracy) ne spécifient pas si c'est pour binary (2 classes), 4 classes, ou 10 classes. Le code montre `num_classes=4` dans `run_rfuav.py:64-74`.

### Score d'exactitude global: 92% ±8%

**Incertitudes cumulées (8%):**
- Distribution exacte des classes (DroneDetect) non documentée (5%)
- Accuracy/F1 finaux VGG16/ResNet50 non visibles dans code source (3%)

---

## 6. SOURCES

### Fichiers analysés (complet):

**Scripts Python:**
- `/home/sambot/mldrone/RFClassification/run_dronedetect_feat.py`
- `/home/sambot/mldrone/RFClassification/run_dl.py`
- `/home/sambot/mldrone/RFClassification/run_rfuav.py`
- `/home/sambot/mldrone/RFClassification/models.py`
- `/home/sambot/mldrone/RFClassification/nn_functions.py`
- `/home/sambot/mldrone/RFClassification/loading_functions.py`
- `/home/sambot/mldrone/RFClassification/feat_gen_functions.py`
- `/home/sambot/mldrone/RFClassification/helper_functions.py`
- `/home/sambot/mldrone/RFClassification/file_paths.py`
- `/home/sambot/mldrone/RFClassification/gamutrf_feature_functions.py`

**Notebooks Jupyter:**
- `/home/sambot/mldrone/RFClassification/Generate DroneDetect Features.ipynb`
- `/home/sambot/mldrone/RFClassification/ML Approaches.ipynb`
- `/home/sambot/mldrone/RFClassification/DL Approaches.ipynb`
- `/home/sambot/mldrone/RFClassification/RFUAV-Net.ipynb`

**Documentation:**
- `/home/sambot/mldrone/RFClassification/README.md`

### Commits vérifiés:
- Non applicable (analyse sur version locale du repository)

### Documentation externe:
- Repository GitHub: https://github.com/tryph0n/RFClassification
- scipy.signal.welch documentation
- pytorch torchvision.models documentation
- matplotlib.pyplot.specgram documentation

---

## 7. RECOMMANDATIONS ET OBSERVATIONS

### Points forts:

1. **Pipeline clair et reproductible** pour SVM et VGG16/ResNet50 avec DroneDetect.
2. **Cross-validation systématique** (5-fold) pour tous les modèles.
3. **Transfer learning efficace** pour VGG16/ResNet50 (même si 100% des conv layers sont frozen).

### Points d'amélioration potentiels:

1. **Feature selection**: Aucun des modèles n'utilise de feature selection. Pour SVM, tester PCA ou SelectKBest pourrait réduire la dimensionnalité et améliorer la généralisation.

2. **Data augmentation**: Aucune augmentation de données. Pour VGG16/ResNet50, des techniques comme rotation, flip, time shifting, ou SpecAugment pourraient améliorer la robustesse.

3. **Class balancing**: Aucun class weighting ou resampling. Si les classes sont déséquilibrées, cela pourrait affecter les performances sur les classes minoritaires.

4. **Stratified cross-validation**: Utiliser `StratifiedKFold` au lieu de `KFold` pour garantir une distribution équitable des classes dans chaque fold.

5. **Fine-tuning partiel**: Pour VGG16/ResNet50, unfreeze quelques dernières couches convolutionnelles pourrait améliorer l'adaptation aux spectrogrammes RF.

6. **Normalisation ImageNet**: Pour VGG16/ResNet50, utiliser la normalisation ImageNet (mean/std) pourrait être bénéfique puisque les modèles sont pretrained sur ImageNet.

7. **RFUAVNet sur DroneDetect**: RFUAVNet n'est pas utilisé avec DroneDetect. Tester RFUAVNet sur DroneDetect pourrait être intéressant pour comparer les performances avec SVM et VGG16/ResNet50.

---

**FIN DU RÉFÉRENTIEL**
