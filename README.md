# Is Geometry Enough? An Evaluation of Landmark-Based Gaze Estimation

This repository contains the official implementation for the paper "Is Geometry Enough? An Evaluation of Landmark-Based Gaze Estimation" (https://arxiv.org/abs/2603.24724). It provides a lightweight, interpretable pipeline for estimating human gaze using sparse geometric features (3D facial landmarks) instead of raw image pixels.
The codebase includes scripts for extracting landmarks via MediaPipe, normalizing the 3D geometry, and training/evaluating three different regression models (Holistic MLP, Siamese MLP, and XGBoost) across ETH-XGaze, Gaze360, GazeGene, and Blender-based synthetic CSV datasets.

## 📊 Data Availability

The pre-processed landmark datasets (approx. 2.5GB) are too large to host directly in the main branch.

- Download the Data: You can find the extracted `.csv` files in the realease section of this repository (https://github.com/daniele-agostinelli/LandmarkGaze/releases/tag/data-v1).
- Once downloaded, place the `.csv` files into the `datasets/` directory maintaining the folder structure required by the training scripts (e.g., `datasets/XGaze_448/`, `datasets/Gaze360/`).

## 📂 Repository Structure
The repository is modularized into data extraction, model training, benchmarking, and core utility classes. Note that multiple versions of the top-level scripts exist to handle the specific formats of different datasets and models.

### 1. Data Extraction
Scripts used to process raw image datasets, extract MediaPipe landmarks, perform 3D head pose estimation (PnP), and apply perspective normalization.

- `ExtractDataset_xgaze.py` (ETH-XGaze)
- `ExtractDataset_gaze360.py` (Gaze360)
- `ExtractDataset_gazegene.py` (GazeGene)

### 2. Training
Scripts to train the regressors on the normalized landmark coordinates. The training CLIs now accept dataset aliases (`gaze360`, `gazegene`, `xgaze`, `blender`), explicit CSV paths, and optional multi-GPU execution via `torch.nn.DataParallel`.

- `Train_MLP.py` (Trains the standard Holistic MLP)
- `Train_siameseMLP.py` (Trains the binocular Siamese MLP)
- `Train_XGBoost.py` (Trains the XGBoost regressor)
- `Split_Dataset_subjectwise.py` (Creates subject-wise TRAIN/VALID/TEST splits, streaming large CSVs chunk-by-chunk)

### 3. Evaluation & Benchmarking
Scripts to run within-domain and cross-domain evaluations, generating comprehensive performance statistics (`.csv` outputs).

- `Test_on_XGaze.py`
- `Test_on_Gaze360.py`
- `Test_on_GazeGene.py`
- `Run_CrossDomain_MLP.py`
- `Run_CrossDomain_Siamese.py`
- `Run_CrossDomain_XGBoost.py`

### 4. Core Modules & Inference
The underlying classes handling the math, geometric modeling, and estimation pipelines:

- `gaze_estimator_normalized.py` (and its `_siamese` and `_XGBoost` variants): The main inference wrappers that pipe features into the trained models.
- `normalization_utils.py`: Applies the Zhang et al. (2018) perspective warping to map physical cameras to the normalized virtual space.
- `face_landmark_estimator.py`: Wrapper for mediapipe.solutions.face_mesh with padding logic for robust detection.
- `face_model.py`: Defines the 3D semantic layout of the face and anchor points.
- `camera.py`: Handles camera intrinsics, distortion coefficients, and 3D projection.

## 🛠️ Prerequisites
Source codes were developed and tested with Python 3.12.3 on Ubuntu 24.04 and libraries as detailed in `requirements.txt`.
You can install the default environment via
```bash
pip install -r requirements.txt
```

For Linux/NVIDIA remote training there is also a dedicated profile:
```bash
pip install -r requirements-gpu.txt
```

## 🚀 Usage
### 1. Prepare the Data
Either download the pre-processed `.csv` files from the Releases tab or generate them from scratch using the extraction scripts (in this case original datasets will be needed).

### 2. Split Blender Synthetic Data
If you want to train on `datasets/Blender - synthetic/normalized_dataset.csv`, first create subject-wise splits:
```bash
python Split_Dataset_subjectwise.py --input-csv "datasets/Blender - synthetic/normalized_dataset.csv"
```

By default this writes:
- `datasets/Blender - synthetic/normalized_dataset_TRAIN.csv`
- `datasets/Blender - synthetic/normalized_dataset_VALID.csv`
- `datasets/Blender - synthetic/normalized_dataset_TEST.csv`

### 3. Train on Blender
Holistic MLP:
```bash
python Train_MLP.py --dataset blender --model-save-path models/blender_MLP.pth
```

Siamese MLP:
```bash
python Train_siameseMLP.py --dataset blender --model-save-path models/blender_siameseMLP.pth
```

XGBoost:
```bash
python Train_XGBoost.py --dataset blender
```

To use all visible GPUs on Linux:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python Train_MLP.py --dataset blender --device cuda --multi-gpu
```

### 4. Run Cross-Domain Validation
After training on Blender, validate the new model on Gaze360, GazeGene, and XGaze:
```bash
python Run_CrossDomain_MLP.py --model-path models/blender_MLP.pth --source-dataset blender
python Run_CrossDomain_Siamese.py --model-path models/blender_siameseMLP.pth --source-dataset blender
python Run_CrossDomain_XGBoost.py --model-path models/blender_xgboost.pkl --source-dataset blender
```

### 5. Remote Linux Batch Execution
For a remote Linux machine with 7 NVIDIA GPUs, the repository now includes:
- `scripts/run_blender_remote_batch_7gpu.sh`
- `scripts/run_blender_remote_tmux.sh`

Example:
```bash
bash scripts/run_blender_remote_batch_7gpu.sh
```

Or launch it inside `tmux`:
```bash
bash scripts/run_blender_remote_tmux.sh
```

## 📝 Citation
If you find this code or our methodology useful in your research, please consider citing our paper: https://arxiv.org/abs/2603.24724
