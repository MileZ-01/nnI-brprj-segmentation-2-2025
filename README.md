# nnInteractive CT Segmentation

This repository contains the training and evaluation pipeline for **nnInteractive**, based on **nnU-Net**, for industrial CT scan segmentation tasks.

---

## Table of Contents

- [Installation](#installation)
- [Dataset Structure](#datasetsTr-structure)
- [Environment Variables](#environment-variables)
- [Training](#training)
- [Evaluation](#evaluation)


---

## Installation

### Using pip

```bash
python -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### Using conda

```bash
conda env create -f environment.yml
conda activate nninteractive
```

## DatasetsTr Structure

```bash
Dataset001_CT_Scans/
├── imagesTr/
│   ├── case_001_0000.nii.gz
│   └── ...
├── labelsTr/
│   ├── case_001.nii.gz
│   └── ...
└── dataset.json
```

## Environment variables

### Using Windows 
```bash
set nnUNet_raw=./Datasets_Tr
set nnUNet_preprocessed=./Datasets_preprocessed
set nnUNet_results=./Datasets_results
```

### Other

```bash
export nnUNet_raw=./Datasets_Tr
export nnUNet_preprocessed=./Datasets_preprocessed
export nnUNet_results=./Datasets_results
```

## Run training 

```bash 
python ./scripts/train.py
```

## Run evaluation
```bash
python ./scripts/evaluate.py
```




