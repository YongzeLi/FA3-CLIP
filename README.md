# FA³-CLIP: Frequency-Aware and Attack-Agnostic CLIP for Unified Face Attack Detection

[**Paper**](https://arxiv.org/pdf/2504.00454?) | [**Pretrained Models**](#) *(coming soon)*

## 📌 Introduction

**FA³-CLIP** is a novel framework that unifies digital and physical face attack detection via frequency-aware feature guidance and attack-agnostic prompt learning. This repository contains the official implementation of the experiments in our paper *"FA³-CLIP: Frequency-Aware and Attack-Agnostic CLIP for Unified Face Attack Detection"*.

## 📁 Directory Overview

- `clip/`: CLIP backbone and modifications
- `configs/`: YAML configuration files
- `datasets/`: Dataset loading and processing
- `losses/`: Custom loss functions
- `trainers/`: Training pipeline
- `util/`: Evaluation and metrics
- `scripts/`: Shell scripts to launch training/inference
- `Dassl.pytorch/`: Modified version of [DASSL.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) with our custom changes

## 🚀 Getting Started

### 1. Clone the Repository and Create Environment

```bash
git clone https://github.com/YongzeLi/FA3-CLIP.git
cd fa3-clip
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/

conda create -y -n your_env python=3.8
conda activate your_env

conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt

python setup.py develop

cd ..
pip install -r requirements.txt
```

### 2. Datasets

This project is evaluated on two datasets:

- **UniAttackData**  
  - 📄 *Unified Physical-Digital Face Attack Detection*, IJCAI 2024  
  - 🔗 [Paper](https://www.ijcai.org/proceedings/2024/0083.pdf)  
  - 📂 [Github Page](https://github.com/liuajian/CASIA-FAS-Dataset/tree/main/UniAttackData)

- **JFSFDB**  
  - 📄 *Benchmarking Joint Face Spoofing and Forgery Detection With Visual and Physiological Cues*, IEEE TDSC 2024  
  - 🔗 [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10387780)  
  - 📂 [GitHub Page](https://github.com/ZitongYu/Benchmarking/tree/main)

### 3. Run Training

```bash
Train:
bash scripts/train.sh

Val checkpoint:
bash scripts/val.sh
```

### 4. Pretrained Models

```bash
We will release pretrained weights soon.
```

