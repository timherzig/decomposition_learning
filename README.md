# 🚀 Decomposition learning 
We propose a novel approach to learn a decomposition of an input sequence into (i) reconstructed base image (ii) light masks (iii) shadow masks (iv) occlusion masks.

[Insert image of framework here]

## 💡 Installation
In root folder:
```bash
conda create -p ./venv python=3.9
conda activate ./venv
pip install -r requirements.txt
```

Stay in the root folder.
You can now activate the environment by running
```bash
source ./venv/bin/activate
```

## 📦 Data preparation
The root directory should contain the following contents:
- `<ROOT>/data/SIAR`

> You can download the data from here (TODO: add link)

## 🏋🏽‍♂️ Training
Run the following command to train the model:
```bash
python train.py --config configs/default.yaml
```