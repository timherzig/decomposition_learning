# 🚀 Decomposition learning 
We propose **Decomposer**, a novel approach to learn a decomposition of an input sequence into (i) reconstructed original image (ii) light masks (iii) shadow masks (iv) occlusion masks.

<!-- inster image -->
![Decomposer ](/reports/figures/3-Advanced_Decomposition_Learning/network.png)

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