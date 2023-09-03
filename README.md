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
You can now activate the environment anytime by running
```bash
conda activate ./venv
```

## 📦 Data and weights
You can download the entire training dataset, eval dataset, generated occlusion pseudo targets, and weights from [here](https://tubcloud.tu-berlin.de/s/m7qPnWkK3FAmPqo).\
The password is "< Abreviation of course name >_2023!".

We provide intuitive [config files](./config/default.yaml) for setting up training and evaluation where you can manually provide the path to the data and weights.\
Nevertheless, the data should be structured as folows:
```
├── data
│   ├── Eval
│   ├── SIAR
│   ├── SIAR_OCC
```

Make sure that all paths in the config files are correct for your system.

## 🏋🏽‍♂️ Training
We propose to pretrain all individual part of our model separately.\
`Run the following commands to train the individual parts:`
### Enocder pretraining
```bash
python train.py --config "config/swin_pretrain.yaml"
```

### Original image reconstruction branch pretraining
```bash
python train.py --config "config/oi_pretraining.yaml"
```

### Shadow and light branch pretraining
```bash
python train.py --config "config/sl_pretraining.yaml"
```

### Occlusion branch pretraining
```bash
python train.py --config "config/occ_binary_pretraining.yaml"
```

### Joint training
```bash
python train.py --config "config/default.yaml"
```

## 📊 Evaluation
To evaluate the model, first run the following command to generate and save the models predicitons:
```bash
python eval.py --config "config/evaluation_script.yaml"
```
Then step through the notebook [`notebooks/evaluation.ipynb`](./notebooks/evaluation.ipynb) to generate the evaluation metrics.