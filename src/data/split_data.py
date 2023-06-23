import os
import numpy as np
import pandas as pd

# --- Arguments ---
dataset = "SIAR"
split_ratio = [0.8, 0.1, 0.1]
split_name = f"split-2_{int(split_ratio[0]*10)}_{int(split_ratio[1]*10)}_{int(split_ratio[2]*10)}"
# -----------------

data_dir = f"../data/{dataset}"

df = pd.DataFrame(os.listdir(data_dir), columns=["id"])

train, validate, test = np.split(
    df.sample(frac=1),
    [int(split_ratio[0] * len(df)), int((1 - split_ratio[1]) * len(df))],
)

data_splits_dir = os.path.join(data_dir, "data_splits", split_name)

train.to_csv(os.path.join(data_splits_dir, "train.csv"), index=False)
test.to_csv(os.path.join(data_splits_dir, "test.csv"), index=False)
validate.to_csv(os.path.join(data_splits_dir, "val.csv"), index=False)
