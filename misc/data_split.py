import os
import numpy as np
import pandas as pd

dir = '/Users/tim/CS_master/Computer_Vision_Project/SIAR'

df = pd.DataFrame(os.listdir(dir), columns=['id'])

train, validate, test = np.split(df.sample(frac=1), [int(.8*len(df)), int(.9*len(df))])

train.to_csv(os.path.join(dir, 'train.csv'), index=False)
test.to_csv(os.path.join(dir, 'test.csv'), index=False)
validate.to_csv(os.path.join(dir, 'val.csv'), index=False)
