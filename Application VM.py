import numpy as np
import pandas as pd
from intensive import FairSplit

# %% Input the demonstrations
demonstration = 1  # 1 or 2
n_realizations = 10  # number of training and test sets. Use one for your application
xdir = 'X_Coord_SMDA'  # the name of the column that contains the X direction
ydir = 'Y_Coord_SMDA'  # the name of the column that contains the Y direction
feature = 'TOC_weight_pct'

vario_model = {
    'nug': 0.0019066257660920183,
    'nst': 2,
    'it1': 1,
    'cc1': 0.6324522412942912,
    'azi1': 76.80758405823096,
    'hmaj1': 3020.6789239768227,
    'hmin1': 11232.347296625365,
    'it2': 2,
    'cc2': 0.36564113293961675,
    'azi2': 76.80758405823096,
    'hmaj2': 74871.58932238017,
    'hmin2': 33581.04567526035
}

# %% read the data
training = pd.read_csv('D:\\PyCharm projects\\FairTrainTest\\Files\\Datasets/training_VM.csv',
                       dtype={xdir: float, ydir: float})

# training.rename(columns={"UWI": "UWI1"}, inplace=True)
training.reset_index(inplace=True)  # use the well index as uwi
training = training.rename(columns={'index': 'UWI'})

real_world = pd.read_csv('D:\\PyCharm projects\\FairTrainTest\\Files\\Datasets/real_world_VM.csv',
                         dtype={xdir: float, ydir: float})

# real_world.rename(columns={"UWI": "UWI1"}, inplace=True)
real_world.reset_index(inplace=True)  # use the well index as uwi
real_world = real_world.rename(columns={'index': 'UWI'})

# %%
modelo = FairSplit(training, xdir, ydir, feature, 500)
output = "D:\\PyCharm projects\\FairTrainTest/kbd2dexe"
kmap, vmap = modelo.s_kriging(vario_model, output)

#%%
import plotly.express as px
fig = px.imshow(kmap)
fig.show()

# TODO make kb2d work
