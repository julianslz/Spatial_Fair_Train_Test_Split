import pickle
import numpy as np
import pandas as pd
from spatial_fair_split2 import SpatialFairSplit, plot_3_realizations, PublicationImages

# %% Input the demonstrations
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
training.rename(columns={"UWI": "UWI1"}, inplace=True)
training.reset_index(inplace=True)  # use the well index as uwi
training = training.rename(columns={'index': 'UWI'})

real_world = pd.read_csv('D:\\PyCharm projects\\FairTrainTest\\Files\\Datasets/real_world_VM.csv',
                         dtype={xdir: float, ydir: float})
real_world.rename(columns={"UWI": "UWI1"}, inplace=True)
real_world.reset_index(inplace=True)  # use the well index as uwi
real_world = real_world.rename(columns={'index': 'UWI'})

# %%
modelo = SpatialFairSplit(
    training, real_world, vario_model, xdir=xdir, ydir=ydir, number_bins=5
)

# %%
# sfs_train, sfs_test, sfs_kvar = modelo.fair_sets_realizations(100)
# with open("fair_train.txt", "wb") as fp:  # Pickling
#     pickle.dump(sfs_train, fp)
#
# with open("fair_test.txt", "wb") as fp:  # Pickling
#     pickle.dump(sfs_test, fp)
#
# np.save("fair_kvar", sfs_kvar)

# %%
vsa_train, vsa_test, vsa_kvar, spatial_cv, spatial_cv_kvar = modelo.create_other_sets(realizations=100)

# %%
with open("vsa_train.txt", "wb") as fp:  # Pickling
    pickle.dump(vsa_train, fp)

with open("vsa_test.txt", "wb") as fp:  # Pickling
    pickle.dump(vsa_test, fp)

with open("spatial_cv.txt", "wb") as fp:  # Pickling
    pickle.dump(spatial_cv, fp)
# %%
# For comparison purposes, compute other sets with different cross-validation methods: the validation set approach (vsa)
# and spatial cross-validation. Moreover, get the kriging variance distribution of each

# %%
plot_3_realizations(
    fair_train=sfs_train,
    fair_test=sfs_test,
    rand_train=vsa_train,
    rand_test=vsa_test,
    spatial_cv=spatial_cv,
    real_world_set=real_world,
    realiz=90,
    xdir=xdir,
    ydir=ydir,
    xmin=2.42e6,
    ymin=5.65e6,
    xmax=2.57e6,
    ymax=5.89e6,
)

# %%
images = PublicationImages(
    test_kvar_random=vsa_kvar,
    test_kvar_fair=sfs_kvar,
    test_kvar_spatial=spatial_cv_kvar,
    rw_kvar=modelo.rw_krig_var
)
# %% Plot the kernel density estimates of the three cross-validations methods: Figures 4 and 8
plot = images.kde_plots(5.0)

# %% Plot the violin plots of the divergence metrics: Figures 5 and 9
plot2 = images.divergence_violins()

