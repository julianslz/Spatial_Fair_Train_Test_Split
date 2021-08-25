import numpy as np
import pandas as pd
from geostatspy.GSLIB import make_variogram
from geostatspy.geostats import kb2d

import spatial_fair_split as sfs

# %% Input the demonstrations
demonstration = 2  # 1 or 2
n_realizations = 2  # number of training and test sets. Use one for your application
xdir = 'X'  # the name of the column that contains the X direction
ydir = 'Y'  # the name of the column that contains the Y direction

# %% read the data
training = pd.read_csv("demo" + str(demonstration) + "_train.csv", dtype={'X': float, 'Y': float})
training.reset_index(inplace=True)  # use the well index as uwi
training = training.rename(columns={'index': 'UWI'})

real_world = pd.read_csv("demo" + str(demonstration) + "_rw.csv", dtype={'X': float, 'Y': float})
real_world.reset_index(inplace=True)  # use the well index as uwi
real_world = real_world.rename(columns={'index': 'UWI'})

# %% Geostatistical setup and definitions. Variogram model using GSLIB convention
vario = make_variogram(
    nug=0.0, nst=1,
    it1=1, cc1=1.0, azi1=0, hmaj1=250, hmin1=157,
    it2=3, cc2=0.0, azi2=0, hmaj2=250, hmin2=157
)

# Compute the rotational matrices for kriging
rotmat1, rotmat2, rotmat3, rotmat4, maxcov = sfs.setup_rotmat(vario['nug'], vario['cc1'], vario['azi1'])

# Define the modeling dictionary
dictionary_model = {
    "anis": vario.get('hmin1') / vario.get('hmaj1'),
    "cc": vario.get("cc1"),
    "aa": vario.get("hmaj1"),
    "nug": vario.get("nug"),
    "rotmat1": rotmat1,
    "rotmat2": rotmat2,
    "rotmat3": rotmat3,
    "rotmat4": rotmat4,
    "maxcov": maxcov
}

# %%
# instantiate the object
fair_cv = sfs.SpatialFairSplit(training, real_world, dictionary_model, xdir=xdir, ydir=ydir)

# obtain the realizations of training and test sets from spatial fair split (sfs). Get the kriging variance
# distribution too
sfs_train, sfs_test, sfs_kvar = fair_cv.fair_sets_realizations(n_realizations)

# sfs_train are the training sets
# sfs_test are the test sets
# sfs_vkar is the kriging variance of sfs_test using sfs_train

# %%
# For comparison purposes, compute other sets with different cross-validation methods: the validation set approach (vsa)
# and spatial cross-validation. Moreover, get the kriging variance distribution of each
vsa_train, vsa_test, vsa_kvar, spatial_cv, spatial_cv_kvar = fair_cv.create_other_sets(n_realizations)

# %% Estimate the kriging variance
_, vmap = kb2d(
    df=training,
    xcol='X',
    ycol='Y',
    vcol='Perm',
    tmin=0,
    tmax=999,
    nx=100,
    xmn=5,
    xsiz=10,
    ny=100,
    ymn=5,
    ysiz=10,
    nxdis=1,
    nydis=1,
    ndmin=0,
    ndmax=10,
    radius=100,
    ktype=0,
    skmean=training['Perm'].mean(),
    vario=vario
)

cell_size = 10
xrange = list(np.linspace(0, 1000, int(np.ceil((1000 - 0) / cell_size))))
yrange = list(np.linspace(0, 1000, int(np.ceil((1000 - 0) / cell_size))))

# %% Plot the 2D dataset and the kriging variance: Figures 2 and 6
figure = sfs.spatial_config_and_kvar(training, real_world, vmap, xrange, yrange)

# %% Plot the spatial configuration of the three cross-validations methods: Figures 3 and 7
sfs.plot_3_realizations(
    fair_train=sfs_train,
    fair_test=sfs_test,
    rand_train=vsa_train,
    rand_test=vsa_test,
    spatial_cv=spatial_cv,
    real_world_set=real_world,
    realiz=0
)

# %%
# Instantiate the object to obtain the KDE and violin plots
images = sfs.PublicationImages(
    test_kvar_random=vsa_kvar,
    test_kvar_fair=sfs_kvar,
    test_kvar_spatial=spatial_cv_kvar,
    rw_kvar=fair_cv.rw_krig_var
)

# Plot the kernel density estimates of the three cross-validations methods: Figures 4 and 8
plot = images.kde_plots(5.0)

# Plot the violin plots of the divergence metrics: Figures 5 and 9
plot2 = images.divergence_violins()
