# %%
# %load_ext autoreload

# %%
# %autoreload 2

# %%
# %load_ext line_profiler

# %%
import os
import numpy as np
import pandas as pd
from geostatspy.GSLIB import make_variogram
from geostatspy.geostats import kb2d
import matplotlib.pyplot as plt
from KDEpy import FFTKDE

import utils_demo as sfs

# %%
SET = 1  # 1 or 2
XDIR = 'X'  # the name of the column that contains the X direction
YDIR = 'Y'  # the name of the column that contains the Y direction

# %%
path = os.path.join(os.getcwd(), "Files", "Datasets", "demo" + str(SET) + "_train.csv")
training = pd.read_csv(path, dtype={'X': float, 'Y': float})
training.reset_index(inplace=True)  # use the well index as uwi
training = training.rename(columns={'index': 'UWI'})

path = os.path.join(os.getcwd(), "Files", "Datasets", "demo" + str(SET) + "_rw.csv")
real_world = pd.read_csv(path, dtype={'X': float, 'Y': float})
real_world.reset_index(inplace=True)  # use the well index as uwi
real_world = real_world.rename(columns={'index': 'UWI'})

# %%
training.shape, real_world.shape

# %% [markdown]
# ## Create covariance matrices

# %%
from scipy.spatial.distance import cdist

training_vectors = training[["X", "Y"]].to_numpy()
real_world_vectors = real_world[["X", "Y"]].to_numpy()

dist_training = cdist(training_vectors, training_vectors, metric='euclidean')
dist_train_rw = cdist(training_vectors, real_world_vectors, metric='euclidean')

def covariance_function(distance):    
    return np.exp(-(distance/400))

cov_matrix = covariance_function(dist_training)
cov_realworld_matrix = covariance_function(dist_train_rw)

# %%
pd.Series(cov_matrix.ravel()).describe()

# %% [markdown]
# ## Create a train test split

# %%
from spatial_split import SpatialSplit, sample_to_match, simple_kriging_variances_loo

splitter = SpatialSplit(cov_matrix, cov_realworld_matrix)
splitter.draw_test_indices(test_size=0.1)

# %% [markdown]
# ## Test the speed

# %%
cov_matrix.shape, cov_realworld_matrix.shape

# %%
# %timeit splitter.draw_test_indices(test_size=0.05)

# %% [markdown]
# ## Create a plot

# %%
# %%time

from spatial_split import simple_kriging_variances

plt.title("Draws of joint variance over test set, conditioned on training set")

vars_realworld = simple_kriging_variances(cov_matrix, cov_realworld_matrix)

bw = "silverman"
x, y = FFTKDE(bw=bw).fit(vars_realworld).evaluate()
plt.plot(x, y, zorder=99, lw=5, label="RW variance, conditioned on train + test set")

splitter = SpatialSplit(cov_matrix, cov_realworld_matrix)
samples = len(training)
rw_samples = len(real_world)

for test in range(50):
    print(test, end=" ")
    
    test_indices = splitter.draw_test_indices(test_size=rw_samples)
    train_indices = list(sorted(set(range(samples)) - set(test_indices)))
    
    test_indices = np.array(test_indices, dtype=int)
    train_indices = np.array(train_indices, dtype=int)
    
    test_cov = cov_matrix[train_indices, :][:, test_indices]
    train_cov = cov_matrix[train_indices, :][:, train_indices]
    
    # Variances of test point, conditioned on training points
    vars_test = simple_kriging_variances(train_cov, test_cov)

    x, y = FFTKDE(bw=bw).fit(vars_test).evaluate()
    plt.plot(x, y, color="k", alpha=0.1, zorder=9)


plt.grid(True, zorder=0)
plt.legend();

# %%
# %%time

from spatial_split import simple_kriging_variances

plt.title("Draws of test set variances, conditioned on remaining training data in each loop")

vars_realworld = simple_kriging_variances(cov_matrix, cov_realworld_matrix)

x, y = FFTKDE(bw=bw).fit(vars_realworld).evaluate()
plt.plot(x, y, zorder=99, lw=5, label="RW variance, conditioned on train + test set")

splitter = SpatialSplit(cov_matrix, cov_realworld_matrix)
samples = len(training)
rw_samples = len(real_world)

for test in range(50):
    print(test, end=" ")
    
    # Variance of test points as the algorithms adds them
    _, vars_test = splitter.draw_test_indices(test_size=rw_samples, return_variance=True)

    x, y = FFTKDE(bw=bw).fit(vars_test).evaluate()
    plt.plot(x, y, color="k", alpha=0.1, zorder=9)


plt.grid(True, zorder=0)
plt.legend();

# %% [markdown]
# ## Non spatial splitting

# %%
# %%time

from spatial_split import simple_kriging_variances
from sklearn.model_selection import train_test_split

plt.title("Draws of joint variance over test set, conditioned on training set")

vars_realworld = simple_kriging_variances(cov_matrix, cov_realworld_matrix)

bw = "silverman"
x, y = FFTKDE(bw=bw).fit(vars_realworld).evaluate()
plt.plot(x, y, zorder=99, lw=5, label="RW variance, conditioned on train + test set")

for test in range(50):
    print(test, end=" ")
    
    train_indices, test_indices = train_test_split(np.arange(len(training)))
    
    test_indices = np.array(test_indices, dtype=int)
    train_indices = np.array(train_indices, dtype=int)
    
    test_cov = cov_matrix[train_indices, :][:, test_indices]
    train_cov = cov_matrix[train_indices, :][:, train_indices]
    
    # Variances of test point, conditioned on training points
    vars_test = simple_kriging_variances(train_cov, test_cov)

    x, y = FFTKDE(bw=bw).fit(vars_test).evaluate()
    plt.plot(x, y, color="k", alpha=0.1, zorder=9)


plt.grid(True, zorder=0)
plt.legend();

# %% [markdown]
# ## Plot the sampling matching algorithm

# %%
from spatial_split import sample_to_match

target_data = (np.random.beta(3, 2, 1000)).tolist()
samples_data = (np.random.beta(1.2, 1.3, 1000))


indices = np.arange(len(samples_data))
def function(indices):
    return samples_data[indices]

num_samples = 500

chosen_indices = list(sample_to_match(indices, function, target_data, num_samples))

chosen_data = [samples_data[i] for (i, v) in chosen_indices]


plt.hist(target_data, bins="fd", label="Target", alpha=0.33, density=False)
plt.hist(samples_data, bins="fd", label="Samples", alpha=0.33, density=False)
plt.hist(chosen_data, bins="fd", label="Chosen", alpha=0.33, density=False)

plt.legend()
plt.show()
