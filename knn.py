# %% import
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# %% functions
def list_reader(name):
    root = "D:\\PyCharm projects\\FairTrainTest\\Files\\Datasets\\Ignore/"
    with open(os.path.join(root, name + ".txt"), "rb") as fp:  # Unpickling
        file = pickle.load(fp)

    return file


def test_errors(train_list, test_list, knn):
    rmse_test = np.zeros(len(test_list))
    for index in range(len(train_list)):
        traini = train_list[index]
        testi = test_list[index]
        scaler = StandardScaler().fit(traini[features].to_numpy())
        traini[features] = scaler.transform(traini[features].to_numpy())
        testi[features] = scaler.transform(testi[features].to_numpy())

        X_tr = traini[features].to_numpy()
        y_tr = traini[target_].to_numpy()
        X_te = testi[features].to_numpy()
        y_te = testi[target_].to_numpy()

        modelo2 = KNeighborsRegressor(n_neighbors=knn)
        modelo2.fit(X_tr, y_tr)
        prediction2 = modelo2.predict(X_te)
        RMSE = np.sqrt(mean_squared_error(y_te, prediction2))

        rmse_test[i] = RMSE

    return rmse_test


# %%
fair_train = list_reader("fair_train")
fair_test = list_reader("fair_test")
vsa_train = list_reader("vsa_train")
vsa_test = list_reader("vsa_test")
features = ['Porosity_Percentage', 'Shc_Percentage', 'Gross_Thickness_m']
target_ = 'TOC_weight_pct'
train = pd.read_csv("D:\\PyCharm projects\\FairTrainTest\\Files\\Datasets\\Ignore/available_data.csv")
rw = pd.read_csv("D:\\PyCharm projects\\FairTrainTest\\Files\\Datasets\\Ignore/real_world.csv")
# %%
rmse = np.zeros(20)
for i in range(1, 21):
    rmse[i - 1] = np.mean(test_errors(fair_train, fair_test, knn=i))
knn_fair = np.argmin(rmse) + 1
# %%
rmse = np.zeros(20)
for i in range(1, 21):
    rmse[i - 1] = np.mean(test_errors(vsa_train, vsa_test, knn=i))
knn_vsa = np.argmin(rmse) + 1
# %%
scaler2 = StandardScaler().fit(train[features].to_numpy())
train[features] = scaler2.transform(train[features].to_numpy())
rw[features] = scaler2.transform(rw[features].to_numpy())
X_tra = train[features].to_numpy()
y_tra = train[target_].to_numpy()
X_tes = rw[features].to_numpy()
y_tes = rw[target_].to_numpy()
# %%
modelo = KNeighborsRegressor(n_neighbors=2)
modelo.fit(X_tra, y_tra)
prediction = modelo.predict(X_tes)
RMSE_fair = np.sqrt(mean_squared_error(y_tes, prediction))

modelo = KNeighborsRegressor(n_neighbors=8)
modelo.fit(X_tra, y_tra)
prediction = modelo.predict(X_tes)
RMSE_vsa = np.sqrt(mean_squared_error(y_tes, prediction))
