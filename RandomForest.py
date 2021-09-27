import os
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from ax.service.ax_client import AxClient


# from ax.utils.notebook.plotting import render, init_notebook_plotting

# %% Functions
def test_errors(train_list, test_list, best_parameters):
    rmse_test = np.zeros(len(test_list))
    for i in range(len(train_list)):
        traini = train_list[i]
        testi = test_list[i]
        X_tr = traini[features].to_numpy()
        y_tr = traini[target_].to_numpy()
        X_te = testi[features].to_numpy()
        y_te = testi[target_].to_numpy()

        modelo2 = RandomForestRegressor(**best_parameters)
        modelo2.fit(X_tr, y_tr)
        predictions = modelo2.predict(X_te)
        rmse_fair = np.sqrt(mean_squared_error(y_te, predictions))

        rmse_test[i] = rmse_fair

    return rmse_test


def list_reader(name):
    root = "D:\\PyCharm projects\\FairTrainTest\\Files\\Datasets\\Ignore/"
    with open(os.path.join(root, name + ".txt"), "rb") as fp:  # Unpickling
        file = pickle.load(fp)

    return file


class BayesianOptimization:
    def __init__(self, train_set, test, parameters_dict, predictors, target, parameter_names):
        self.parameter_names = parameter_names
        self.parameters_dict = parameters_dict
        self.X_train = train_set[predictors].to_numpy()
        self.y_train = train_set[target].to_numpy()
        self.X_test = test[predictors].to_numpy()
        self.y_test = test[target].to_numpy()

    def _model_loss(self, parameter):
        params = {}
        for p in self.parameter_names:
            params[p] = parameter.get(p)
        modelo = RandomForestRegressor(random_state=42, **params)
        modelo.fit(self.X_train, self.y_train)
        predictions = modelo.predict(self.X_test)
        rmse = np.sqrt(mean_squared_error(self.y_test, predictions))

        return rmse

    def _evaluate(self, parameters):
        return {"objective": self._model_loss(parameters)}

    def hyperparam_tuning(self, trials):
        ax_client = AxClient(verbose_logging=False)
        # create the experiment
        ax_client.create_experiment(
            parameters=self.parameters_dict,
            objective_name='objective',
            minimize=True,
        )

        with tqdm(total=trials) as pbar:
            for i in range(trials):
                parameters, trial_index = ax_client.get_next_trial()
                # Local evaluation here can be replaced with deployment to external system.
                ax_client.complete_trial(trial_index=trial_index, raw_data=self._evaluate(parameters))
                pbar.update(1)  # progress bar

        dframe = ax_client.get_trials_data_frame().sort_values('trial_index')
        dframe.sort_values('objective', ascending=True, inplace=True)
        dframe.reset_index(inplace=True, drop=True)
        best_parameters, _ = ax_client.get_best_parameters()

        return dframe, best_parameters


# %% Read data
fair_train = list_reader("fair_train")
fair_test = list_reader("fair_test")
vsa_train = list_reader("vsa_train")
vsa_test = list_reader("vsa_test")
# %% define parameters
parameters_dictionary = [
    {
        "name": "n_estimators",
        "type": "range",
        "bounds": [2, 10],
        "value_type": "int",
    },
    {
        "name": "min_samples_leaf",
        "type": "range",
        "bounds": [2, 5],
        "value_type": "int",
    },
    {
        "name": "max_leaf_nodes",
        "type": "range",
        "bounds": [20, 26],
        "value_type": "int",
    },
    {
        "name": "max_depth",
        "type": "range",
        "bounds": [32, 41],
        "value_type": "int",
    },
]
# parameter_constraints=["x0 + x2 >= 5.0", "x1 + x3 >= 5.0", "x0 + x2 <= 5.0", "x1 + x3 <= 5.0", "x2 + x3 <= 4.0"]
# https://github.com/facebook/Ax/issues/270


features = ['Porosity_Percentage', 'Shc_Percentage', 'Gross_Thickness_m']
target_ = 'TOC_weight_pct'
all_params = ["n_estimators", "min_samples_leaf", "max_leaf_nodes", "max_depth"]

# %% instantiate class for fair model
# df_train = fair_train[0]
# df_test = fair_test[0]
# modelo = BayesianOptimization(df_train, df_test, parameters_dictionary, features, target_, all_params)
# result, _ = modelo.hyperparam_tuning(30)
# %% instantiate class for vsa
# df_train = vsa_train[0]
# df_test = vsa_test[0]
# modelo = BayesianOptimization(df_train, df_test, parameters_dictionary, features, target_, all_params)
# result2, _ = modelo.hyperparam_tuning(30)

# %% read the real world and whole data
train = pd.read_csv("D:\\PyCharm projects\\FairTrainTest\\Files\\Datasets\\Ignore/available_data.csv")
rw = pd.read_csv("D:\\PyCharm projects\\FairTrainTest\\Files\\Datasets\\Ignore/real_world.csv")

X_train_ = train[features].to_numpy()
y_train_ = train[target_].to_numpy()
X_test_ = rw[features].to_numpy()
y_test_ = rw[target_].to_numpy()
# %% tuned fair model
best_params = {
    "n_estimators": 744,
    "min_samples_leaf": 9,
    "max_leaf_nodes": 35,
    "max_depth": 29,
    "random_state": 42
}

rmse_fair_test = test_errors(fair_test, fair_test, best_params)

model = RandomForestRegressor(**best_params)
model.fit(X_train_, y_train_)
predict = model.predict(X_test_)
rmse_fair_rw = np.sqrt(mean_squared_error(y_test_, predict))

# %% tune validation set approach
best_params = {
    "n_estimators": 2,
    "min_samples_leaf": 2,
    "max_leaf_nodes": 24,
    "max_depth": 35,
    "random_state": 42
}
rmse_vsa_test = test_errors(vsa_test, vsa_test, best_params)
model = RandomForestRegressor(**best_params)
model.fit(X_train_, y_train_)
predict = model.predict(X_test_)
rmse_vsa_rw = np.sqrt(mean_squared_error(y_test_, predict))

# %%
print("Mean RMSE 100 test sets: ")
print(f"Fair approach: {np.mean(rmse_fair_test):.2f} \u00B1 {np.std(rmse_fair_test):.2f}")
print(f"VSA approach : {np.mean(rmse_vsa_test):.2f} \u00B1 {np.std(rmse_vsa_test):.2f}\n")
print("RMSE in real world")
print(f"Fair approach: {rmse_fair_rw:.2f}")
print(f"VSA approach : {rmse_vsa_rw:.2f}\n")

# %%
