import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# %%
with open("fair_train.txt", "rb") as fp:  # Unpickling
    fair_train = pickle.load(fp)

with open("fair_test.txt", "rb") as fp:  # Unpickling
    fair_test = pickle.load(fp)

with open("fair_test.txt", "rb") as fp:  # Unpickling
    fair_test = pickle.load(fp)


# %% Machine learning

def ml_application(train_set, test_set, real_world):
    rmse = np.zeros((len(train_set), 2))
    rmse_rw = np.zeros((len(train_set), 2))

    for set_i in range(len(train_set)):
        train = train_set[set_i]
        test = test_set[set_i]
        features = ['Porosity_Percentage', 'Shc_Percentage', 'Gross_Thickness_m']
        target = 'TOC_weight_pct'

        X_train = train[features]
        y_train = train[target]
        X_test = test[features]
        y_test = test[target]

        reg = LinearRegression().fit(X_train, y_train)
        rf = RandomForestRegressor(
            max_depth=5, random_state=set_i, n_estimators=200,
            min_samples_leaf=6, max_leaf_nodes=10
        )
        rf.fit(X_train, y_train)

        predictions_reg = reg.predict(X_test)
        predictions_rf = rf.predict(X_test)

        # real world
        prd_reg_rw = reg.predict(real_world[features])
        prd_rf_rw = rf.predict(real_world[features])

        rmse[set_i, 0] = np.sqrt(mean_squared_error(y_test, predictions_reg))
        rmse[set_i, 1] = np.sqrt(mean_squared_error(y_test, predictions_rf))

        rmse_rw[set_i, 0] = np.sqrt(mean_squared_error(real_world[target], prd_reg_rw))
        rmse_rw[set_i, 1] = np.sqrt(mean_squared_error(real_world[target], prd_rf_rw))

    return rmse, rmse_rw


def statistics(test_results, rw_results, titulo):
    print("Mean " + titulo)
    print("Linear regression " + str(np.mean(test_results, axis=0)))
    print("Random forest " + str(np.mean(rw_results, axis=0)))
    print("\nStandard deviation: " + titulo)
    print("Linear regression " + str(np.std(test_results, axis=0)))
    print("Random forest " + str(np.std(rw_results, axis=0)) + "\n")


def ml_cv(spatial_set):
    cv_rmse = np.zeros((len(spatial_set), 2))
    features = ['Porosity_Percentage', 'Shc_Percentage', 'Gross_Thickness_m']
    target = 'TOC_weight_pct'
    cv_results = np.zeros((5, 2))
    for set_i in range(len(spatial_set)):
        dataset = spatial_set[set_i]
        for kfold_i in range(5):
            train = dataset.query("kfold != @kfold_i")
            test = dataset.query("kfold == @kfold_i")

            X_train = train[features]
            y_train = train[target]
            X_test = test[features]
            y_test = test[target]

            print(str(set_i) + " " + str(len(X_train)) + " " + str(len(X_test)))

            reg = LinearRegression().fit(X_train, y_train)
            rf = RandomForestRegressor(
                max_depth=5, random_state=set_i, n_estimators=200,
                min_samples_leaf=6, max_leaf_nodes=10
            )
            rf.fit(X_train, y_train)

            predictions_reg = reg.predict(X_test)
            predictions_rf = rf.predict(X_test)

            cv_results[kfold_i, 0] = np.sqrt(mean_squared_error(y_test[target], predictions_reg))
            cv_results[kfold_i, 1] = np.sqrt(mean_squared_error(y_test[target], predictions_rf))

        cv_rmse[set_i, ...] = np.mean(cv_results, axis=0)
        cv_results = np.zeros((5, 2))

    return cv_rmse


# %%
fair_rmse, fair_rmse_rw = ml_application(fair_train, fair_test, real_world)
# %%
vsa_rmse, vsa_rmse_rw = ml_application(vsa_train, vsa_test, real_world)

# %%
statistics(fair_rmse, fair_rmse_rw, 'Fair split')
statistics(vsa_rmse, vsa_rmse_rw, 'Validation set approach')

# %%
cv_rmse = ml_cv(spatial_cv)
