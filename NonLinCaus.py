from cgi import test
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import grangercausalitytests
import statsmodels.api as sm
import pickle
from sklearn.model_selection import train_test_split
import re
from arch.unitroot.cointegration import engle_granger


from FRUFS import FRUFS
import matplotlib.pyplot as plt
import optuna
import joblib, gc
import lightgbm as lgb
import seaborn as sns

from sklearn.datasets import make_regression
from scipy.stats import pearsonr
from tqdm.notebook import trange, tqdm
from FRUFS import FRUFS
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import grangercausalitytests
import pickle
from sklearn.model_selection import train_test_split
import re
import logging
import plotly.express as px


from Sun_Model_Class import Sun_Model
# logging.INFO

import nonlincausality as nlc

from nonlincausality.utils import prepare_data_for_prediction, calculate_pred_and_errors




granger_df = pd.read_csv("gr_test_df.csv")
granger_df = granger_df.iloc[:,1:]



#%% Data generation Y->X
np.random.seed(10)
y = (
    np.cos(np.linspace(0, 20, 10_100))
    + np.sin(np.linspace(0, 3, 10_100))
    - 0.2 * np.random.random(10_100)
)
np.random.seed(20)
x = 2 * y ** 3 - 5 * y ** 2 + 0.3 * y + 2 - 0.05 * np.random.random(10_100)
data = np.vstack([x[:-100], y[100:]]).T

plt.figure()
plt.plot(data[:, 0], label="X")
plt.plot(data[:, 1], label="Y")
plt.xlabel("Number of sample")
plt.ylabel("Signals [AU]")
plt.legend()

# Test in case of presence of the causality
lags = [50, 150]
# lags = [1,2]
data_train = data[:7000, :]
data_test = data[7000:, :]

results = nlc.nonlincausalityMLP(
    x=data_train,
    maxlag=lags,
    Dense_layers=2,
    Dense_neurons=[100, 100],
    x_test=data_test,
    run=1,
    add_Dropout=True,
    Dropout_rate=0.01,
    epochs_num=[50, 100],
    learning_rate=[0.001, 0.0001],
    batch_size_num=128,
    verbose=True,
    plot=True,
)

# results = nlc.nonlincausalityARIMA(x=data_train, maxlag=lags, x_test=data_test)

#%% Example of obtaining the results
for lag in lags:
    best_model_X = results[lag].best_model_X
    best_model_XY = results[lag].best_model_XY

    p_value = results[lag].p_value
    test_statistic = results[lag].test_statistic

    best_history_X = results[lag].best_history_X
    best_history_XY = results[lag].best_history_XY

    nlc.plot_history_loss(best_history_X, best_history_XY)
    plt.title("Lag = %d" % lag)
    best_errors_X = results[lag].best_errors_X
    best_errors_XY = results[lag].best_errors_XY

    cohens_d = np.abs(
        (np.mean(np.abs(best_errors_X)) - np.mean(np.abs(best_errors_XY)))
        / np.std([best_errors_X, best_errors_XY])
    )
    print("For lag = %d Cohen's d = %0.3f" % (lag, cohens_d))
    print(f"Test statistic = {test_statistic} p-value = {p_value}")

    # Using models for prediction
    data_X, data_XY = prepare_data_for_prediction(data_test, lag)
    X_pred_X = best_model_X.predict(data_X)
    X_pred_XY = best_model_XY.predict(data_XY)

    # Plot of true X vs X predicted
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(data_test[lag:, 0], X_pred_X, "o")
    ax[0].set_xlabel("X test values")
    ax[0].set_ylabel("Predicted X values")
    ax[0].set_title("Model based on X")
    ax[1].plot(data_test[lag:, 0], X_pred_XY, "o")
    ax[1].set_xlabel("X test values")
    ax[1].set_ylabel("Predicted X values")
    ax[1].set_title("Model based on X and Y")
    plt.suptitle("Lag = %d" % lag)

    # Another way of obtaining predicted values (and errors)
    X_pred_X, X_pred_XY, error_X, error_XY = calculate_pred_and_errors(
        data_test[lag:, 0], 
        data_X, 
        data_XY, 
        best_model_X, 
        best_model_XY
    )
    # Plot of X predicted vs prediction error
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(X_pred_X, error_X, "o")
    ax[0].set_xlabel("Predicted X values")
    ax[0].set_ylabel("Prediction errors")
    ax[0].set_title("Model based on X")
    ax[1].plot(X_pred_XY, error_XY, "o")
    ax[1].set_xlabel("Predicted X values")
    ax[1].set_ylabel("Prediction errors")
    ax[1].set_title("Model based on X and Y")
    plt.suptitle("Lag = %d" % lag)





# results = nlc.nonlincausalityMLP(
#     x=data_train,
#     maxlag=lags,
#     Dense_layers=2,
#     Dense_neurons=[100, 100],
#     x_test=data_test,
#     run=1,
#     add_Dropout=True,
#     Dropout_rate=0.01,
#     epochs_num=[50, 100],
#     learning_rate=[0.001, 0.0001],
#     batch_size_num=128,
#     verbose=True,
#     plot=True,
# )

# granger_df_train = granger_df.iloc[:2000,:].to_numpy()
# granger_df_test = granger_df.iloc[2000:,:].to_numpy()

# results_ARIMA = nlc.nonlincausalityARIMA(x=granger_df_train, maxlag=[1, 10], x_test=granger_df_test)

# results_ARIMA = nlc.nonlincausalityARIMA(x=data_train, maxlag=lags, x_test=data_test)


print("Stop")


