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
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.eval_measures import meanabs



from FRUFS import FRUFS
import matplotlib.pyplot as plt
import optuna
import joblib, gc
# import lightgbm as lgb
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

import pmdarima as pmd
from pmdarima.arima import auto_arima
import math
from my_metrics import rae, rrse

# Unpickling the data
infile = open("Sun_Model_Data",'rb')
Model_Data = pickle.load(infile)
infile.close()

#Reading in the data
df_aapl = pd.read_csv("df_aaple.csv")
# Truncating the dataw
aapl_medium = df_aapl.iloc[:2000,:16]
aapl_short = df_aapl.iloc[:2000,:16]


df_air_q = pd.read_csv("AirQualityUCI.csv")
# # endogs = df_air_q.iloc[:,1]
# # exogs = df_air_q.iloc[:,2:]
def SARIMA(df, target, test_size):

    target_df = df.loc[:, target]

    y_train, y_test = train_test_split(target_df, test_size=test_size, shuffle=False)

    arima_fit = auto_arima(y=y_train, information_criterion="bic", random_state=42)
    arima_pred_in = arima_fit.predict_in_sample(n_periods=y_train.shape[0])
    arima_pred_out = arima_fit.predict(n_periods=y_test.shape[0])

    MAE_train = meanabs(y_train, arima_pred_in)
    MSE_train = mean_squared_error(arima_pred_in, y_train)
    RMSE_train = np.sqrt(MSE_train)
    RAE_train = rae(actual=y_train, predicted = arima_pred_in)
    RRSE_train = rrse(actual=y_train, predicted = arima_pred_in)

    MAE_test = meanabs(y_test, arima_pred_out)
    MSE_test = mean_squared_error(arima_pred_out, y_test)
    RMSE_test = math.sqrt(MSE_test)
    RAE_test = rae(actual=y_test, predicted = arima_pred_out)
    RRSE_test = rrse(actual=y_test, predicted = arima_pred_out)

    my_metrics_test = {"MAE":[MAE_test], "RMSE":[RMSE_test], "RAE":[RAE_test], "RRSE":[RRSE_test]}
    my_metrics_train = {"MAE":[MAE_train], "RMSE":[RMSE_train], "RAE":[RAE_train], "RRSE":[RRSE_train]}
    my_metrics = {"train":my_metrics_train, "test":my_metrics_test}

    return arima_fit, arima_pred_in, arima_pred_out, MAE_train, MAE_test, my_metrics

def SARIMAX(df, target, test_size):

    feature_df = df.loc[:, ~df.columns.isin([target, "Date"])]
    target_df = df.loc[:, target]

    feature_df = feature_df.shift(1).dropna()
    feature_df.reset_index(drop=True, inplace=True)

    target_df = target_df.iloc[1:]
    target_df.reset_index(drop=True, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(feature_df, target_df, test_size=test_size, shuffle=False)

    arima_fit = auto_arima(y=y_train, X = X_train, information_criterion="bic", random_state=42)
    arima_pred_in = arima_fit.predict_in_sample(X=X_train, n_periods=X_train.shape[0])
    arima_pred_out = arima_fit.predict(X=X_test, n_periods=X_test.shape[0])

    MAE_train = meanabs(y_train, arima_pred_in)
    MSE_train = mean_squared_error(arima_pred_in, y_train)
    RMSE_train = np.sqrt(MSE_train)
    RAE_train = rae(actual=y_train, predicted = arima_pred_in)
    RRSE_train = rrse(actual=y_train, predicted = arima_pred_in)

    MAE_test = meanabs(y_test, arima_pred_out)
    MSE_test = mean_squared_error(arima_pred_out, y_test)
    RMSE_test = math.sqrt(MSE_test)
    RAE_test = rae(actual=y_test, predicted = arima_pred_out)
    RRSE_test = rrse(actual=y_test, predicted = arima_pred_out)

    my_metrics_test = {"MAE":[MAE_test], "RMSE":[RMSE_test], "RAE":[RAE_test], "RRSE":[RRSE_test]}
    my_metrics_train = {"MAE":[MAE_train], "RMSE":[RMSE_train], "RAE":[RAE_train], "RRSE":[RRSE_train]}
    my_metrics = {"train":my_metrics_train, "test":my_metrics_test}

    return arima_fit, arima_pred_in, arima_pred_out, MAE_train, MAE_test, my_metrics

Auto_SARIMA, pred_in, pred_out, MAE_train, MAE_test, sarima_metrics = SARIMA(df=df_air_q, target="CO(GT)", test_size=0.2) 

Auto_SARIMAX, pred_in, pred_out, MAE_train, MAE_test, sarimax_metrics = SARIMAX(df=df_air_q, target="CO(GT)", test_size=0.2)


sun_metrics = Model_Data.my_metrics

all_metrics = dict(sun_metrics)

train_test = all_metrics.keys()

for split in train_test:
    metrics = all_metrics[split].keys()
    for metric in metrics:
        all_metrics[split][metric].append(sarima_metrics[split][metric][0])
        all_metrics[split][metric].append(sarimax_metrics[split][metric][0])


print(Auto_SARIMAX.summary(), "\n\n", MAE_train,"\n", MAE_test)

all_metrics_df = pd.DataFrame.from_dict(all_metrics["test"])
all_metrics_df.to_csv("all_metrics.csv", index=False)


print("Stop")

