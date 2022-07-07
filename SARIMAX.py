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




#Reading in the data



df_aapl = pd.read_csv("df_aaple.csv")
# Truncating the dataw
aapl_medium = df_aapl.iloc[:2500,:16]
endogs = aapl_medium["Close"].iloc[1:2000]
endogs.reset_index(drop=True, inplace=True)

exogs = aapl_medium.iloc[:2000,2:].shift(1).dropna()
exogs.reset_index(drop=True, inplace=True)

exogs_test = aapl_medium.iloc[2000:,2:].shift(1).dropna()
exogs_test.reset_index(drop=True, inplace=True)
endogs_test = aapl_medium["Close"].iloc[2001:]
endogs_test.reset_index(drop=True, inplace=True)

# aapl_long = df_aapl.iloc[:,:16]
# exogs = aapl_long.iloc[:,2:].shift(1).dropna()
# exogs.reset_index(drop=True, inplace=True)

# endogs = aapl_long["Close"].iloc[1:]
# endogs.reset_index(drop=True, inplace=True)



# df_air_q = pd.read_csv("AirQualityUCI.csv")
# endogs = df_air_q.iloc[:,1]
# exogs = df_air_q.iloc[:,2:]


# sarimax = SARIMAX(endog = endogs, exog=exogs).fit()

# pred_in = sarimax.predict()

# MAE_train = meanabs(endogs, pred_in) 

# print(sarimax.summary(), MAE_train)



arima_fit = auto_arima(y=endogs, X = exogs, information_criterion="bic", random_state=42)
arima_pred = arima_fit.predict_in_sample(X=exogs, n_periods=exogs.shape[0])


# arima_fit = auto_arima(y=endogs, information_criterion="bic", random_state=42)
# arima_pred = arima_fit.predict_in_sample()

MAE_arima = meanabs(endogs, arima_pred)


arima_pred_out = arima_fit.predict(X=exogs_test, n_periods=exogs_test.shape[0])
MAE_arima_test = meanabs(endogs_test, arima_pred_out)

print(arima_fit.summary(), MAE_arima)


print("Stop")

