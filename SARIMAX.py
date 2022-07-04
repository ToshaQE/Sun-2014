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
aapl_medium = df_aapl.iloc[:1600,:16]
aapl_long = df_aapl.iloc[:,:16]


exogs = aapl_medium.iloc[:,2:].shift(1).dropna()
endogs = aapl_medium["Close"].iloc[1:]

sarimax = SARIMAX(endog = endogs, exog=exogs).fit()

pred_in = sarimax.predict()

MAE_train = meanabs(endogs, pred_in) 

print(sarimax.summary(), MAE_train)

# arima_fit = auto_arima(y=endogs, X = exogs, information_criterion="bic", random_state=42)

# arima_pred = arima_fit.predict(X=exogs, n_periods=exogs.shape[0])

# MAE_arima = meanabs(endogs, arima_pred)


print("Stop")

