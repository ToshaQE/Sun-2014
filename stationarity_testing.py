from pmdarima.arima import auto_arima

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

from arch.unitroot import *


#Reading in the data
df_aapl = pd.read_csv("df_aaple.csv")
# Truncating the dataw
aapl_medium = df_aapl.iloc[:2000,:16]
aapl_long = df_aapl.iloc[:,:16]

df_jpm = pd.read_csv("jpm.csv")
jpm_medium = df_jpm.iloc[:2000,:16]
jpm_long = df_jpm.iloc[:,:16]

df_fb = pd.read_csv("fb.csv")
fb_medium = df_fb.iloc[:2000,:16]
fb_long = df_fb.iloc[:,:16]

df_google = pd.read_csv("googl.csv")
google_medium = df_google.iloc[:2000,:16]
google_long = df_google.iloc[:,:16]


y = aapl_medium.iloc[:2000,1]
# y = aapl_long.iloc[:3200,1]
# y=y.diff().dropna()

# y = jpm_medium.iloc[:,1]
# y = jpm_long.iloc[:,1]


# ADF_n = ADF(y, trend = 'n')
# ADF_c = ADF(y, trend = 'c')
# ADF_ct = ADF(y, trend = 'ct')

# ADF_list = [ADF_n, ADF_c, ADF_ct]

# for i in ADF_list:
#     print(i.pvalue)


# ADF_n = ADF(y_long.diff().dropna(), trend = 'n')
# ADF_c = ADF(y_long.diff().dropna(), trend = 'c')
# ADF_ct = ADF(y_long.diff().dropna(), trend = 'ct')

# ADF_list = [ADF_n, ADF_c, ADF_ct]

# for i in ADF_list:
#     print(i.pvalue)


tests = [ADF, KPSS, PhillipsPerron]
tests_str = ["ADF", "KPSS", "PhillipsPerron"]

results = {}
counter = 0
for t in tests:
    type = {}
    if t != KPSS:
        type["n"] = t(y, trend = 'n').pvalue
        type["c"] = t(y, trend = 'c').pvalue
        type["ct"] = t(y, trend = 'ct').pvalue
    else:
        type["c"] = t(y, trend = 'c').pvalue
        type["ct"] = t(y, trend = 'ct').pvalue
    print(tests_str[counter] + f" results are as follows: {type}\n")
    results[tests_str[counter]] = type
    counter += 1

result = adfuller(y, autolag="t-stat", regression="c")[1]


print("Stop")