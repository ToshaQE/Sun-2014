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


df_dehli = pd.read_csv("dehli_weather.csv")

df_air_q = pd.read_csv("AirQualityUCI.csv")


arima_fit = auto_arima(y=aapl_long.iloc[:,1], information_criterion="bic", random_state=42)

y_pred = arima_fit.predict()
arima_fit.summary()