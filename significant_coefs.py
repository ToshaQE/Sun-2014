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

from sigfig import round

# Unpickling the data
infile = open("Sun_Model_Data",'rb')
Model_Data = pickle.load(infile)
infile.close()

# Reading in the data
feature_n_dfs_merge = Model_Data.train_x
y_train_m = Model_Data.train_y
sun_y_test = Model_Data.test_y
sun_x_test = Model_Data.test_x
sun_MAE_train = Model_Data.MAE["train"]
sun_MAE_test = Model_Data.MAE["test"]
fin_model = Model_Data.fin_model

print("@")

sig_col_names = list(fin_model.params[np.where(fin_model.pvalues < 0.05)[0]].index)[0]

# Using backward elimination to drop insignificant features
# Defining critiacl p-value determining whether a feture is to be dropped
critical_p_value = 0.05

# Finding p-value of the lesat siginificant feature
max_p_value = max(fin_model.pvalues)

while max_p_value >= critical_p_value:
    # Column name of the least significant feature
    least_sig_var = list(fin_model.params[np.where(fin_model.pvalues == max_p_value)[0]].index)[0]
    # If least_sig_var is the constant we run Autoreg without it
    if least_sig_var == "const":
        fin_model = AutoReg(y_train_m, lags=0, exog=feature_n_dfs_merge, trend="n").fit()
        # Defining const_dropped to know whether we run Autoreg with or w/o const
        const_dropped = True


    else:
        # Dropping the least_sig_var from the df
        feature_n_dfs_merge.pop(least_sig_var)
        # If const has been dropped, we run Autoreg w/o it
        if const_dropped:
            fin_model = AutoReg(y_train_m, lags=0, exog=feature_n_dfs_merge, trend="n").fit()
        else:
            fin_model = AutoReg(y_train_m, lags=0, exog=feature_n_dfs_merge).fit()

    # At the end of each iteration we find the new highest p-value        
    max_p_value = max(fin_model.pvalues)






print("@")