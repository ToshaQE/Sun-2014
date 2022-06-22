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
sun_x_train = Model_Data.train_x
sun_y_train = Model_Data.train_y
sun_y_test = Model_Data.test_y
sun_x_test = Model_Data.test_x
sun_MAE_train = Model_Data.MAE["train"]
sun_MAE_test = Model_Data.MAE["test"]
sun_model = Model_Data.fin_model

print("@")

sig_col_names = list(sun_model.params[np.where(sun_model.pvalues < 0.5)[0]].index)


print("@")