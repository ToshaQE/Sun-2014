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

#%% Test in case of presence of the causality
lags = [50, 150]
data_train = data[:7000, :]
data_test = data[7000:, :]

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

granger_df_train = granger_df.iloc[:2000,:].to_numpy()
granger_df_test = granger_df.iloc[2000:,:].to_numpy()

results_ARIMA = nlc.nonlincausalityARIMA(x=granger_df_train, maxlag=[1,2], x_test=granger_df_test)

# results_ARIMA = nlc.nonlincausalityARIMA(x=data_train, maxlag=[1,2], x_test=data_test)


print("Stop")


