import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.ar_model import AutoReg
import pickle

df_aapl = pd.read_csv("df_aaple.csv")
df_small = df_aapl.iloc[:,:4]
df_small.drop(columns="Adj. Close", inplace=True)
df_small["P/E"] = df_small["P/E (LTM)"]
df_small.drop(columns="P/E (LTM)", inplace=True)
df_small["# Buys"] = df_aapl["# Buys"]

df_small_raw = df_small

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.ar_model import AutoReg
import pickle

df_aapl = pd.read_csv("df_aaple.csv")
df_small = df_aapl.iloc[:,:4]
df_small.drop(columns="Adj. Close", inplace=True)
df_small["P/E"] = df_small["P/E (LTM)"]
df_small.drop(columns="P/E (LTM)", inplace=True)
df_small["# Buys"] = df_aapl["# Buys"]

df_small_raw = df_small

def algo(df, max_lag):

    # Step 1: Tranformation for stationarity
    features = list(df_small.columns)[1:]

    for feature in features:
        result = adfuller(df_small[feature], autolag=None)
        counter = 0
        while result[1] > 0.05:
            df_small[feature] = df_small[feature] - df_small[feature].shift(1)
            #df_small.dropna()
            counter += 1
            #dropna(inplace=False) because it drops one observation for each feature
            result = adfuller(df_small.dropna()[feature], autolag=None)
        print(f'Order of integration for feature "{feature}" is {counter}')
    df_small.dropna(inplace=True)
    df_small.reset_index(drop=True, inplace=True)

    # Step 2: Building a univariate model and finding the optimal l
    BICs = []
    for i in list(range(max_lag)):
        model = AutoReg(df_small.iloc[:,1], lags=i).fit()
        BICs.append(model.bic)

    min_bic_ind = BICs.index(min(BICs))

    # model = AutoReg(df_small.iloc[:,1], lags=min_bic_ind).fit()
    # model.summary()


    # Step 2: Bulding augmented model and finding the optimal w for each Xi
    
    # Defining dictionary to store all augmented models
    aug_models = {}
    feature_n_dfs = {}

    
    for n in list(range(1, len(features))):
        columns = []
        for i in list(range(1, max_lag)):
            columns.append(features[n]+".L"+str(i))

        feature_n_df = pd.DataFrame(columns=columns)
        for i in list(range(max_lag - 1)):
            feature_n_df[columns[i]] = df_small[features[n]].shift(i)

        feature_n_df.fillna(1, inplace=True)

        BICs = []
        for i in list(range(max_lag - 1)):
            model = AutoReg(df_small.iloc[:,1], lags=min_bic_ind, exog=feature_n_df.iloc[:,:i+1]).fit()
            BICs.append(model.bic)

        min_bic_ind_aug = BICs.index(min(BICs))

        model = AutoReg(df_small.iloc[:,1], lags=min_bic_ind, exog=feature_n_df.iloc[:,:min_bic_ind_aug+1]).fit()

        aug_models[features[n]] = model
        feature_n_dfs[features[n]] = feature_n_df
        #model.summary()
    return aug_models, feature_n_dfs

