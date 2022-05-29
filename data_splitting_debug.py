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


df_aapl = pd.read_csv("df_aaple.csv")
df_small = df_aapl.iloc[:,:4]
df_small.drop(columns="Adj. Close", inplace=True)
df_small["P/E"] = df_small["P/E (LTM)"]
df_small.drop(columns="P/E (LTM)", inplace=True)
df_small["# Buys"] = df_aapl["# Buys"]

df_small_raw = df_small

def algo(df, target, max_lag):

    # Step 1: Tranformation for stationarity d
    # Here features are everything except for the date
    features = [n for n in list(df.columns) if n != "Date"]

    for feature in features:
        result = adfuller(df[feature], autolag=None)
        counter = 0
        while result[1] > 0.05:
            df[feature] = df[feature] - df[feature].shift(1)
            #df_small.dropna()
            counter += 1
            #dropna(inplace=False) because it drops one observation for each feature
            result = adfuller(df.dropna()[feature], autolag=None)
        print(f'Order of integration for feature "{feature}" is {counter}')
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    feature_df = df.loc[:, ~df.columns.isin([target, "Date"])]
    target_df = df.loc[:, target]

    X_train, X_test, y_train, y_test = train_test_split(feature_df, target_df, test_size=0.4, shuffle=False) 

    # Step 2: Building a univariate model and finding the optimal l
    BICs = []
    for i in list(range(max_lag)):
        model = AutoReg(y_train, lags=i).fit()
        BICs.append(model.bic)

    min_bic_ind = BICs.index(min(BICs))

    # model = AutoReg(df_small.iloc[:,1], lags=min_bic_ind).fit()
    # model.summary()


    # Step 2: Bulding augmented model and finding the optimal w for each Xi
    
    Xs = list(X_train.columns)

    # Defining dictionary to store all augmented models
    aug_models = {}
    feature_n_dfs = {}
    feature_n_dfs_merge = []
    
    for n in list(range(len(Xs))):
        columns = []
        for i in list(range(1, max_lag+1)):
            columns.append(Xs[n]+".L"+str(i))

        feature_n_df = pd.DataFrame(columns=columns)
        for i in list(range(max_lag)):
            feature_n_df[columns[i]] = X_train[Xs[n]].shift(i+1)

        feature_n_df.fillna(1, inplace=True)

        BICs = []
        #Why do I have max_lag-1 and then i+1?
        for i in list(range(max_lag-1)):
            model = AutoReg(y_train, lags=min_bic_ind, exog=feature_n_df.iloc[:,:i+1]).fit()
            BICs.append(model.bic)

        min_bic_ind_aug = BICs.index(min(BICs))
        #Full and Partial autocorrelation plot?
        feature_n_df1 = feature_n_df
        feature_n_df = feature_n_df.iloc[:,:min_bic_ind_aug+1]

        model = AutoReg(y_train, lags=min_bic_ind, exog=feature_n_df).fit()

        gr_test_df = pd.concat([X_train[Xs[n]], y_train], axis=1)
        granger_p_stat = grangercausalitytests(gr_test_df, maxlag=[min_bic_ind_aug+1])[min_bic_ind_aug+1][0]['params_ftest'][1]
        if granger_p_stat <= 0.05:
            aug_models[Xs[n]] = model
            feature_n_dfs[Xs[n]] = feature_n_df1
            feature_n_dfs_merge.append(feature_n_df)
            #model.summary()
        elif granger_p_stat <= 0.1:
            print(f'\n\nGranger causality from "{target}" to "{Xs[n]}" is rejected with a p-value={granger_p_stat:.3}')
        else:
            continue


        # aug_models[features[n]] = model
        # feature_n_dfs[features[n]] = feature_n_df1
        # feature_n_dfs_merge.append(feature_n_df)
        # #model.summary()
    
    try:
        feature_n_dfs_merge = pd.concat(feature_n_dfs_merge, axis=1)

        fin_model = AutoReg(y_train, lags=min_bic_ind, exog=feature_n_dfs_merge).fit()

        MAE = np.nanmean(abs(fin_model.predict() - y_train))

        return fin_model, aug_models, feature_n_dfs, feature_n_dfs_merge, MAE
    
    except ValueError:
        print("Can not reject that the target variable 'reverse causes' independent features.")



fin_model, aug_models, dfs, dfs_merged, MAE = algo(df=df_small, target="Close", max_lag=20)

print(fin_model.summary())

print(MAE)