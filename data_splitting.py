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


from Sun_Model_Class import Sun_Model
# logging.INFO



#Reading in the data
df_aapl = pd.read_csv("df_aaple.csv")
# Cleaning the names
col_names = list(df_aapl.columns)
new_names = []
for n in col_names:
    n = re.sub('[^A-Za-z0-9#% ]+', '', n)
    n = re.sub('[^A-Za-z0-9% ]+', 'n', n)
    n = re.sub('[^A-Za-z0-9 ]+', 'pc', n)
    n = re.sub('[^A-Za-z0-9]+', '_', n)
    new_names.append(n)
df_aapl.columns = new_names

# Truncating the dataw
aapl_medium = df_aapl.iloc[:2000,:16]
aapl_long = df_aapl.iloc[:,:16]

df_jpm = pd.read_csv("jpm.csv")
df_jpm = df_jpm.iloc[:2000,:16]

df_dehli = pd.read_csv("dehli_weather.csv")

df_air_q = pd.read_csv("AirQualityUCI.csv")
# df_air_q.drop("Date", axis=1, inplace=True)
# pd.DataFrame.to_csv(df_air_q, "df_air_q_no_date.csv", index=True)



# df_SO_qs = pd.read_csv("MLTollsStackOverflow.csv")
# df_SO_qs.drop("Tableau", axis=1, inplace=True)
# df_SO_qs.drop("rasa", axis=1, inplace=True)
# df_SO_qs.drop("AllenNLP", axis=1, inplace=True)
# df_SO_qs.drop("Ray", axis=1, inplace=True)
# df_SO_qs.drop("Trifacta", axis=1, inplace=True)
# df_SO_qs.drop("Venes", axis=1, inplace=True)
# df_SO_qs.drop("Flair", axis=1, inplace=True)





def algo(df, target, max_lag, test_size):

    # Step 1: Tranformation for stationarity d
    # Here features are everything except for the date
    features = [n for n in list(df.columns) if n != "Date"]

    for feature in features:
        result = adfuller(df[feature], autolag="t-stat")
        counter = 0
        while result[1] >= 0.01:
            df[feature] = df[feature] - df[feature].shift(1)
            #df_small.dropna()
            counter += 1
            #dropna(inplace=False) because it drops one observation for each feature
            result = adfuller(df.dropna()[feature], autolag="t-stat")
        print(f'Order of integration for feature "{feature}" is {counter}')
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    feature_df = df.loc[:, ~df.columns.isin([target, "Date"])]
    target_df = df.loc[:, target]

    X_train, X_test, y_train, y_test = train_test_split(feature_df, target_df, test_size=test_size, shuffle=False) 

    # Step 2: Building a univariate model and finding the optimal l
    BICs = []
    for i in list(range(max_lag)):
        model = AutoReg(y_train, lags=i).fit()
        BICs.append(model.bic)

    min_bic_ind = BICs.index(min(BICs))

    # model = AutoReg(df_small.iloc[:,1], lags=min_bic_ind).fit()
    # model.summary()

    # Due to statsmodels weird properties, you can not test trained model on unseen y-data, but only on unseen X-data.
    # Hence we need to perform some data manipulations to make the testing possible.

    columns_y = []
    for i in list(range(1, min_bic_ind+1)):
        columns_y.append(target+".L"+str(i))

    y_lags_df = pd.DataFrame(columns=columns_y)
    for i in list(range(min_bic_ind)):
        y_lags_df[columns_y[i]] = y_train.shift(i+1)

    # Truncating lags of y at the maximum lag length
    y_lags_df.fillna(1, inplace=True)
    y_lags_df = y_lags_df.iloc[max_lag:,:]
    y_lags_df.reset_index(drop=True, inplace=True)

    # Step 2: Bulding augmented model and finding the optimal w for each Xi
    
    Xs = list(X_train.columns)

    # Truncating y_train for model training at max_lag length
    y_train_m = y_train.iloc[max_lag:]
    y_train_m.reset_index(drop=True, inplace=True)
    # Defining dictionary to store all augmented models
    aug_models = {}
    feature_n_dfs = {}
    feature_n_dfs_merge = [y_lags_df]
    n_lags_for_xi = {}
    
    for n in list(range(len(Xs))):
        columns = []
        for i in list(range(1, max_lag+1)):
            columns.append(Xs[n]+".L"+str(i))

        feature_n_df = pd.DataFrame(columns=columns)
        for i in list(range(max_lag)):
            feature_n_df[columns[i]] = X_train[Xs[n]].shift(i+1)

        # NAs filled with are later automatically truncated by the AutoReg
        feature_n_df.fillna(1, inplace=True)

        feature_n_df = feature_n_df.iloc[max_lag:,:]
        feature_n_df.reset_index(drop=True, inplace=True)
        y_and_x_lags_df = pd.concat([y_lags_df, feature_n_df], axis=1)

        BICs = []
        #Why do I have max_lag-1 and then i+1?
        # +1 is to not make X lags = 0
        # y_and_x_lags_df_m = y_and_x_lags_df.iloc[:,:i+len(list(y_lags_df.columns))+1]
        #y_and_x_lags_df.reset_index(drop=True, inplace=True)
        for i in list(range(max_lag-1)):
            model = AutoReg(y_train_m, lags=0, exog=y_and_x_lags_df.iloc[:,:i+len(list(y_lags_df.columns))+1]).fit()
            BICs.append(model.bic)

        min_bic_ind_aug = BICs.index(min(BICs))
        #Full and Partial autocorrelation plot?
        feature_n_df1 = y_and_x_lags_df
        y_and_x_lags_df = y_and_x_lags_df.iloc[:,:min_bic_ind_aug+len(list(y_lags_df.columns))+1]
        y_and_x_lags_df.reset_index(drop=True, inplace=True)

        model = AutoReg(y_train_m, lags=0, exog=y_and_x_lags_df).fit()

        gr_test_df = pd.concat([X_train[Xs[n]], y_train], axis=1)
        granger_p_stat = grangercausalitytests(gr_test_df, maxlag=[min_bic_ind_aug+1])[min_bic_ind_aug+1][0]['params_ftest'][1]
        if granger_p_stat > 0.05:
            aug_models[Xs[n]] = model
            feature_n_dfs[Xs[n]] = feature_n_df1
            feature_n_dfs_merge.append(y_and_x_lags_df.iloc[:,len(list(y_lags_df.columns)):])
            n_lags_for_xi[Xs[n]] = min_bic_ind_aug + 1
            #model.summary()
        elif granger_p_stat >= 0.01:
            print(f'\n\nGranger causality from "{target}" to "{Xs[n]}" can not be rejected with a p-value={granger_p_stat:.3}')
        else:
            continue


        # aug_models[features[n]] = model
        # feature_n_dfs[features[n]] = feature_n_df1
        # feature_n_dfs_merge.append(feature_n_df)
        # #model.summary()
    
    try:
        feature_n_dfs_merge = pd.concat(feature_n_dfs_merge, axis=1)

        fin_model = AutoReg(y_train_m, lags=0, exog=feature_n_dfs_merge).fit()

        y_pred_in = fin_model.predict()
        MAE_train = np.nanmean(abs(y_pred_in - y_train_m))

        # Formatting the test dataframes to suit the model's exog format
        test_data = []

        #Finding the maximum seleceted lag length to truncate the test data appropriately
        selected_lag_lens = []
        selected_lag_lens.append(min_bic_ind)
        for x_name, lag_len in n_lags_for_xi.items():
            selected_lag_lens.append(lag_len)

        max_sel_lag = max(selected_lag_lens)

        # Formatting y
        y_lags_df = pd.DataFrame(columns=columns_y)
        for i in list(range(min_bic_ind)):
            y_lags_df[columns_y[i]] = y_test.shift(i+1)

        # Truncating lags of y at the maximum lag length
        # y_lags_df.fillna(1, inplace=True)
        y_lags_df = y_lags_df.iloc[max_sel_lag:,:]
        y_lags_df.reset_index(drop=True, inplace=True)

        test_data.append(y_lags_df)

        #Formatting Xs    
        for x_name, lag_len in n_lags_for_xi.items():
            columns = []
            for i in list(range(1, lag_len+1)):
                columns.append(x_name+".L"+str(i))

            feature_x_df = pd.DataFrame(columns=columns)
            for i in list(range(lag_len)):
                feature_x_df[columns[i]] = X_test[x_name].shift(i+1)
            
            feature_x_df = feature_x_df.iloc[max_sel_lag:,:]
            feature_x_df.reset_index(drop=True, inplace=True)
            
            test_data.append(feature_x_df)
    
        # Merging y and Xs
        test_data = pd.concat(test_data, axis=1)

        # Truncating y_test, so its length corresponds to that of y_train_m
        y_test = y_test.iloc[max_sel_lag:]
        y_test.reset_index(drop=True, inplace=True)

        first_oos_ind = len(y_train_m)
        last_oos_ind = first_oos_ind + len(y_test) - 1
        y_pred_out = fin_model.predict(start=first_oos_ind, end=last_oos_ind, exog_oos=test_data)
        y_pred_out.reset_index(drop=True, inplace=True)
        MAE_test = np.nanmean(abs(y_pred_out - y_test))
        #MAE_test = np.nanmean(abs(fin_model.predict(start=1579, end=1972, exog_oos=test_data) - y_test))
        
        MAE = {"train": MAE_train, "test": MAE_test}
        logging.info("Check")

        Model_Data = Sun_Model(fin_model, fin_model.summary(), aug_models, MAE,
                                y_train_m, feature_n_dfs_merge,
                                y_test, test_data,
                                y_pred_out)

        #return fin_model, aug_models, feature_n_dfs, feature_n_dfs_merge, MAE, Sun_Model1
        return Model_Data
    except ValueError:
        logging.error("Can not reject that the target variable 'reverse causes' independent features.")



#fin_model, aug_models, dfs, dfs_merged, MAE, Model = algo(df=df_medium, target="Close", max_lag=20)
Model_Data = algo(df=df_air_q, target="meantemp", max_lag=20, test_size=0.2)

print(Model_Data.summary)


print(Model_Data.MAE)
# print(Model_Data.train_y)


filename = 'Sun_Model_Data'
outfile = open(filename,'wb')
pickle.dump(Model_Data,outfile)
outfile.close()

# {'train': 1.2125139241871459, 'test': 1.199242993289765}