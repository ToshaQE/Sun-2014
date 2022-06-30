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
# import lightgbm as lgb
import seaborn as sns

from sklearn.datasets import make_regression
from scipy.stats import pearsonr
from tqdm.notebook import trange, tqdm
from FRUFS import FRUFS
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler




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






def algo(df, target, max_lag, stationarity_method, test_size):

    # Cleaning column names
    col_names = list(df.columns)
    new_names = []
    for n in col_names:
        n = re.sub('[^A-Za-z0-9#% ]+', '', n)
        n = re.sub('[^A-Za-z0-9% ]+', 'n', n)
        n = re.sub('[^A-Za-z0-9 ]+', 'pc', n)
        n = re.sub('[^A-Za-z0-9]+', '_', n)
        new_names.append(n)
    df.columns = new_names

    # Step 1: Tranformation for stationarity d
    # Here features are everything except for the date
    feature_df = df.loc[:, ~df.columns.isin([target, "Date"])]
    target_df = df.loc[:, target]

    X_train, X_test, y_train, y_test = train_test_split(feature_df, target_df, test_size=test_size, shuffle=False)
    
    column_names = list(X_train.columns)

    scaling = MinMaxScaler(feature_range=(0,1)).fit(X_train)
    X_train = pd.DataFrame(scaling.transform(X_train), columns = column_names)
    X_test = pd.DataFrame(scaling.transform(X_test), columns = column_names)

    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    
    staionarity_df = pd.concat([y_train, X_train], axis=1)

    features = list(staionarity_df.columns)

    # features = [n for n in list(X_train.columns) if n != "Date"]
    
    yx_train_original = staionarity_df.copy()

    orders_of_integ = {}
    const_counters = {}

    for feature in features:
        result = adfuller(staionarity_df[feature], autolag="t-stat")
        counter = 0
        if stationarity_method == 0:
            while result[1] >= 0.01:
                staionarity_df[feature] = staionarity_df[feature] - staionarity_df[feature].shift(1)
                #df_small.dropna()
                counter += 1
                #dropna(inplace=False) because it drops one observation for each feature
                result = adfuller(staionarity_df.dropna()[feature], autolag="t-stat")
            print(f'Order of integration for feature "{feature}" is {counter}')
            orders_of_integ[feature] = counter
        elif stationarity_method == 1:
            feature_logged = np.log(staionarity_df[feature])
            inf_count = np.isinf(feature_logged).sum()
            const_counter = 0
            # If inf count is greater than 0 it is likely that original series contains 0s or negative values
            # Hence we add a constant to the series and only then apply log tranformations until there are no zeroes/nrgative values
            while inf_count > 0:
                if const_counter == 0:
                    feature_with_constant = staionarity_df[feature] + 1
                    feature_logged = np.log(feature_with_constant)
                    inf_count = np.isinf(feature_logged).sum()
                    const_counter += 1
                else:
                    feature_with_constant = feature_with_constant + 1
                    feature_logged = np.log(feature_with_constant)
                    inf_count = np.isinf(feature_logged).sum()
                    const_counter += 1

            while result[1] >= 0.01:
                feature_differenced = feature_logged.diff()
                staionarity_df[feature] = feature_differenced
                #df_small.dropna()
                counter += 1
                #dropna(inplace=False) because it drops one observation for each feature
                result = adfuller(staionarity_df.dropna()[feature], autolag="t-stat")
            print(f'Order of integration for feature "{feature}" is {counter}')
            orders_of_integ[feature] = counter
            const_counters[feature] = const_counter

    staionarity_df.dropna(inplace=True)
    staionarity_df.reset_index(drop=True, inplace=True)

    y_train = staionarity_df[target]
    X_train = staionarity_df[[n for n in list(staionarity_df.columns) if n != target]]

    #Error Correction Model
    cointegr_dict = {}
    ECM_residuals = {}
    ECM_OLSs = {}
    x_features = list(X_train.columns)

    for x_feature in x_features:
        if orders_of_integ[target] == orders_of_integ[x_feature] & orders_of_integ[target] != 0:
            E_G = engle_granger(yx_train_original[target], yx_train_original[x_feature], trend = "n", method = "t-stat")
            if E_G.pvalue < 0.05:
                cointegr_dict[x_feature] = 1
                ECM_X_const = sm.add_constant(yx_train_original[x_feature])
                ECM_1 = OLS(yx_train_original[target], ECM_X_const, hasconst=True).fit(cov_type=("HC0"))
                ECM_res = ECM_1.resid
                ECM_res = ECM_res.shift(1).dropna().reset_index(drop=True)
                ECM_res = ECM_res.iloc[max_lag:].reset_index(drop=True)
                ECM_residuals[x_feature] = ECM_res
                ECM_OLSs[x_feature] = ECM_1

            # ECM_1 = OLS(yx_train_original[target], yx_train_original[feature]).fit(cov_type=("HC0"))
            # ECM_res = ECM_1.resid
            # ECM_adf = adfuller(ECM_res, autolag = "t-stat", regrression = "nc")[1]

            







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
    # y_lags_df.fillna(1, inplace=True)
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
    
    for x in Xs:
        columns = []
        for i in list(range(1, max_lag+1)):
            columns.append(x + ".L"+str(i))

        feature_n_df = pd.DataFrame(columns=columns)
        for i in list(range(max_lag)):
            feature_n_df[columns[i]] = X_train[x].shift(i+1)

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

        # Adding ECM residual to the feature n dataset if it appears to be cointegrated with the target
        if x in list(ECM_residuals.keys()):
            ECM_res_name = x + "_ECM_Res.L1"
            y_and_x_lags_df[ECM_res_name] = ECM_residuals[x]

        model = AutoReg(y_train_m, lags=0, exog=y_and_x_lags_df).fit()

        gr_test_df = pd.concat([X_train[x], y_train], axis=1)
        granger_p_stat = grangercausalitytests(gr_test_df, maxlag=[min_bic_ind_aug+1])[min_bic_ind_aug+1][0]['params_ftest'][1]
        if granger_p_stat >= 0.05:
            aug_models[x] = model
            feature_n_dfs[x] = feature_n_df1
            feature_n_dfs_merge.append(y_and_x_lags_df.iloc[:,len(list(y_lags_df.columns)):])
            n_lags_for_xi[x] = min_bic_ind_aug + 1
            #model.summary()
        elif granger_p_stat >= 0.01:
            print(f'\n\nGranger causality from "{target}" to "{x}" can not be rejected with a p-value={granger_p_stat:.3}')
        else:
            continue


        # aug_models[features[n]] = model
        # feature_n_dfs[features[n]] = feature_n_df1
        # feature_n_dfs_merge.append(feature_n_df)
        # #model.summary()
    
    try:
        
        feature_n_dfs_merge = pd.concat(feature_n_dfs_merge, axis=1)

        if len(feature_n_dfs_merge) == 0:
                    print("\n\n\n\nZero lags of y have been selected and H0 of reverse causlity could not be rejected for any X.\n\n\n\n")

        fin_model = AutoReg(y_train_m, lags=0, exog=feature_n_dfs_merge).fit()




        # Using backward elimination to drop insignificant features
        # Defining critiacl p-value determining whether a feture is to be dropped
        critical_p_value = 0.05
        # Finding p-value of the lesat siginificant feature
        max_p_value = max(fin_model.pvalues)
        # Defining const_dropped to know whether we run Autoreg with or w/o const
        const_dropped = False
        while max_p_value >= critical_p_value:
            # Column name of the least significant feature
            least_sig_var = list(fin_model.params[np.where(fin_model.pvalues == max_p_value)[0]].index)[0]
            # If least_sig_var is the constant we run Autoreg without it
            if least_sig_var == "const":
                fin_model = AutoReg(y_train_m, lags=0, exog=feature_n_dfs_merge, trend="n").fit()
                const_dropped = True

            else:
                # Dropping the least_sig_var from the df
                feature_n_dfs_merge.pop(least_sig_var)
                # If const has been dropped, we run Autoreg w/o it
                if const_dropped:
                    try:
                        fin_model = AutoReg(y_train_m, lags=0, exog=feature_n_dfs_merge, trend="n").fit()
                    except ValueError:
                        print("\n\n\n\nNo coefficients appear to be significant in the estimated model.\n\n\n\n")
                else:
                    fin_model = AutoReg(y_train_m, lags=0, exog=feature_n_dfs_merge).fit()

            # At the end of each iteration we find the new highest p-value        
            max_p_value = max(fin_model.pvalues)

        # Defining list of all significant variables (except for const - because it is not in the data)
        names_of_sig_vars = [n for n in list(fin_model.params.index) if n!= "const"]


        y_pred_in = fin_model.predict()
        MAE_train = np.nanmean(abs(y_pred_in - y_train_m))

        # Coppying data for ECM imlplementation
        y_test_non_stat = y_test.copy()
        X_test_non_stat = X_test.copy()

        # Stationarising test data
        stationarity_df_test = pd.concat([y_test, X_test], axis=1)
        features = list(stationarity_df_test.columns)

        if stationarity_method == 0:
        # if transformation is simple differencing
            for feature in features:
                # Continue if the feature was found to be stationary withoud tranformation
                if orders_of_integ[feature] == 0:
                    continue
                else:
                    order = orders_of_integ[feature]
                    integr_list = list(range(order, order+1))
                    # Difference o times as with the train data
                    for o in integr_list:
                        stationarity_df_test[feature] = stationarity_df_test[feature].diff()

        elif stationarity_method == 1:
        # if transformation is log differencing
            for feature in features:
                # Continue if the feature was found to be stationary withoud tranformation
                if orders_of_integ[feature] == 0:
                        continue
                else:
                    # Check whether any constants were added to the training data
                    if const_counters[feature] > 0:
                        stationarity_df_test[feature] = stationarity_df_test[feature] + const_counters[feature]
                    # Logging the data
                    stationarity_df_test[feature] = np.log(stationarity_df_test[feature])
                    order = orders_of_integ[feature]
                    integr_list = list(range(order, order+1))
                    # Difference o times as with the train data
                    for o in integr_list:
                        stationarity_df_test[feature] = stationarity_df_test[feature].diff()


        stationarity_df_test.dropna(inplace=True)
        stationarity_df_test.reset_index(drop=True, inplace=True)

        y_test = stationarity_df_test[target]
        X_test = stationarity_df_test[[n for n in list(stationarity_df_test.columns) if n != target]]

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


        #Formatting Xs and implementing ECM on the test data
        Cointegrated_Xs = list(ECM_residuals.keys())

        for x_name, lag_len in n_lags_for_xi.items():
            columns = []
            for i in list(range(1, lag_len+1)):
                columns.append(x_name+".L"+str(i))

            feature_x_df = pd.DataFrame(columns=columns)
            for i in list(range(lag_len)):
                feature_x_df[columns[i]] = X_test[x_name].shift(i+1)
            
            feature_x_df = feature_x_df.iloc[max_sel_lag:,:]
            feature_x_df.reset_index(drop=True, inplace=True)

            if x_name in Cointegrated_Xs:
                ECM_OLS = ECM_OLSs[x_name]
                X_test_ECM = sm.add_constant(X_test_non_stat[x_name])
                y_ECM_pred = ECM_OLS.predict(X_test_ECM)
                y_test_ECM_res = y_ECM_pred - y_test_non_stat
                y_test_ECM_res = y_test_ECM_res.shift(1)
                y_test_ECM_res = y_test_ECM_res.iloc[max_sel_lag:]
                y_test_ECM_res.reset_index(drop=True, inplace=True)
                feature_x_df[x_name + "_ECM_Res.L1"] = y_test_ECM_res
            
            test_data.append(feature_x_df)
    
        # Merging y and Xs
        test_data = pd.concat(test_data, axis=1)
        # Only keeping the significant features
        test_data = test_data[names_of_sig_vars]
        # Truncating y_test, so its length corresponds to that of y_train_m
        y_test = y_test.iloc[max_sel_lag:]
        y_test.reset_index(drop=True, inplace=True)

        first_oos_ind = len(y_train_m)
        last_oos_ind = first_oos_ind + len(y_test) - 1
        y_pred_out = fin_model.predict(start=first_oos_ind, end=last_oos_ind, exog_oos=test_data)
        y_pred_out.reset_index(drop=True, inplace=True)
        MAE_test = np.nanmean(abs(y_pred_out - y_test))
        
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
# df_air_q.drop("Date", axis=1, inplace=True)
# pd.DataFrame.to_csv(df_air_q, "df_air_q_no_date.csv", index=True)


airpassengers_df = pmd.datasets.load_airpassengers(as_series = True)
msft_pmd_df = pmd.datasets.load_msft()
msft_pmd_df = msft_pmd_df.iloc[:,:-1]


#fin_model, aug_models, dfs, dfs_merged, MAE, Model = algo(df=df_medium, target="Close", max_lag=20)

Model_Data = algo(df=aapl_medium, target="Close", max_lag=20, stationarity_method = 0, test_size=0.2)

print(Model_Data.summary)


print(Model_Data.MAE)
# print(Model_Data.train_y)


filename = 'Sun_Model_Data'
outfile = open(filename,'wb')
pickle.dump(Model_Data,outfile)
outfile.close()

# apple_long_evoML = pd.concat([Model_Data.y_train])

# {'train': 1.2125139241871459, 'test': 1.199242993289765}











