import pandas as pd
import numpy as np
from scipy import interpolate
import joblib
import plotly
import plotly.graph_objs as go
import plotly.io as pio
from plotly.graph_objects import Layout
from random import randint
from datetime import time
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from time import time
from sklearn import preprocessing
import seaborn as sns; sns.set()
from tensorflow.compat.v1.keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestRegressor
from tensorflow.compat.v1.keras.callbacks import ModelCheckpoint
from sklearn import svm
from sklearn import metrics
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from plotly.subplots import make_subplots
import plotly.graph_objects as go


from random import randint
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from tensorflow.compat.v1.keras.models import Sequential
from tensorflow.compat.v1.keras.layers import Dense
from tensorflow.compat.v1.keras.layers import Masking

from tensorflow.compat.v1.keras.models import model_from_json
import pickle
from pickle import dump
import influxdb
import datetime
import plotly

#FUNCTIONS
######################################################################################################################
def ts(new_data, look_back, pred_col, dim, names):
    t = new_data.copy()
    t['id'] = range(0, len(t) )
    #t = t.iloc[:-look_back, :]
    t = t.iloc[look_back:, :]
    t.set_index('id', inplace=True)
    pred_value = new_data.copy()
    pred_value = pred_value.iloc[:-look_back, pred_col]
    # pred_value.columns = names[np.array([3,4,5])]
    pred_value.columns = names[pred_col]
    pred_value = pd.DataFrame(pred_value)

    pred_value['id'] = range(1, len(pred_value) + 1)
    pred_value.set_index('id', inplace=True)
    final_df = pd.concat([t, pred_value], axis=1)

    return final_df

#Train and save the random forest models and its data scalers
def Training(x_train,y_train, ntest,rf_param_grid,filename,var,med):
    x_train = x_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    test = [0 for x in range(ntest)]
    for i in range(ntest):
        test[i] = randint(0, x_train.shape[0] - 1)

    x_test= x_train.iloc[test]
    y_test = y_train.iloc[test]
    x_train= x_train.drop(test)
    y_train = y_train.drop(test)

    names = x_train.columns
    temperaturas_train = np.array(x_train.iloc[:, np.array([1, 4, 7, 10, 13, 16, 19, 22, 26, 31, 34, 37, 40, 43, 46, 49, 52])])
    temperaturas_test = np.array(x_test.iloc[:, np.array([1, 4, 7, 10, 13, 16, 19, 22, 26, 31, 34, 37, 40, 43, 46, 49, 52])])
    humedad_train = np.array(x_train.iloc[:, np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 30, 33, 36, 39, 42, 45, 48, 51])])
    humedad_test = np.array(x_test.iloc[:, np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 30, 33, 36, 39, 42, 45, 48, 51])])
    co2s_train = np.array(x_train.iloc[:, np.array([2, 5, 8, 11, 14, 17, 20, 23, 32, 35, 38, 41, 44, 47, 50, 53])])
    co2s_test = np.array(x_test.iloc[:, np.array([2, 5, 8, 11, 14, 17, 20, 23, 32, 35, 38, 41, 44, 47, 50, 53])])
    diss_train = np.array(x_train.iloc[:, range(x_train.shape[1] - 24, x_train.shape[1])])
    diss_test = np.array(x_test.iloc[:, range(x_train.shape[1] - 24, x_train.shape[1])])
    rad_train = np.array(x_train.iloc[:, np.array([25])])
    rad_test = np.array(x_test.iloc[:, np.array([25])])
    resto_train = x_train.iloc[:, np.array([27, 28, 29])]
    resto_test = x_test.iloc[:, np.array([27, 28, 29])]

    scalar_temp = MinMaxScaler(feature_range=(-1, 1))
    scalar_hum = MinMaxScaler(feature_range=(-1, 1))
    scalar_co2 = MinMaxScaler(feature_range=(-1, 1))
    scalardist = MinMaxScaler(feature_range=(-1, 1))
    scalar_rad = MinMaxScaler(feature_range=(-1, 1))
    scalarresto = MinMaxScaler(feature_range=(-1, 1))
    scalarY = MinMaxScaler(feature_range=(-1, 1))
    if var == 'temp':
        scalar_temp.fit(np.concatenate((np.concatenate(temperaturas_train), np.concatenate(temperaturas_test), y_train, y_test)).reshape(-1, 1))
        scalardist.fit(np.concatenate((np.concatenate(diss_train), np.concatenate(diss_test))).reshape(-1, 1))
        scalar_hum.fit(np.concatenate((np.concatenate(humedad_train), np.concatenate(humedad_test))).reshape(-1, 1))
        scalar_co2.fit(np.concatenate((np.concatenate(co2s_train), np.concatenate(co2s_test))).reshape(-1, 1))
        scalar_rad.fit(np.concatenate((np.concatenate(rad_train), np.concatenate(rad_test))).reshape(-1, 1))
        scalarresto.fit(pd.concat([resto_train, resto_test], axis=0))
        y_train1 = pd.DataFrame(scalar_temp.transform(np.array(y_train).reshape(-1, 1)))
        y_test1 = pd.DataFrame(scalar_temp.transform(np.array(y_test).reshape(-1, 1)))
    elif var == 'hum':
        scalar_temp.fit(
            np.concatenate((np.concatenate(temperaturas_train), np.concatenate(temperaturas_test))).reshape(-1, 1))
        scalardist.fit(np.concatenate((np.concatenate(diss_train), np.concatenate(diss_test))).reshape(-1, 1))
        scalar_hum.fit(
            np.concatenate((np.concatenate(humedad_train), np.concatenate(humedad_test), y_train, y_test)).reshape(-1,
                                                                                                                     1))
        scalar_co2.fit(np.concatenate((np.concatenate(co2s_train), np.concatenate(co2s_test))).reshape(-1, 1))
        scalar_rad.fit(np.concatenate((np.concatenate(rad_train), np.concatenate(rad_test))).reshape(-1, 1))
        scalarresto.fit(pd.concat([resto_train, resto_test], axis=0))
        y_train1 = pd.DataFrame(scalar_hum.transform(np.array(y_train).reshape(-1, 1)))
        y_test1 = pd.DataFrame(scalar_hum.transform(np.array(y_test).reshape(-1, 1)))
    else:
        scalar_temp.fit(
            np.concatenate((np.concatenate(temperaturas_train), np.concatenate(temperaturas_test))).reshape(-1, 1))
        scalardist.fit(np.concatenate((np.concatenate(diss_train), np.concatenate(diss_test))).reshape(-1, 1))
        scalar_hum.fit(np.concatenate((np.concatenate(humedad_train), np.concatenate(humedad_test))).reshape(-1, 1))
        scalar_co2.fit(
            np.concatenate((np.concatenate(co2s_train), np.concatenate(co2s_test), y_train, y_test)).reshape(-1, 1))
        scalar_rad.fit(np.concatenate((np.concatenate(rad_train), np.concatenate(rad_test))).reshape(-1, 1))
        scalarresto.fit(pd.concat([resto_train, resto_test], axis=0))
        y_train1 = pd.DataFrame(scalar_co2.transform(np.array(y_train).reshape(-1, 1)))
        y_test1 = pd.DataFrame(scalar_co2.transform(np.array(y_test).reshape(-1, 1)))
    temperaturas_train1 = np.zeros((temperaturas_train.shape[0], temperaturas_train.shape[1]))
    temperaturas_test1 = np.zeros((temperaturas_test.shape[0], temperaturas_test.shape[1]))
    humedad_train1 = np.zeros((temperaturas_train.shape[0], temperaturas_train.shape[1]))
    humedad_test1 = np.zeros((temperaturas_test.shape[0], temperaturas_test.shape[1]))
    for i in range(temperaturas_train.shape[1]):
        temperaturas_train1[:, i] = scalar_temp.transform(temperaturas_train[:, i].reshape(-1, 1))[:, 0]
        temperaturas_test1[:, i] = scalar_temp.transform(temperaturas_test[:, i].reshape(-1, 1))[:, 0]
        humedad_train1[:, i] = scalar_hum.transform(humedad_train[:, i].reshape(-1, 1))[:, 0]
        humedad_test1[:, i] = scalar_hum.transform(humedad_test[:, i].reshape(-1, 1))[:, 0]
    temperaturas_train1 = pd.DataFrame(temperaturas_train1)
    temperaturas_test1 = pd.DataFrame(temperaturas_test1)
    humedad_train1 = pd.DataFrame(humedad_train1)
    humedad_test1 = pd.DataFrame(humedad_test1)
    co2s_train1 = np.zeros((co2s_train.shape[0], co2s_train.shape[1]))
    co2s_test1 = np.zeros((co2s_test.shape[0], co2s_train.shape[1]))
    for i in range(co2s_train.shape[1]):
        co2s_train1[:, i] = scalar_co2.transform(co2s_train[:, i].reshape(-1, 1))[:, 0]
        co2s_test1[:, i] = scalar_co2.transform(co2s_test[:, i].reshape(-1, 1))[:, 0]
    co2s_train1 = pd.DataFrame(co2s_train1)
    co2s_test1 = pd.DataFrame(co2s_test1)
    diss_train1 = np.zeros((diss_train.shape[0], diss_train.shape[1]))
    diss_test1 = np.zeros((diss_test.shape[0], diss_train.shape[1]))
    for i in range(diss_train.shape[1]):
        diss_train1[:, i] = scalardist.transform(diss_train[:, i].reshape(-1, 1))[:, 0]
        diss_test1[:, i] = scalardist.transform(diss_test[:, i].reshape(-1, 1))[:, 0]
    rad_train1 = np.zeros((rad_train.shape[0], rad_train.shape[1]))
    rad_test1 = np.zeros((rad_test.shape[0], rad_train.shape[1]))
    for i in range(rad_train.shape[1]):
        rad_train1[:, i] = scalar_rad.transform(rad_train[:, i].reshape(-1, 1))[:, 0]
        rad_test1[:, i] = scalar_rad.transform(rad_test[:, i].reshape(-1, 1))[:, 0]
    diss_train1 = pd.DataFrame(diss_train1)
    diss_test1 = pd.DataFrame(diss_test1)
    rad_train1 = pd.DataFrame(rad_train1)
    rad_test1 = pd.DataFrame(rad_test1)
    resto_train1 = pd.DataFrame(scalarresto.transform(resto_train))
    resto_test1 = pd.DataFrame(scalarresto.transform(resto_test))

    if var=='temp':
        x_train1 = pd.concat([temperaturas_train1, rad_train1, resto_train1, diss_train1], axis=1)
        x_test1 = pd.concat([temperaturas_test1, rad_test1, resto_test1, diss_test1], axis=1)

        miss = x_train1.apply(lambda x: x.count(), axis=1) - 45
        miss = np.where(miss <= -6)[0]
        x_train1 = x_train1.drop(miss, axis=0)
        x_train1 = x_train1.reset_index(drop=True)
        y_train1 = y_train1.drop(miss, axis=0)
        y_train1 = y_train1.reset_index(drop=True)

        miss = x_test1.apply(lambda x: x.count(), axis=1) - 45
        miss = np.where(miss <= -6)[0]
        x_test1 = x_test1.drop(miss, axis=0)
        x_test1 = x_test1.reset_index(drop=True)
        y_test1 = y_test1.drop(miss, axis=0)
        y_test1 = y_test1.reset_index(drop=True)

    elif var=='hum':
        x_train1 = pd.concat([humedad_train1, rad_train1, resto_train1, diss_train1], axis=1)
        x_test1 = pd.concat([humedad_test1,  rad_test1, resto_test1, diss_test1], axis=1)

        miss = x_train1.apply(lambda x: x.count(), axis=1) - 45
        miss = np.where(miss <= -6)[0]
        x_train1 = x_train1.drop(miss, axis=0)
        x_train1 = x_train1.reset_index(drop=True)
        y_train1 = y_train1.drop(miss, axis=0)
        y_train1 = y_train1.reset_index(drop=True)

        miss = x_test1.apply(lambda x: x.count(), axis=1) - 45
        miss = np.where(miss <= -6)[0]
        x_test1 = x_test1.drop(miss, axis=0)
        x_test1 = x_test1.reset_index(drop=True)
        y_test1 = y_test1.drop(miss, axis=0)
        y_test1 = y_test1.reset_index(drop=True)

    else:
        x_train1 = pd.concat([co2s_train1, resto_train1, diss_train1], axis=1)
        x_test1 = pd.concat([co2s_test1 , resto_test1, diss_test1], axis=1)

        miss = x_train1.apply(lambda x: x.count(), axis=1) - 43
        miss = np.where(miss <= -8)[0]
        x_train1 = x_train1.drop(miss, axis=0)
        x_train1 = x_train1.reset_index(drop=True)
        y_train1 = y_train1.drop(miss, axis=0)
        y_train1 = y_train1.reset_index(drop=True)

        miss = x_test1.apply(lambda x: x.count(), axis=1) - 43
        miss = np.where(miss <= -8)[0]
        x_test1 = x_test1.drop(miss, axis=0)
        x_test1 = x_test1.reset_index(drop=True)
        y_test1 = y_test1.drop(miss, axis=0)
        y_test1 = y_test1.reset_index(drop=True)

    x_train1.columns = range(x_train1.shape[1])
    x_test1.columns = range(x_test1.shape[1])

    zz = []
    for t in range(x_train1.shape[1]):
        a = np.array(x_train1.iloc[:, t], dtype=float)
        if len(np.where(np.isnan(a))[0]) > 0:
            zz.append(np.where(np.isnan(a))[0])
            a[np.where(np.isnan(a))[0]] = np.repeat(-10, len(np.where(np.isnan(a))[0]))
            x_train1.iloc[:, t] = a
    for t in range(x_test1.shape[1]):
        a = np.array(x_test1.iloc[:, t], dtype=float)
        if len(np.where(np.isnan(a))[0]) > 0:
            zz.append(np.where(np.isnan(a))[0])
            a[np.where(np.isnan(a))[0]] = np.repeat(-10, len(np.where(np.isnan(a))[0]))
            x_test1.iloc[:, t] = a
    out_train = np.where(np.isnan(y_train1))[0]
    if len(out_train) > 0:
        y_train1 = y_train1.drop(out_train, axis=0)
        y_train1 = y_train1.reset_index(drop=True)
        x_train1 = x_train1.drop(out_train, axis=0)
        x_train1 = x_train1.reset_index(drop=True)
    out_test = np.where(np.isnan(y_test1))[0]
    if len(out_train) > 0:
        y_test1 = y_test1.drop(out_test, axis=0)
        y_test1 = y_test1.reset_index(drop=True)
        x_test1 = x_test1.drop(out_test, axis=0)
        x_test1 = x_test1.reset_index(drop=True)

    # RANDOM FOREST
    model = RandomForestRegressor()
    model_RF = GridSearchCV(estimator=model, param_grid=rf_param_grid, scoring='neg_mean_absolute_error', cv=10,
                            verbose=1, n_jobs=2)

    X = pd.concat([x_train1, x_test1], axis=0)
    Y = pd.concat([y_train1, y_test1], axis=0)
    model_RF.fit(X, np.ravel(np.array(Y)))
    names_pam = list(model_RF.param_grid)
    model = RandomForestRegressor(n_estimators=model_RF.best_params_[names_pam[0]],
                                  max_depth=model_RF.best_params_[names_pam[1]],
                                  min_samples_split=model_RF.best_params_[names_pam[2]], n_jobs=5)
    # model = RandomForestRegressor(min_samples_split=grid_rf.best_params_[names_pam[0]], n_jobs=5)
    model.fit(x_train1, np.ravel(np.array(y_train1)))
    ##################################################################################################
    y_pred= model.predict(x_test1)
    if var == 'temp':
        predictions = np.array(scalar_temp.inverse_transform(y_pred.reshape(x_test1.shape[0], 1)))
        y_test11 = np.array(scalar_temp.inverse_transform(y_test1))
    elif var == 'hum':
        predictions = np.array(scalar_hum.inverse_transform(y_pred.reshape(x_test1.shape[0], 1)))
        y_test11 = np.array(scalar_hum.inverse_transform(y_test1))
    else:
        predictions = np.array(scalar_co2.inverse_transform(y_pred.reshape(x_test1.shape[0], 1)))
        y_test11 = np.array(scalar_co2.inverse_transform(y_test1))


    accuracy = 100*(np.sqrt(metrics.mean_squared_error(y_test11, predictions)) / med)
    print('The accuracy (CV RMSE) of the model trained is ', accuracy)
    pickle.dump(model, open(filename, 'wb'))
    print('Model RF saved')

    y = [filename, 'Scaler_temp']
    y2 = [filename, 'Scaler_hum']
    y3 = [filename, 'Scaler_co2']
    y4 = [filename, 'Scaler_dist']
    y5 = [filename, 'Scaler_rad']
    y6 = [filename, 'Scaler_resto']
    sep = '-'
    dump(scalar_temp, open(sep.join(y), 'wb'))
    dump(scalar_hum, open(sep.join(y2), 'wb'))
    dump(scalar_co2, open(sep.join(y3), 'wb'))
    dump(scalardist, open(sep.join(y4), 'wb'))
    dump(scalar_rad, open(sep.join(y5), 'wb'))
    dump(scalarresto, open(sep.join(y6), 'wb'))


    print('Scalers RF saved')

def loading(var_name, sensor_name, host, variables, time_end):
    #info influx

    influx = influxdb.InfluxDBClient.from_dsn(CONN_STR)
    influx.switch_database(DB_NAME)

    place = ["sensor_data.autogen",sensor_name]
    sep='.'
    place = sep.join(place)
    place2 = [sensor_name,"address"]
    sep = '.'
    place2 = sep.join(place2)
    time_end_str = time_end.strftime('%Y-%m-%dT%H:%M:%SZ')

    if host=='none':
        for u in range(len(var_name)):

            var2 = [var_name[u], 'vc']
            sep = '_'
            var2 = sep.join(var2)
            query = f"""
                SELECT mean({var_name[u]}) AS {var_name[u]} FROM {place} 
                WHERE time > '2020-06-17T11:00:00Z' AND time < '{time_end_str}' AND {var2}<3
                  AND {place2} != '69' GROUP BY time(10m) FILL(9999)
            """

            results = influx.query(query)

            point = list(results)[0]
            values = [0 for x in range(len(point))]
            for t in range(len(point)):
                values[t] = point[t][var_name[u]]

            variables.append(values)

    else:

        for u in range(len(var_name)):
            query = f"""
                SELECT mean({var_name[u]}) AS {var_name[u]} FROM {place} 
                WHERE time > '2020-06-17T11:00:00Z' AND time < '{time_end_str}'  AND "host"='{host}'
                  AND {place2} != '69' GROUP BY time(1m) FILL(9999)
            """

            results = influx.query(query)

            point = list(results)[0]
            values = [0 for x in range(len(point))]
            #dates = [0 for x in range(len(point))]
            for t in range(len(point)):
                values[t]=point[t][var_name[u]]
                #dates[t] = point[t]['time']

            variables.append(values)
            #dates = pd.to_datetime(pd.Series(dates), format='%Y-%m-%dT%H:%M:%SZ')

    return variables

######################################################################################################################
def loading_carrito(var_name, sensor_name, host, variables, time_end):
    #info influx

    influx = influxdb.InfluxDBClient.from_dsn(CONN_STR)
    influx.switch_database(DB_NAME)

    place = ["sensor_data.autogen",sensor_name]
    sep='.'
    place = sep.join(place)
    place2 = [sensor_name,"address"]
    sep = '.'
    place2 = sep.join(place2)
    time_end_str = time_end.strftime('%Y-%m-%dT%H:%M:%SZ')

    if host=='none':
        for u in range(len(var_name)):

            var2 = [var_name[u], 'vc']
            sep = '_'
            var2 = sep.join(var2)
            query = f"""
                            SELECT mean({var_name[u]}) AS {var_name[u]} FROM {place} 
                            WHERE time > '2020-06-17T11:00:00Z' AND time < '{time_end_str}' AND {var2}<3
                              AND {place2} != '69' GROUP BY time(10m) FILL(9999)
                        """

            results = influx.query(query)

            point = list(results)[0]
            values = [0 for x in range(len(point))]
            #dates = [0 for x in range(len(point))]
            for t in range(len(point)):
                values[t] = point[t][var_name[u]]
                #dates[t] = point[t]['time']

            variables.append(values)

            #dates = pd.to_datetime(pd.Series(dates), format='%Y-%m-%dT%H:%M:%SZ')
            #dates = dates.dt.floor('Min')

    else:

        for u in range(len(var_name)):
            query = f"""
                            SELECT mean({var_name[u]}) AS {var_name[u]} FROM {place} 
                            WHERE time > '2020-06-17T11:00:00Z' AND time < '{time_end_str}'  AND "host"='{host}'
                              AND {place2} != '69' GROUP BY time(1m) FILL(linear)
                        """

            results = influx.query(query)

            point = list(results)[0]
            values = [0 for x in range(len(point))]
            #dates = [0 for x in range(len(point))]
            for t in range(len(point)):
                values[t]=point[t][var_name[u]]
                #dates[t] = point[t]['time']

            variables.append(values)

            #dates= pd.to_datetime(pd.Series(dates), format='%Y-%m-%dT%H:%M:%SZ')
            #dates = dates.dt.floor('Min')

    return variables


def MTI_train(DATA, date_init, ntest, rf_param_grid,var,names):
    #names = DATA.columns
    date = pd.date_range(date_init, periods=DATA.shape[0], freq='1min')
    # TRAINING
    # Hour of day
    HOUR = [0 for x in range(len(date))]
    for h in range(len(date)):
        HOUR[h] = date[h].hour + date[h].minute / 100

    # Hour of the year
    DAY = [0 for x in range(len(date))]
    for h in range(len(date)):
        DAY[h] = date[h].dayofyear * 24 + date[h].hour + date[h].minute / 100

    # Day of the year
    yearday = [0 for x in range(len(date))]
    for h in range(len(date)):
        yearday[h] = date[h].timetuple().tm_yday

    # Day of the week
    WEEK = [0 for x in range(len(date))]
    for h in range(len(date)):
        WEEK[h] = date[h].weekday()

    #############################################################################
    DATA_C = pd.concat([DATA, pd.DataFrame(HOUR).set_index(DATA.index), pd.DataFrame(WEEK).set_index(DATA.index),
                        pd.DataFrame(yearday).set_index(DATA.index), pd.DataFrame(yearday).set_index(DATA.index)],
                       axis=1, ignore_index=True)

    temporal = ['hour', 'week', 'day', 'yearday']
    names2 = np.concatenate([names, temporal])
    DATA_C.columns = names2
    DATA_C = DATA_C.drop(range(21486), axis=0)
    DATA_C = DATA_C.reset_index(drop=True)
    DATA_C = DATA_C.drop(DATA_C.shape[0] - 1, axis=0)
    DATA_C = DATA_C.reset_index(drop=True)

    a = np.array(DATA_C)
    names = DATA_C.columns
    for t in np.array([3, 9, 12, 15, 18, 21, 24, 27, 30]):
        aa = np.where(np.array(a[:, t], dtype=float) > 800)[0]
        if len(aa) > 0:
            a[aa, t] = np.repeat(np.nan, len(aa))
        aa = np.where(np.array(a[:, t], dtype=float) < 400)[0]
        if len(aa) > 0:
            a[aa, t] = np.repeat(400, len(aa))
    DATA_C = pd.DataFrame(a)
    DATA_C.columns = names
    DATA_C = DATA_C.reset_index(drop=True)
    names = DATA_C.columns
    D = np.array(DATA_C)
    for i in range(DATA_C.shape[1]):
        print(i)
        a = D[:, i]
        N = np.where(np.array(a) == 9999)[0]
        if len(N) >= 1:
            D[N, i] = np.repeat(np.nan, len(N))

    DATA_C = pd.DataFrame(D)
    DATA_C.columns = names
    DATA_C = DATA_C.drop(range(5804))
    DATA_C = DATA_C.reset_index(drop=True)
    dd = pd.DataFrame(np.array(DATA_C, dtype=float))
    names = DATA_C.columns
    dd = dd.interpolate(method='linear', limit_direction='forward')

    for t in range(DATA_C.shape[1]):
        if any(t == np.array([4, 5, 6, 34, 35, 36])):
            DATA_C = DATA_C.reset_index(drop=True)
            dd = dd.reset_index(drop=True)
        elif any(t == np.array([0, 1, 2, 3])):
            a = DATA_C.iloc[:, t]
            a2 = dd.iloc[:, t]
            y_smooth = a.rolling(window=5, min_periods=4)
            y_s = a2.rolling(window=5, min_periods=4)
            y_smooth = y_smooth.mean()
            y_s = y_s.mean()
            DATA_C.iloc[:, t] = y_smooth
            dd.iloc[:, t] = y_s
        else:
            a = DATA_C.iloc[:, t]
            a2 = dd.iloc[:, t]
            y_smooth = a.rolling(window=5, min_periods=4)
            y_s = a2.rolling(window=5, min_periods=4)
            y_smooth = y_smooth.mean()
            y_s = y_s.mean()
            DATA_C.iloc[:, t] = y_smooth
            dd.iloc[:, t] = y_s

    DATA_C = DATA_C.drop(range(4))
    DATA_C = DATA_C.reset_index(drop=True)
    dd = dd.drop(range(4))
    dd = dd.reset_index(drop=True)
    dd.columns = names
    DATA_C = ts(DATA_C, 1, range(7, 31), DATA_C.shape[0], DATA_C.columns)
    DATA_C = DATA_C.reset_index(drop=True)
    dd = ts(dd, 1, range(7, 31), dd.shape[0], dd.columns)
    dd = dd.reset_index(drop=True)

    yearday = DATA_C.iloc[:, 36]

    if var == 'co2':
        X = dd
        D1 = DATA_C

        carrito = X.iloc[:, range(4)]
        co2 = carrito.iloc[:, 3]
        co2 = co2.reset_index(drop=True)

        # Crreción variable altura carrito
        X.iloc[:, 6] = (X.iloc[:, 6] + 550) / 1000
        D1.iloc[:, 6] = (D1.iloc[:, 6] + 550) / 1000

        y_co2 = pd.DataFrame(co2).reset_index(drop=True)

        X = X.reset_index(drop=True)
        yearday = X.iloc[:, 36]
        posis = pd.concat([X.iloc[:, 4], X.iloc[:, 5]], axis=1)
        # tt = pd.concat([X.iloc[:, np.array([4, 7, 10, 13, 16, 19, 22, 25]) + 4]])
        tt_co2 = pd.concat([X.iloc[:, np.array([5, 8, 11, 14, 17, 20, 23, 26]) + 4]])
        pos_cajas_x = np.array([50.1, 0, 47.1, 4, 20.6, 30.6, 44, 11.5])
        pos_cajas_y = np.array([3, 16, 14.4, 0, 9.4, 15.7, 6.3, 7.8])
        pos_cajas_z = np.array([3.2, 2.4, 3.2, 3, 2.2, 2.5, 2.1, 1.9])
        pos_cajas = pd.concat([pd.DataFrame(pos_cajas_x), pd.DataFrame(pos_cajas_y)], axis=1)
        pos_cajasT = pd.concat([pd.DataFrame(pos_cajas_x), pd.DataFrame(pos_cajas_y), pd.DataFrame(pos_cajas_z)],
                               axis=1)

        #####################################################################################################################
        # CORRECIÓN
        #####################################################################################################################
        p1 = 0
        p = yearday[0]
        # average_temp = []
        average_co2 = []
        indices = []
        while p1 < len(yearday):
            indi = np.where(np.array(yearday) == p)[0]
            indices.append(len(indi))
            if np.sum(posis.iloc[indi, 0] - posis.iloc[indi, 0].iloc[0]) == 0 and np.sum(
                    posis.iloc[indi, 1] - posis.iloc[indi, 1].iloc[0]) == 0:
                pp = np.array([posis.iloc[indi, 0].iloc[0], posis.iloc[indi, 1].iloc[0]])
                diff = np.abs(pos_cajas - pp)
                diff = np.sqrt(diff.iloc[:, 0] ** 2 + diff.iloc[:, 1] ** 2)
                # diff=np.sqrt(np.array(diff[:, 0] ** 2 + diff[:, 1] ** 2).astype(np.float64))
                diff = np.array(diff)
                pond = [0 for x in range(len(diff))]
                for t in range(len(diff)):
                    pond[t] = diff[t] / np.sum(diff)
                # avr1 = [0 for x in range(len(indi))]
                avr_co2 = [0 for x in range(len(indi))]
                # tt1 = tt.iloc[indi, :]
                tt_co21 = tt_co2.iloc[indi, :]
                for t in range(len(indi)):
                    # avr1[t] = np.average(np.array(tt1.iloc[t, :], dtype=float), weights=pond)
                    avr_co2[t] = np.average(np.array(tt_co21.iloc[t, :], dtype=float), weights=pond)
                # avr1 = pd.DataFrame(avr1)
                avr_co2 = pd.DataFrame(avr_co2)
                # average_temp.append(avr1)
                average_co2.append(avr_co2)
            else:
                ii = np.unique(np.where((posis.iloc[indi, 0] == posis.iloc[indi, :].iloc[0, 0]) & (
                        posis.iloc[indi, 1] == posis.iloc[indi, :].iloc[0, 1]))[0])
                ii2 = np.unique(np.where((posis.iloc[indi, 0] != posis.iloc[indi, :].iloc[0, 0]) | (
                        posis.iloc[indi, 1] != posis.iloc[indi, :].iloc[0, 1]))[0])
                posis3 = posis.iloc[indi, :].iloc[ii]
                posis3_2 = posis.iloc[indi, :].iloc[ii2]
                pp3 = posis3.iloc[0]
                pp3_2 = posis3_2.iloc[0]
                diff3 = np.abs(pos_cajas - np.array(pp3))
                diff3_2 = np.abs(pos_cajas - np.array(pp3_2))
                diff3 = np.sqrt(np.array(diff3.iloc[:, 0] ** 2 + diff3.iloc[:, 1] ** 2, dtype=float))
                diff3_2 = np.sqrt(np.array(diff3_2.iloc[:, 0] ** 2 + diff3_2.iloc[:, 1] ** 2, dtype=float))
                diff3 = np.array(diff3)
                diff3_2 = np.array(diff3_2)
                pond3 = [0 for x in range(len(diff3))]
                pond3_2 = [0 for x in range(len(diff3_2))]
                for t in range(len(diff3)):
                    pond3[t] = diff3[t] / np.sum(diff3)
                for t in range(len(diff3_2)):
                    pond3_2[t] = diff3_2[t] / np.sum(diff3_2)
                # tt1 = tt.iloc[indi, :].reset_index(drop=True)
                tt_co21 = tt_co2.iloc[indi, :].reset_index(drop=True)
                # tt3 = tt1.iloc[ii, :]
                tt_co23 = tt_co21.iloc[ii, :]
                # tt3_2 = tt1.iloc[ii2, :]
                tt_co23_2 = tt_co21.iloc[ii2, :]
                # avr3 = [0 for x in range(len(ii))]
                # avr3_2 = [0 for x in range(len(ii2))]
                avr_co23 = [0 for x in range(len(ii))]
                avr_co23_2 = [0 for x in range(len(ii2))]
                for t in range(len(ii)):
                    # avr3[t] = np.average(np.array(tt3.iloc[t, :], dtype=float), weights=pond3)
                    avr_co23[t] = np.average(tt_co23.iloc[t, :], weights=pond3)
                for t in range(len(ii2)):
                    # avr3_2[t] = np.average(np.array(tt3_2.iloc[t, :], dtype=float), weights=pond3_2)
                    avr_co23_2[t] = np.average(np.array(tt_co23_2.iloc[t, :], dtype=float), weights=pond3_2)
                # avr = pd.concat([pd.DataFrame(avr3), pd.DataFrame(avr3_2)], axis=0)
                avr_co2 = pd.concat([pd.DataFrame(avr_co23), pd.DataFrame(avr_co23_2)], axis=0)
                # average_temp.append(avr.reset_index(drop=True))
                average_co2.append(avr_co2.reset_index(drop=True))
            p1 = p1 + len(indi)
            if p1 < len(yearday):
                p = yearday[p1]
            else:
                p = 0

        average_co2F = pd.DataFrame(np.concatenate(average_co2)).iloc[:, 0]

        st = np.where(yearday == 340)[0][0] - 1
        st2 = np.where(yearday == 359)[0][len(np.where(yearday == 359)[0]) - 1]

        st3 = np.where(yearday == 24)[0][0] - 1
        st4 = np.where(yearday == 34)[0][len(np.where(yearday == 359)[0]) - 1]

        st_F = [0 for x in range(st2 - st + 1)]
        j = 0
        for i in range(st2 - st + 1):
            st_F[i] = st + j
            j = j + 1

        st_F2 = [0 for x in range(st4 - st3 + 1)]
        j = 0
        for i in range(st4 - st3 + 1):
            st_F2[i] = st3 + j
            j = j + 1

        st = np.concatenate([np.array(st_F), np.array(st_F2)])

        yearday2 = yearday.iloc[st]
        yearday2 = yearday2.reset_index(drop=True)
        avr1C = average_co2F.drop(st, axis=0)
        avr2 = average_co2F.iloc[st]
        avr1C = avr1C.reset_index(drop=True)
        avr2 = avr2.reset_index(drop=True)
        y_co2_init = y_co2
        y_co21 = y_co2.drop(st, axis=0)
        y_co22 = y_co2.iloc[st]
        y_co21 = y_co21.reset_index(drop=True)
        y_co22 = y_co22.reset_index(drop=True)
        posis2 = posis.iloc[st]
        posis2 = posis2.reset_index(drop=True)
        ratio1 = np.mean(y_co21.iloc[:, 0] / avr1C)
        p1 = 0
        yearday2 = np.array(yearday2)
        p = yearday2[0]
        mean_co2 = []
        while p1 < len(yearday2):
            indi = np.where(np.array(yearday2) == p)[0]
            if np.sum(posis2.iloc[indi, 0] - posis2.iloc[indi, 0].iloc[0]) == 0 and np.sum(
                    posis2.iloc[indi, 1] - posis2.iloc[indi, 1].iloc[0]) == 0:
                rat = y_co22.iloc[indi, 0] / avr2.iloc[indi]
                r1 = np.mean(rat) / ratio1
                y_co22.iloc[indi, 0] = y_co22.iloc[indi, 0] / r1
            else:
                yyy = y_co22.iloc[indi, 0]
                avr22 = avr2.iloc[indi]
                ii = np.unique(np.where((posis2.iloc[indi, 0] == posis2.iloc[indi, :].iloc[0, 0]) & (
                        posis2.iloc[indi, 1] == posis2.iloc[indi, :].iloc[0, 1]))[0])
                ii2 = np.unique(np.where((posis2.iloc[indi, 0] != posis2.iloc[indi, :].iloc[0, 0]) | (
                        posis2.iloc[indi, 1] != posis2.iloc[indi, :].iloc[0, 1]))[0])
                rat3 = yyy.iloc[ii] / avr22.iloc[ii]
                r13 = np.mean(rat3) / ratio1
                yyy.iloc[ii] = yyy.iloc[ii] / r13
                rat3_2 = yyy.iloc[ii2] / avr22.iloc[ii2]
                r13_2 = np.mean(rat3_2) / ratio1
                yyy.iloc[ii2] = yyy.iloc[ii2] / r13_2
                y_co22.iloc[indi, 0] = yyy.iloc[:]
            p1 = p1 + len(indi)
            if p1 < len(yearday2):
                p = yearday2[p1]
            else:
                p = 0

        y_co2_init.iloc[st] = y_co22
        y_co2 = pd.DataFrame(y_co2_init).reset_index(drop=True)

        #####################################################################################################################
        D1.iloc[:, 3] = y_co2
        DATA_C = D1

    stop = np.where(yearday == 323)[0][0]
    DATA_C = DATA_C.drop(range(stop), axis=0)
    DATA_C = DATA_C.reset_index(drop=True)
    # dd= dd.drop(range(stop), axis=0)
    # dd= dd.reset_index(drop=True)
    yearday = yearday.drop(range(stop), axis=0)
    yearday = yearday.reset_index(drop=True)
    carrito = DATA_C.iloc[:, range(4)]
    temp_I = carrito.iloc[:, 1]
    hum_I = carrito.iloc[:, 0]
    co2_I = carrito.iloc[:, 3]
    press_I = carrito.iloc[:, 2]

    DATA_C.iloc[:, 6] = (DATA_C.iloc[:, 6] + 550) / 1000
    x_train = DATA_C
    x_train = x_train.drop(np.array([x_train.shape[0] - 2, x_train.shape[0] - 1]), axis=0)
    x_train = x_train.reset_index(drop=True)
    temp_I = temp_I.drop(np.array([temp_I.shape[0] - 2, temp_I.shape[0] - 1]), axis=0)
    temp_I = temp_I.reset_index(drop=True)
    hum_I = hum_I.drop(np.array([hum_I.shape[0] - 2, hum_I.shape[0] - 1]), axis=0)
    hum_I = hum_I.reset_index(drop=True)
    co2_I = co2_I.drop(np.array([co2_I.shape[0] - 2, co2_I.shape[0] - 1]), axis=0)
    co2_I = co2_I.reset_index(drop=True)

    x_train = x_train.drop(x_train.columns[range(4)], axis=1)
    names = np.delete(names, np.array([0, 1, 2, 3]))

    ############################################################################################
    # CORRECIÃ“N DATOS CARRITO
    X = x_train.reset_index(drop=True)
    yearday = X.iloc[:, 32]
    posis = pd.concat([X.iloc[:, 0], X.iloc[:, 1]], axis=1)
    tt = pd.concat([X.iloc[:, np.array([4, 7, 10, 13, 16, 19, 22, 25])]])
    tt_co2 = pd.concat([X.iloc[:, np.array([5, 8, 11, 14, 17, 20, 23, 26])]])

    pos_cajas_x = np.array([50.1, 0, 47.1, 4, 20.6, 30.6, 44, 11.5])
    pos_cajas_y = np.array([3, 16, 14.4, 0, 9.4, 15.7, 6.3, 7.8])
    pos_cajas_z = np.array([3.2, 2.4, 3.2, 3, 2.2, 2.5, 2.1, 1.9])
    pos_cajas = pd.concat([pd.DataFrame(pos_cajas_x), pd.DataFrame(pos_cajas_y)], axis=1)
    pos_cajasT = pd.concat([pd.DataFrame(pos_cajas_x), pd.DataFrame(pos_cajas_y), pd.DataFrame(pos_cajas_z)], axis=1)

    X_final = x_train
    X_final = X_final.reset_index(drop=True)

    dayofyear = X_final.iloc[:, 32]
    # dayofyear_F = x_test.iloc[:, 32]
    X_final = X_final.drop(X_final.columns[33], axis=1)
    # x_test = x_test.drop(x_test.columns[31], axis=1)

    # distances to fixed devices
    posit = X_final.iloc[:, range(3)]
    # posit_test = x_test.iloc[:, range(3)]

    distanceX_02 = pd.DataFrame(pos_cajasT.iloc[0, 0] - posit.iloc[:, 0])
    distanceY_02 = pd.DataFrame(pos_cajasT.iloc[0, 1] - posit.iloc[:, 1])
    distanceZ_02 = pd.DataFrame(pos_cajasT.iloc[0, 2] - posit.iloc[:, 2])
    distanceX_09 = pd.DataFrame(pos_cajasT.iloc[1, 0] - posit.iloc[:, 0])
    distanceY_09 = pd.DataFrame(pos_cajasT.iloc[1, 1] - posit.iloc[:, 1])
    distanceZ_09 = pd.DataFrame(pos_cajasT.iloc[1, 2] - posit.iloc[:, 2])
    distanceX_1a = pd.DataFrame(pos_cajasT.iloc[2, 0] - posit.iloc[:, 0])
    distanceY_1a = pd.DataFrame(pos_cajasT.iloc[2, 1] - posit.iloc[:, 1])
    distanceZ_1a = pd.DataFrame(pos_cajasT.iloc[2, 2] - posit.iloc[:, 2])
    distanceX_3d = pd.DataFrame(pos_cajasT.iloc[3, 0] - posit.iloc[:, 0])
    distanceY_3d = pd.DataFrame(pos_cajasT.iloc[3, 1] - posit.iloc[:, 1])
    distanceZ_3d = pd.DataFrame(pos_cajasT.iloc[3, 2] - posit.iloc[:, 2])
    distanceX_B1 = pd.DataFrame(pos_cajasT.iloc[4, 0] - posit.iloc[:, 0])
    distanceY_B1 = pd.DataFrame(pos_cajasT.iloc[4, 1] - posit.iloc[:, 1])
    distanceZ_B1 = pd.DataFrame(pos_cajasT.iloc[4, 2] - posit.iloc[:, 2])
    distanceX_B2 = pd.DataFrame(pos_cajasT.iloc[5, 0] - posit.iloc[:, 0])
    distanceY_B2 = pd.DataFrame(pos_cajasT.iloc[5, 1] - posit.iloc[:, 1])
    distanceZ_B2 = pd.DataFrame(pos_cajasT.iloc[5, 2] - posit.iloc[:, 2])
    distanceX_B3 = pd.DataFrame(pos_cajasT.iloc[6, 0] - posit.iloc[:, 0])
    distanceY_B3 = pd.DataFrame(pos_cajasT.iloc[6, 1] - posit.iloc[:, 1])
    distanceZ_B3 = pd.DataFrame(pos_cajasT.iloc[6, 2] - posit.iloc[:, 2])
    distanceX_B4 = pd.DataFrame(pos_cajasT.iloc[7, 0] - posit.iloc[:, 0])
    distanceY_B4 = pd.DataFrame(pos_cajasT.iloc[7, 1] - posit.iloc[:, 1])
    distanceZ_B4 = pd.DataFrame(pos_cajasT.iloc[7, 2] - posit.iloc[:, 2])

    names = X_final.columns
    X_final = pd.concat(
        [X_final, distanceX_02, distanceX_09, distanceX_1a, distanceX_3d, distanceX_B1, distanceX_B2, distanceX_B3,
         distanceX_B4,
         distanceY_02, distanceY_09, distanceY_1a, distanceY_3d, distanceY_B1, distanceY_B2, distanceY_B3,
         distanceY_B4,
         distanceZ_02, distanceZ_09, distanceZ_1a, distanceZ_3d, distanceZ_B1, distanceZ_B2, distanceZ_B3,
         distanceZ_B4], axis=1)

    nn = ["distanceX_02", "distanceX_09", "distanceX_1a", "distanceX_3d", "distanceX_B1", "distanceX_B2",
          "distanceX_B3", "distanceX_B4",
          "distanceY_02", "distanceY_09", "distanceY_1a", "distanceY_3d", "distanceY_B1", "distanceY_B2",
          "distanceY_B3", "distanceY_B4",
          "distanceZ_02", "distanceZ_09", "distanceZ_1a", "distanceZ_3d", "distanceZ_B1", "distanceZ_B2",
          "distanceZ_B3", "distanceZ_B4"]
    names = np.concatenate([names, nn])
    X_final.columns = names

    ##########################################################################
    yy1 = np.where((X_final.iloc[:, 0] == 6.9) | (X_final.iloc[:, 0] == 26))[0]
    yy2 = np.where((X_final.iloc[:, 1] == 4) | (X_final.iloc[:, 1] == 14.55))[0]

    yy3 = np.where((X_final.iloc[:, 0] == 46.3) | (X_final.iloc[:, 0] == 28.8))[0]
    yy4 = np.where((X_final.iloc[:, 1] == 7.6) | (X_final.iloc[:, 1] == 10.1))[0]

    zz1 = np.intersect1d(yy1, yy2)
    zz2 = np.intersect1d(yy3, yy4)
    zz1 = np.sort(np.concatenate((zz1, zz2)))

    temp_I = temp_I.drop(zz1, axis=0)
    temp_I = temp_I.reset_index(drop=True)
    hum_I = hum_I.drop(zz1, axis=0)
    hum_I = hum_I.reset_index(drop=True)
    co2_I = co2_I.drop(zz1, axis=0)
    co2_I = co2_I.reset_index(drop=True)

    x_train = X_final.drop(zz1, axis=0)
    x_train = x_train.reset_index(drop=True)

    med_temp = np.nanmean(temp_I)
    med_hum = np.nanmean(hum_I)
    med_co2 = np.nanmean(co2_I)

    x_train = x_train.drop(['pos_z', 'pos_x', 'pos_y'], axis=1)

    if var=='temp':
        Training(x_train, temp_I, ntest,rf_param_grid, 'model_temp_trained','temp', med_temp)
    elif var=='co2':
        Training(x_train, co2_I, ntest,rf_param_grid, 'model_co2_trained','co2', med_co2)
    else:
        Training(x_train, hum_I, ntest, rf_param_grid, 'model_hum_trained', 'hum', med_hum)

def position(positions):
    dat = positions.drop(['x', 'y'], axis=1)
    changes = []
    for i in range(len(dat)):
        dat1 = dat.iloc[i]
        dd = datetime.datetime(dat1[0], dat1[1], dat1[2], dat1[3], dat1[4], dat1[5])
        changes.append(dd)

    pos_x = [0 for x in range(len(variables_02[0]))]
    pos_y = [0 for x in range(len(variables_02[0]))]
    j=0
    for h in range(1,len(changes)):
        diff =changes[h]-changes[h-1]
        days, seconds = diff.days, diff.total_seconds()
        minutes = int(seconds/60)
        pos_x[j:(j+minutes)] = np.repeat(positions['x'][h-1], minutes)
        pos_y[j:(j+minutes)] = np.repeat(positions['y'][h-1],minutes)
        j=j+minutes

    return pos_x, pos_y


def position_meteo(var, l):
    var1 = [0 for x in range(l)]
    k=0
    for i in range(len(var)):
        vv = np.repeat(var[i], 10)
        var1[k:(k+10)]=vv
        k=k+10
    return var1

######################################################################################################################
######################################################################################################################
# Loading of data from the different hosts
variables = []
time_end = datetime.datetime(2021, 4, 26, 12, 0, 0)

#Data of the moving cart
# 0x6a52
pos_z= loading_carrito(['vert'], 'vertpantilt', '0x6a52', variables, time_end)
variables=[]
variables_52 = loading_carrito(['humidity', 'temperature'], 'sht31d', '0x6a52', variables, time_end)
variables_52 = loading_carrito(['pressure'], 'bme680_bsec', '0x6a52', variables_52, time_end)
variables_52 = loading_carrito(['co2'], 'mhz14', '0x6a52', variables_52, time_end)
names_52 = ['humidity_C','temperature_C','pressure_C','co2_C']

#Data of the boxes
# 0x6a02
variables=[]
variables_02 = loading(['humidity', 'temperature'], 'sht31d', '0x6a02', variables, time_end)
#variables_02 = loading(['lux'], 'tsl2561', '0x6a02', variables_02, time_end)
variables_02 = loading(['co2'], 'mhz14', '0x6a02', variables_02, time_end)
names_02 = ['humidity_02','temperature_02','co2_02']
# 0x6a09
variables = []
variables_09 = loading(['humidity', 'temperature'], 'sht31d', '0x6a09', variables, time_end)
variables_09 = loading(['co2'], 'mhz14', '0x6a09', variables_09, time_end)
#variables_09 = loading(['lux'], 'tsl2561', '0x6a09', variables_09, time_end)
names_09 = ['humidity_09','temperature_09','co2_09']
# 0x6a1a
variables = []
variables_1a = loading(['humidity', 'temperature'], 'sht31d', '0x6a1a', variables, time_end)
#variables_1a = loading(['lux'], 'tsl2561', '0x6a1a', variables_1a, time_end)
variables_1a = loading(['co2'], 'mhz14', '0x6a1a', variables_1a, time_end)
names_1a = ['humidity_1a','temperature_1a','co2_1a']
# 0x6a3d
variables = []
variables_3d = loading(['humidity', 'temperature'], 'sht31d', '0x6a3d', variables, time_end)
variables_3d = loading(['co2'], 'mhz14', '0x6a3d', variables_3d, time_end)
#variables_3d = loading(['lux'], 'tsl2561', '0x6a3d', variables_3d, time_end)
names_3d = ['humidity_3d','temperature_3d','co2_3d']
# rpiB1
variables = []
variables_B1 = loading(['humidity', 'temperature'], 'sht31d', 'rpiB1', variables, time_end)
#variables_B1 = loading(['lux'], 'tsl2591', 'rpiB1', variables_B1, time_end)
variables_B1 = loading(['co2'], 'mhz14', 'rpiB1', variables_B1, time_end)
names_B1 = ['humidity_B1','temperature_B1','co2_B1']
# rpiB2
variables = []
variables_B2 = loading(['humidity', 'temperature'], 'sht31d', 'rpiB2', variables, time_end)
#variables_B2 = loading(['lux'], 'tsl2591', 'rpiB2', variables_B2, time_end)
variables_B2 = loading(['co2'], 'mhz14', 'rpiB2', variables_B2, time_end)
names_B2 = ['humidity_B2','temperature_B2','co2_B2']
# rpiB3
variables = []
variables_B3 = loading(['humidity', 'temperature'], 'sht31d', 'rpiB3', variables, time_end)
#variables_B3 = loading(['lux'], 'tsl2591', 'rpiB3', variables_B3, time_end)
variables_B3 = loading(['co2'], 'mhz14', 'rpiB3', variables_B3, time_end)
names_B3 = ['humidity_B3','temperature_B3','co2_B3']
# rpiB4
variables = []
variables_B4 = loading(['humidity', 'temperature'], 'sht31d', 'rpiB4', variables, time_end)
#variables_B4 = loading(['lux'], 'tsl2591', 'rpiB4', variables_B4, time_end)
variables_B4 = loading(['co2'], 'mhz14', 'rpiB4', variables_B4, time_end)
names_B4 = ['humidity_B4','temperature_B4','co2_B4']

#Meteo data
variables = []
variables_meteo = loading(['humidity','radiation','temperature'], 'meteo', 'none', variables, time_end)
names_meteo=['humidity_M','radiation_M','temperature_M']
var_meteo=[]
for u in range(len(variables_meteo)):
    var_meteo.append(position_meteo(variables_meteo[u],len(variables_B4[0])))

#Cart positions
positions = pd.read_csv("positions_digital.csv", sep=";", decimal=",")
pos_x, pos_y= position(positions)

car = pd.DataFrame(np.array(variables_52).transpose())
z = pd.DataFrame(pos_z[0])
car=car.reset_index(drop=True)

#DATA JOIN
variables = pd.concat([car,pd.DataFrame(pos_x).set_index(car.index),pd.DataFrame(pos_y).set_index(car.index),z.set_index(car.index), pd.DataFrame(np.array(variables_02).transpose()).set_index(car.index),
                       pd.DataFrame(np.array(variables_09).transpose()).set_index(car.index),pd.DataFrame(np.array(variables_1a).transpose()).set_index(car.index),
                       pd.DataFrame(np.array(variables_3d).transpose()).set_index(car.index),pd.DataFrame(np.array(variables_B1).transpose()).set_index(car.index), pd.DataFrame(np.array(variables_B2).transpose()).set_index(car.index),
                       pd.DataFrame(np.array(variables_B3).transpose()).set_index(car.index),pd.DataFrame(np.array(variables_B4).transpose()).set_index(car.index),pd.DataFrame(np.array(var_meteo).transpose()).set_index(car.index)], axis=1)

pos=['pos_x','pos_y','pos_z']
variables.columns = np.concatenate([names_52, pos,names_02, names_09, names_1a, names_3d,names_B1,names_B2,names_B3,names_B4,names_meteo])
names = variables.columns
#Parameters definition
date_init= '2020-06-17 12:00:00'
ntest=1440
rf_param_grid = {
    'n_estimators': [50,100,200],
    'max_depth': [50,100,200],
    'min_samples_split':[50,100,200]
}
#General function to train and save the models
MTI_train(variables, date_init, ntest, rf_param_grid,'temp',names)
MTI_train(variables, date_init, ntest, rf_param_grid,'co2',names)
MTI_train(variables, date_init, ntest, rf_param_grid,'hum',names)