import os
import random
import sys

# Third party imports:
import numpy

# Third party imports (with specific controls for TensorFlow):
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow

tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Set constants:
# if len(sys.argv) > 1:
#     GLOBAL_SEED = sys.argv[1]
# else:
#     GLOBAL_SEED = 1
GLOBAL_SEED = 1

# Set random seeds:
os.environ['PYTHONHASHSEED'] = str(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
numpy.random.seed(GLOBAL_SEED)
tensorflow.compat.v1.set_random_seed(GLOBAL_SEED)

# Configure a new global session for TensorFlow:
session_conf = tensorflow.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tensorflow.compat.v1.Session(
    graph=tensorflow.compat.v1.get_default_graph(), config=session_conf)
tensorflow.compat.v1.keras.backend.set_session(sess)

import warnings
import pandas as pd
import numpy as np
from scipy import interpolate
import joblib
from random import randint
from datetime import datetime
from pymoo.model.problem import FunctionalProblem
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from time import time
# from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import seaborn as sns;

sns.set()
from tensorflow.compat.v1.keras.callbacks import EarlyStopping
# from sklearn.ensemble import RandomForestRegressor
from tensorflow.compat.v1.keras.callbacks import ModelCheckpoint
import multiprocessing
from sklearn import svm
from sklearn import metrics
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from random import randint
from pymoo.model.repair import Repair

from pathlib import Path
from pymoo.util.termination.f_tol import MultiObjectiveSpaceToleranceTermination
import matplotlib.pyplot as plt
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

import numpy as np
# import os, psutil
from pymoo.util.misc import stack
from pymoo.model.problem import Problem
# from pymoo.util.misc import covert_to_type
from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
from sklearn.metrics import r2_score
from pymoo.factory import get_problem, get_visualization, get_decomposition

# import xgboost as xgb
from random import randint
from sklearn.metrics import r2_score
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from tensorflow.compat.v1.keras.models import Sequential
from tensorflow.compat.v1.keras.layers import Dense
from tensorflow.compat.v1.keras.models import model_from_json
import pickle
from pickle import dump
import influxdb
import datetime
from time import time

t = time()


# FUNCTIONS
def MBE(y_true, y_pred, med):
    '''
    Parameters:
        y_true (array): Array of observed values
        y_pred (array): Array of prediction values

    Returns:
        mbe (float): Biais score
    '''
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true = y_true.reshape(len(y_true), 1)
    y_pred = y_pred.reshape(len(y_pred), 1)
    # mbe = np.sum(y_true-y_pred)/np.sum(y_true)
    mbe = np.mean(y_true - y_pred) / med
    # mbe = mbe/np.mean(y_true)
    # print('MBE = ', mbe)
    return (mbe * 100)


def complex(neurons1, neurons2, neurons3, max_N, max_H):
    if neurons1 > 0 and neurons2 == 0 and neurons3 == 0:
        u = 1
        W = neurons1
    elif neurons1 > 0 and neurons2 > 0 and neurons3 == 0:
        u = 2
        W = np.array([neurons1, neurons2])
    elif neurons1 > 0 and neurons2 > 0 and neurons3 > 0:
        u = 3
        W = np.array([neurons1, neurons2, neurons3])

    #F = 0.5*(u / max_H) + np.sum(W / max_N)
    F = 0.25 *(u/max_H) + 0.75*np.sum((neurons1, neurons2,neurons3))/max_N

    return F


def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))


def cortes(x, D, zc):
    Y = np.zeros((zc, round(D / zc)))
    i = 0
    s = 0
    while i <= D:
        if D - i < zc and D - i > 0:
            # Y = np.delete(Y,s,1)
            Y = np.delete(Y, Y.shape[1] - 1, 1)
            break
        elif D - i == 0:
            break
        else:
            Y[:, s] = x[i:(i + zc)][:, 0]
            i = i + zc
            s = s + 1
    return (Y)


def ts(new_data, look_back, pred_col, dim, names):
    t = new_data.copy()
    t['id'] = range(0, len(t))
    # t = t.iloc[:-look_back, :]
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


def cortes(x, D, zc):
    Y = np.zeros((zc, round(D / zc)))
    i = 0
    s = 0
    while i <= D:
        if D - i < zc and D - i > 0:
            # Y = np.delete(Y,s,1)
            Y = np.delete(Y, Y.shape[1] - 1, 1)
            break
        elif D - i == 0:
            break
        else:
            Y[:, s] = x[i:(i + zc)]
            i = i + zc
            s = s + 1
    return (Y)


def means(X, sep):
    LISTA = []
    for i in range(X.shape[1]):
        data = X.iloc[:, i]
        dat = cortes(data, len(data), sep)
        dat = np.nanmean(dat, axis=0)
        LISTA.append(pd.Series(dat))
    L = pd.DataFrame(LISTA).T
    return (L)


######################################################################################################################
def loading(var_name, sensor_name, host, variables, time_end):
#info influx

    influx = influxdb.InfluxDBClient.from_dsn(CONN_STR)
    influx.switch_database(DB_NAME)

    place = ["sensor_data.autogen", sensor_name]
    sep = '.'
    place = sep.join(place)
    place2 = [sensor_name, "address"]
    sep = '.'
    place2 = sep.join(place2)
    time_end_str = time_end.strftime('%Y-%m-%dT%H:%M:%SZ')

    if host == 'none':
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
            # dates = [0 for x in range(len(point))]
            for t in range(len(point)):
                values[t] = point[t][var_name[u]]
                # dates[t] = point[t]['time']

            variables.append(values)
            # dates = pd.to_datetime(pd.Series(dates), format='%Y-%m-%dT%H:%M:%SZ')

    return variables


######################################################################################################################
def loading_carrito(var_name, sensor_name, host, variables, time_end):
#info influx

    influx = influxdb.InfluxDBClient.from_dsn(CONN_STR)
    influx.switch_database(DB_NAME)

    place = ["sensor_data.autogen", sensor_name]
    sep = '.'
    place = sep.join(place)
    place2 = [sensor_name, "address"]
    sep = '.'
    place2 = sep.join(place2)
    time_end_str = time_end.strftime('%Y-%m-%dT%H:%M:%SZ')

    if host == 'none':
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
            # dates = [0 for x in range(len(point))]
            for t in range(len(point)):
                values[t] = point[t][var_name[u]]
                # dates[t] = point[t]['time']

            variables.append(values)

            # dates = pd.to_datetime(pd.Series(dates), format='%Y-%m-%dT%H:%M:%SZ')
            # dates = dates.dt.floor('Min')

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
            # dates = [0 for x in range(len(point))]
            for t in range(len(point)):
                values[t] = point[t][var_name[u]]
                # dates[t] = point[t]['time']

            variables.append(values)

            # dates= pd.to_datetime(pd.Series(dates), format='%Y-%m-%dT%H:%M:%SZ')
            # dates = dates.dt.floor('Min')

    return variables


def nas_function(x):
    for j in range(x.shape[0]):
        # Possibility of missing values
        if any(np.array(np.isnan(x.iloc[j, :]))):
            ii = np.where(np.isnan(x.iloc[j, :]))[0]
            x.iloc[j, ii] = 0
    return (x)


def ANN2(X, y, med, neurons1, neurons2, neurons3):
    name1 = tuple([neurons1, neurons2, neurons3])
    try:
        a0, a1 = dictionary[name1]
        return a0, a1

    except KeyError:
        pass

    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    errors = [0 for x in range(3)]
    complexity = [0 for x in range(3)]
    possX = [9.8, 40.9, 17.6]
    possY = [7.8, 11.6, 1.6]
    for j in range(3):
        yy1 = np.where(X.iloc[:, 0] == possX[j])[0]
        yy2 = np.where(X.iloc[:, 1] == possY[j])[0]
        zz1 = np.intersect1d(yy1, yy2)
        X_test1 = X.iloc[zz1]
        X_test1 = X_test1.reset_index(drop=True)
        y_test1 = y.iloc[zz1]
        y_test1 = y_test1.reset_index(drop=True)
        X_final = X.drop(zz1, axis=0)
        X_finaL = X_final.reset_index(drop=True)
        y1 = y.drop(zz1, axis=0)
        y1 = y1.reset_index(drop=True)

        X_final = X_final.drop(X_final.columns[np.array([0, 1, 2])], axis=1)
        X_test1= X_test1.drop(X_test1.columns[np.array([0, 1, 2])], axis=1)

        temperaturas_train = np.array(X_final.iloc[:, np.array([1, 4, 7, 10, 13, 16, 19, 22, 26, 31, 34, 37, 40, 43, 46, 49, 52])])
        temperaturas_test = np.array(X_test1.iloc[:, np.array([1, 4, 7, 10, 13, 16, 19, 22, 26, 31, 34, 37, 40, 43, 46, 49, 52])])

        humedad_train = np.array(X_final.iloc[:, np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 30, 33, 36, 39, 42, 45, 48, 51])])
        humedad_test = np.array(X_test1.iloc[:, np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 30, 33, 36, 39, 42, 45, 48, 51])])

        co2s_train = np.array(X_final.iloc[:, np.array([2, 5, 8, 11, 14, 17, 20, 23, 32, 35, 38, 41, 44, 47, 50, 53])])
        co2s_test = np.array(X_test1.iloc[:, np.array([2, 5, 8, 11, 14, 17, 20, 23, 32, 35, 38, 41, 44, 47, 50, 53])])

        diss_train = np.array(X_final.iloc[:, range(X_final.shape[1] - 24, X_final.shape[1])])
        diss_test = np.array(X_test1.iloc[:, range(X_final.shape[1] - 24, X_final.shape[1])])

        rad_train = np.array(X_final.iloc[:, np.array([25])])
        rad_test = np.array(X_test1.iloc[:, np.array([25])])

        resto_train = X_final.iloc[:, np.array([27, 28, 29])]
        resto_test = X_test1.iloc[:, np.array([27, 28, 29])]

        scalar_temp = MinMaxScaler(feature_range=(-1,1))
        scalar_hum = MinMaxScaler(feature_range=(-1,1))
        scalar_co2 = MinMaxScaler(feature_range=(-1,1))
        scalardist = MinMaxScaler(feature_range=(-1,1))
        scalar_rad = MinMaxScaler(feature_range=(-1,1))
        scalarresto = MinMaxScaler(feature_range=(-1,1))


        scalar_temp.fit(np.concatenate((np.concatenate(temperaturas_train), np.concatenate(temperaturas_test),np.array(y1)[:,0], np.array(y_test1)[:,0])).reshape(-1, 1))
        scalardist.fit(np.concatenate((np.concatenate(diss_train), np.concatenate(diss_test))).reshape(-1, 1))
        scalar_hum.fit(np.concatenate((np.concatenate(humedad_train), np.concatenate(humedad_test))).reshape(-1, 1))
        scalar_co2.fit(np.concatenate((np.concatenate(co2s_train), np.concatenate(co2s_test))).reshape(-1, 1))
        scalar_rad.fit(np.concatenate((np.concatenate(rad_train), np.concatenate(rad_test))).reshape(-1, 1))
        scalarresto.fit(pd.concat([resto_train, resto_test], axis=0))
        y_scaled = pd.DataFrame(scalar_temp.transform(np.array(y1).reshape(-1, 1)))
        y_test1 = pd.DataFrame(scalar_temp.transform(np.array(y_test1).reshape(-1, 1)))

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

        X_scaled = pd.concat([temperaturas_train1, rad_train1, resto_train1, diss_train1], axis=1)
        X_test1 = pd.concat([temperaturas_test1, rad_test1, resto_test1, diss_test1], axis=1)


        X_scaled.columns = range(X_scaled.shape[1])
        X_test1.columns = range(X_test1.shape[1])

        for t in range(X_scaled.shape[1]):
            a = X_scaled.iloc[:, t]
            if len(np.where(np.isnan(a))[0]) > 0:
                a[np.where(np.isnan(a))[0]] = np.repeat(-10, len(np.where(np.isnan(a))[0]))
                X_scaled.iloc[:, t] = a
        for t in range(X_test1.shape[1]):
            a = X_test1.iloc[:, t]
            if len(np.where(np.isnan(a))[0]) > 0:
                a[np.where(np.isnan(a))[0]] = np.repeat(-10, len(np.where(np.isnan(a))[0]))
                X_test1.iloc[:, t] = a

        out_train = np.where(np.isnan(y_scaled))[0]
        if len(out_train) > 0:
            y_scaled = y_scaled.drop(out_train, axis=0)
            y_scaled = y_scaled.reset_index(drop=True)
            X_scaled = X_scaled.drop(out_train, axis=0)
            X_scaled = X_scaled.reset_index(drop=True)

        out_T = np.where(np.isnan(y_test1))[0]
        if len(out_T) > 0:
            y_test1 = y_test1.drop(out_T, axis=0)
            y_test1 = y_test1.reset_index(drop=True)
            X_test1 = X_test1.drop(out_T, axis=0)
            X_test1 = X_test1.reset_index(drop=True)

        h_path = Path('./best_models')
        h_path.mkdir(exist_ok=True)
        h = h_path / f'best_{randint(0, 1000000)}_model.h9'

        ####################################################################################################################
        ####################################################################################################################
        ####################################################################################################################
        ####################################################################################################################
        # if :
        if neurons1 > 0 and neurons2 == 0 and neurons3 == 0:
            ANN_model = Sequential()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                # The Input Layer :
                ANN_model.add(
                    Dense(X_scaled.shape[1], kernel_initializer='normal', input_dim=X_scaled.shape[1],
                          activation='relu'))
                # The Hidden Layers :
                ANN_model.add(Dense(neurons1, kernel_initializer='normal', activation='relu'))
                # The Output Layer :
                ANN_model.add(Dense(1, kernel_initializer='normal', activation='linear'))
                # Compile the network :
                ANN_model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])
                # ANN_model.summary()
                # Checkpoitn callback
                #es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)
                #mc = ModelCheckpoint(str(h), monitor='val_loss', mode='min', verbose=0, save_best_only=True)
                # Train the model
                # time_start = time()
                #ANN_model.fit(X_scaled, y_scaled, epochs=30, validation_data=(X_test1, y_test1), callbacks=[es, mc])
                ANN_model.fit(X_scaled, y_scaled, epochs=30)
                # time1 = round(time() - time_start, 3)
                y_pred = ANN_model.predict(pd.DataFrame(X_test1))
                y_pred = scalar_temp.inverse_transform(y_pred)
                y_real = scalar_temp.inverse_transform(y_test1)
                accuracy = np.sqrt(metrics.mean_squared_error(y_real, y_pred)) / med

                compx = complex(neurons1, neurons2, neurons3, 600, 3)
        elif neurons1 > 0 and neurons2 > 0 and neurons3 == 0:
            ANN_model = Sequential()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                # The Input Layer :
                ANN_model.add(
                    Dense(X_scaled.shape[1], kernel_initializer='normal', input_dim=X_scaled.shape[1],
                          activation='relu'))
                # The Hidden Layers :
                ANN_model.add(Dense(neurons1, kernel_initializer='normal', activation='relu'))
                ANN_model.add(Dense(neurons2, kernel_initializer='normal', activation='relu'))
                # The Output Layer :
                ANN_model.add(Dense(1, kernel_initializer='normal', activation='linear'))
                # Compile the network :
                ANN_model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])
                # ANN_model.summary()
                # Checkpoitn callback
                #es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)
                #mc = ModelCheckpoint(str(h), monitor='val_loss', mode='min', verbose=0, save_best_only=True)
                # Train the model
                # time_start = time()
                #ANN_model.fit(X_scaled, y_scaled, epochs=30, validation_data=(X_test1, y_test1), callbacks=[es, mc]
                ANN_model.fit(X_scaled, y_scaled, epochs=30)
                # time1 = round(time() - time_start, 3)
                y_pred = ANN_model.predict(pd.DataFrame(X_test1))
                y_pred = scalar_temp.inverse_transform(y_pred)
                y_real = scalar_temp.inverse_transform(y_test1)
                accuracy = np.sqrt(metrics.mean_squared_error(y_real, y_pred)) / med
                compx = complex(neurons1, neurons2, neurons3, 600, 3)
        elif neurons1 > 0 and neurons2 > 0 and neurons3 > 0:
            ANN_model = Sequential()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                # The Input Layer :
                ANN_model.add(
                    Dense(X_scaled.shape[1], kernel_initializer='normal', input_dim=X_scaled.shape[1],
                          activation='relu'))
                # The Hidden Layers :
                ANN_model.add(Dense(neurons1, kernel_initializer='normal', activation='relu'))
                ANN_model.add(Dense(neurons2, kernel_initializer='normal', activation='relu'))
                ANN_model.add(Dense(neurons3, kernel_initializer='normal', activation='relu'))
                # The Output Layer :
                ANN_model.add(Dense(1, kernel_initializer='normal', activation='linear'))
                # Compile the network :
                ANN_model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])
                # ANN_model.summary()
                # Checkpoitn callback
                #es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)
                #mc = ModelCheckpoint(str(h), monitor='val_loss', mode='min', verbose=0, save_best_only=True)
                # Train the model
                # time_start = time()
                #ANN_model.fit(X_scaled, y_scaled, epochs=30, validation_data=(X_test1, y_test1), callbacks=[es, mc])
                ANN_model.fit(X_scaled, y_scaled, epochs=30)
                # time1 = round(time() - time_start, 3)
                y_pred = ANN_model.predict(pd.DataFrame(X_test1))
                y_pred = scalar_temp.inverse_transform(y_pred)
                y_real = scalar_temp.inverse_transform(y_test1)
                accuracy = np.sqrt(metrics.mean_squared_error(y_real, y_pred)) / med
                compx = complex(neurons1, neurons2, neurons3, 600, 3)
        elif neurons1 > 0 and neurons2 == 0 and neurons3 > 0:
            accuracy = 10000
            compx = 100000
            # time1 = 100000000
        errors[j] = accuracy
        complexity[j] = compx
    ##################################################################################################
    # y_pred = ANN_model.predict(pd.DataFrame(X_test))
    # y_pred = scalarY.inverse_transform(y_pred)
    # y_real = scalarY.inverse_transform(y_test)
    # accuracy= np.sqrt(metrics.mean_squared_error(y_real, y_pred)) / np.mean(y_real)*100
    # nmbe= MBE(y_real, y_pred)
    # r_coeff = r2_score(y_real, y_pred)*100
    # h_file = Path('.')/h
    # h.unlink(True)

    accuracy = np.nanmean(np.array(errors, dtype=float))
    compx = np.nanmean(np.array(complexity, dtype=float))
    dictionary[name1] = accuracy, compx
    return accuracy, compx
    # return accuracy, nmbe


def bool4(x2, x3):
    if x2 == 0 and x3 > 0:
        a = 1
    else:
        a = 0

    return (a)


class MyRepair(Repair):

    def _do(self, problem, pop, **kwargs):
        for k in range(len(pop)):
            x = pop[k].X

            # x[2] = 2 - x[0]
            # bool4(x[1],x[2])
            if bool4(x[1], x[2]) == 1:
                x[2] = 0

        return pop


class MyProblem(Problem):

    def __init__(self, X1, y, med, contador, **kwargs):
        self.X1 = X1
        self.y = y
        self.med = med
        # self.dictionary = dictionary
        self.contador = contador
        # self.dict_cuenta={}
        super().__init__(n_var=3,
                         n_obj=2,
                         n_constr=1,
                         xl=np.array([1, 0, 0]),
                         xu=np.array([20, 20, 20]),
                         # xl=0,
                         # xu=1,
                         type_var=np.int,
                         elementwise_evaluation=True,
                         **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        # process = psutil.Process(os.getpid())
        # print(process.memory_info().rss)
        X1 = self.X1
        g1 = bool4(x[1], x[2])
        out["G"] = g1

        print(x)
        # self.cuenta +=1

        f1, f2 = ANN2(X1, self.y, self.med, int(x[0] * 10), int(x[1] * 10), int(x[2] * 10))
        # f1, f2 = ANN2(X1, self.y, self.X2, self.y2,self.med, int(x[0] * 10), int(x[1] * 10),int(x[2] * 10))

        print(
            '\n ############################################## \n ############################# \n ########################## EvaluaciÃ³n ',
            self.contador, '\n #########################')
        self.contador[0] += 1
        # f1,f2 = ANN2(X1,self.y,int(x[0]),int(x[1]),int(x[2]), self.ntest)
        # g2= bool4(x[3])+2*x[3]+bool12(x[2])-14
        # out["F"] = np.column_stack([f1, 100-f2])
        out["F"] = np.column_stack([f1, f2])


# class MyProblem2(Problem):
#
#    def __init__(self, X1, y, ntest):
#        self.X1= X1
#        self.y = y
#        self.ntest = ntest
#
#        super().__init__(n_var=10,
#                         n_obj=2,
#                         n_constr=2,
#                         xl=np.array([0,0,0,0,0,0,0,0,1,1]),
#                         xu=np.array([1,1,1,1,1,1,1,1,20,3]),
#                         #xl=0,
#                         #xu=1,
#                         type_var=np.int,
#                         elementwise_evaluation=True)
#
#    def _evaluate(self, x, out, *args, **kwargs):
#        X1=self.X1
#        X1.iloc[:, np.array([3,4,5,6,7])]=X1.iloc[:, np.array([3,4,5,6,7])]*x[0]
#        X1.iloc[:, np.array([8,9,10,11,12,13])]=X1.iloc[:, np.array([8,9,10,11,12,13])]*x[1]
#        X1.iloc[:, np.array([14,15,16,17,18,19])]=X1.iloc[:, np.array([14,15,16,17,18,19])]*x[2]
#        X1.iloc[:, np.array([20,21,22,23,24])]=X1.iloc[:, np.array([20,21,22,23,24])]*x[3]
#        X1.iloc[:, np.array([25,26,27,28,29])]=X1.iloc[:, np.array([25,26,27,28,29])]*x[4]
#        X1.iloc[:, np.array([30,31,32,33,34])]=X1.iloc[:, np.array([30,31,32,33,34])]*x[5]
#        X1.iloc[:, np.array([35,36,37,38,39])]=X1.iloc[:, np.array([35,36,37,38,39])]*x[6]
#        X1.iloc[:, np.array([40,41,42,43,44])]=X1.iloc[:, np.array([40,41,42,43,44])]*x[7]
#        #f1 = x[0] ** 2 + x[1] ** 2 - x[1]
#        f1,f2 = ANN(X1,self.y,int(x[8]*20), self.ntest, x[9])
#
#        g1 = np.sum(np.delete(x, [8,9]))- 6
#        g2 = - np.sum(np.delete(x, [8,9]))+2
#        out["F"] = np.column_stack([f1, f2])
#        out["G"] = np.column_stack([g1, g2])
#


# def nsga2_global(X,y, ntest):
#    problem = MyProblem(X, y, ntest)
#
#    algorithm = NSGA2(pop_size=2,sampling=get_sampling("int_random"),
#                      crossover=get_crossover("int_sbx", prob=1.0, eta=2.0),
#                      mutation=get_mutation("int_pm", eta=3.0),
#                      eliminate_duplicates=True)
#
#    res = minimize(problem,
#                   algorithm,
#                   ("n_gen", 2),
#                   verbose=True,
#                   seed=7)
#
#    weights = np.array([0.5, 0.5])
#    decomp = get_decomposition("asf")
#    I = get_decomposition("weighted-sum").do(res.F, weights).argmin()
#
#
#    return(res.X)
#
#    #res.pop.get("X"), res.pop.get("F")
#    plt.scatter(res.F, color="red")
#    plt.show()


def nsga2_individual(X, y, med, contador):
    n_proccess = 7
    pool = multiprocessing.Pool(n_proccess)
    problem = MyProblem(X, y, med, contador, parallelization=('starmap', pool.starmap))
    # problem = MyProblem(X, y, X2, y2)

    # algorithm = NSGA2(pop_size=50,sampling=get_sampling("int_random"),
    #                 crossover=get_crossover("int_sbx", prob=1.0, eta=2.0),
    #                 mutation=get_mutation("int_pm", eta=3.0),
    #                 eliminate_duplicates=True)
    algorithm = NSGA2(pop_size=50, repair=MyRepair(), eliminate_duplicates=True,
                      sampling=get_sampling("int_random"),
                      # sampling =g,
                      # crossover=0.9,
                      # mutation=0.1)
                      crossover=get_crossover("int_sbx"),
                      mutation=get_mutation("int_pm", prob=0.1))
    termination = MultiObjectiveSpaceToleranceTermination(tol=0.0025,
                                                          n_last=10, nth_gen=2, n_max_gen=None,
                                                          n_max_evals=4000)

    res = minimize(problem,
                   algorithm,
                   termination,
                   # ("n_gen", 20),
                   pf=True,
                   verbose=True,
                   seed=7)

    if res.F.shape[0] > 1:
        weights = np.array([0.75, 0.25])
        I = get_decomposition("pbi").do(res.F, weights).argmin()
        obj_T = res.F
        struct_T = res.X
        obj = res.F[I, :]
        struct = res.X[I, :]
    else:
        obj_T = res.F
        struct_T = res.X
        obj = res.F
        struct = res.X

    print(dictionary)
    pool.close()

    return (obj_T, struct_T, obj, struct, res)


def MTI_train(DATA, date_init, ntest, names, contador):
    date = pd.date_range(date_init, periods=DATA.shape[0], freq='1min')
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

    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    DATA_C = DATA_C.drop(range(21486), axis=0)
    DATA_C = DATA_C.reset_index(drop=True)
    DATA_C = DATA_C.drop(DATA_C.shape[0] - 1, axis=0)
    DATA_C = DATA_C.reset_index(drop=True)

    a = np.array(DATA_C)
    names = DATA_C.columns
    for t in np.array([3, 9, 12, 15, 18, 21, 24, 27, 30]):
        # for t in np.array([3,9, 12, 15, 18, 21, 24, 27, 30]):
        aa = np.where(np.array(a[:, t], dtype=float) > 800)[0]
        # aa2 = np.where(np.array(a[:, t], dtype=float) < 400)[0]

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
    # indices21 = np.array(189139)
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
    #dd = pd.DataFrame(np.array(DATA_C, dtype=float))
    names = DATA_C.columns
    #dd = dd.interpolate(method='linear', limit_direction='forward')

    for t in range(DATA_C.shape[1]):
        if any(t == np.array([4, 5, 6, 34, 35, 36])):
            DATA_C = DATA_C.reset_index(drop=True)
            #dd = dd.reset_index(drop=True)
        elif any(t == np.array([0, 1, 2, 3])):
            a = DATA_C.iloc[:, t]
            #a2 = dd.iloc[:, t]
            y_smooth = a.rolling(window=5, min_periods=4)
            #y_s = a2.rolling(window=5, min_periods=4)
            y_smooth = y_smooth.mean()
            #y_s = y_s.mean()
            DATA_C.iloc[:, t] = y_smooth
            #dd.iloc[:, t] = y_s
        else:
            a = DATA_C.iloc[:, t]
            #a2 = dd.iloc[:, t]
            y_smooth = a.rolling(window=5, min_periods=4)
            #y_s = a2.rolling(window=5, min_periods=4)
            y_smooth = y_smooth.mean()
            #y_s = y_s.mean()
            DATA_C.iloc[:, t] = y_smooth
            #dd.iloc[:, t] = y_s

    DATA_C = DATA_C.drop(range(4))
    DATA_C = DATA_C.reset_index(drop=True)
    #dd = dd.drop(range(4))
    #dd = dd.reset_index(drop=True)
    #dd.columns = names
    DATA_C = ts(DATA_C, 1, range(7, 31), DATA_C.shape[0], DATA_C.columns)
    DATA_C = DATA_C.reset_index(drop=True)
    #dd = ts(dd, 1, range(7, 31), dd.shape[0], dd.columns)
    #dd = dd.reset_index(drop=True)


    yearday = DATA_C.iloc[:, 36]

    stop = np.where(yearday == 323)[0][0]
    DATA_C = DATA_C.drop(range(stop), axis=0)
    DATA_C = DATA_C.reset_index(drop=True)
    # dd= dd.drop(range(stop), axis=0)
    # dd= dd.reset_index(drop=True)
    yearday = yearday.drop(range(stop), axis=0)
    yearday = yearday.reset_index(drop=True)

    #iii = -1
    #fo#r j in range(DATA_C.shape[1]):
    #    g = DATA_C.iloc[:, j]
    #    h = np.where(pd.isnull(g))[0]
    #    print(len(h))
    #    if len(h) > 0:
    #        # DATA_C= DATA_C.drop(h, axis=0)
    #        # DATA_C = DATA_C.reset_index(drop=True)
    #        iii = np.union1d(iii, h)
#
    #iii = np.delete(iii, 0)
#
    ## r=np.where(DATA_C.iloc[:,6]!=900)[0]
    ## iii=np.union1d(iii,r)
    #D1 = DATA_C
    #D1 = D1.reset_index(drop=True)
    #X = dd
    # temp_I.iloc[iii]=np.repeat(np.nan,len(iii))
    # temp_I = temp_I.iloc[oo]
    # temp_I = temp_I.reset_index(drop=True)

    # DATA_C = DATA_C.interpolate(method ='linear', limit_direction ='forward')
    # DATA_C =DATA_C.drop(np.array([0,1,2,3,4,5]), axis=0)
    # DATA_C = DATA_C.reset_index(drop=True)
    # DATA_C =DATA_C.drop(range(501))
    # DATA_C = DATA_C.reset_index(drop=True)
    #
    carrito = DATA_C.iloc[:, range(4)]
    temp_I = carrito.iloc[:, 1]
    hum_I = carrito.iloc[:, 0]
    co2_I = carrito.iloc[:, 3]
    press_I = carrito.iloc[:, 2]

    #temp = temp.reset_index(drop=True)
    #hum = hum.reset_index(drop=True)
    #co2 = co2.reset_index(drop=True)
    #press = press.reset_index(drop=True)

   #X = X.drop(X.columns[range(4)], axis=1)
   #D1 = D1.drop(D1.columns[range(4)], axis=1)
   #names = np.delete(names, np.array([0, 1, 2, 3]))

    # CorreciÃ³n variable altura carrito
    DATA_C.iloc[:, 6] = (DATA_C.iloc[:, 6] + 550) / 1000
    # X_init.iloc[:,2]=(X_init.iloc[:,2]+550)/1000
    # x_test.iloc[:, 2] = (x_test.iloc[:, 2] + 550) / 1000

    y_temp = pd.DataFrame(temp_I).reset_index(drop=True)
    y_hum = pd.DataFrame(hum_I).reset_index(drop=True)
    y_co2 = pd.DataFrame(co2_I).reset_index(drop=True)
    ############################################################################################
    # CORRECIÃ“N DATOS CARRITO
    X = DATA_C.reset_index(drop=True)
    X = X.drop(X.columns[range(4)], axis=1)
    names = np.delete(names, np.array([0, 1, 2, 3]))
    yearday = X.iloc[:, 32]
    #posis = pd.concat([X.iloc[:, 0], X.iloc[:, 1]], axis=1)
    #tt = pd.concat([X.iloc[:, np.array([4, 7, 10, 13, 16, 19, 22, 25])]])
    #tt_co2 = pd.concat([X.iloc[:, np.array([5, 8, 11, 14, 17, 20, 23, 26])]])

    pos_cajas_x = np.array([50.1, 0, 47.1, 4, 20.6, 30.6, 44, 11.5])
    pos_cajas_y = np.array([3, 16, 14.4, 0, 9.4, 15.7, 6.3, 7.8])
    pos_cajas_z = np.array([3.2, 2.4, 3.2, 3, 2.2, 2.5, 2.1, 1.9])
    pos_cajas = pd.concat([pd.DataFrame(pos_cajas_x), pd.DataFrame(pos_cajas_y)], axis=1)
    pos_cajasT = pd.concat([pd.DataFrame(pos_cajas_x), pd.DataFrame(pos_cajas_y), pd.DataFrame(pos_cajas_z)], axis=1)

    ####################################################################################################################
    # Training correction
    ####################################################################################################################
    #tempet1 = y_temp
#
    #p1 = 0
    #p = yearday[0]
#
    #average_temp = []
    #average_co2 = []
    #indices = []
    #while p1 < len(yearday):
    #    indi = np.where(np.array(yearday) == p)[0]
    #    indices.append(len(indi))
#
    #    if np.sum(posis.iloc[indi, 0] - posis.iloc[indi, 0].iloc[0]) == 0 and np.sum(
    #            posis.iloc[indi, 1] - posis.iloc[indi, 1].iloc[0]) == 0:
    #        pp = np.array([posis.iloc[indi, 0].iloc[0], posis.iloc[indi, 1].iloc[0]])
    #        diff = np.abs(pos_cajas - pp)
    #        diff = np.sqrt(diff.iloc[:, 0] ** 2 + diff.iloc[:, 1] ** 2)
    #        # diff=np.sqrt(np.array(diff[:, 0] ** 2 + diff[:, 1] ** 2).astype(np.float64))
    #        diff = np.array(diff)
    #        pond = [0 for x in range(len(diff))]
    #        for t in range(len(diff)):
    #            pond[t] = diff[t] / np.sum(diff)
#
    #        avr1 = [0 for x in range(len(indi))]
    #        avr_co2 = [0 for x in range(len(indi))]
    #        tt1 = tt.iloc[indi, :]
    #        tt_co21 = tt_co2.iloc[indi, :]
    #        for t in range(len(indi)):
    #            avr1[t] = np.average(np.array(tt1.iloc[t, :], dtype=float), weights=pond)
    #            avr_co2[t] = np.average(np.array(tt_co21.iloc[t, :], dtype=float), weights=pond)
#
    #        avr1 = pd.DataFrame(avr1)
    #        avr_co2 = pd.DataFrame(avr_co2)
    #        average_temp.append(avr1)
    #        average_co2.append(avr_co2)
#
    #    else:
    #        ii = np.unique(np.where((posis.iloc[indi, 0] == posis.iloc[indi, :].iloc[0, 0]) & (
    #                posis.iloc[indi, 1] == posis.iloc[indi, :].iloc[0, 1]))[0])
    #        ii2 = np.unique(np.where((posis.iloc[indi, 0] != posis.iloc[indi, :].iloc[0, 0]) | (
    #                posis.iloc[indi, 1] != posis.iloc[indi, :].iloc[0, 1]))[0])
    #        posis3 = posis.iloc[indi, :].iloc[ii]
    #        posis3_2 = posis.iloc[indi, :].iloc[ii2]
    #        pp3 = posis3.iloc[0]
    #        pp3_2 = posis3_2.iloc[0]
#
    #        diff3 = np.abs(pos_cajas - np.array(pp3))
    #        diff3_2 = np.abs(pos_cajas - np.array(pp3_2))
    #        diff3 = np.sqrt(np.array(diff3.iloc[:, 0] ** 2 + diff3.iloc[:, 1] ** 2, dtype=float))
    #        diff3_2 = np.sqrt(np.array(diff3_2.iloc[:, 0] ** 2 + diff3_2.iloc[:, 1] ** 2, dtype=float))
    #        diff3 = np.array(diff3)
    #        diff3_2 = np.array(diff3_2)
#
    #        pond3 = [0 for x in range(len(diff3))]
    #        pond3_2 = [0 for x in range(len(diff3_2))]
    #        for t in range(len(diff3)):
    #            pond3[t] = diff3[t] / np.sum(diff3)
    #        for t in range(len(diff3_2)):
    #            pond3_2[t] = diff3_2[t] / np.sum(diff3_2)
#
    #        tt1 = tt.iloc[indi, :].reset_index(drop=True)
    #        tt_co21 = tt_co2.iloc[indi, :].reset_index(drop=True)
    #        tt3 = tt1.iloc[ii, :]
    #        tt_co23 = tt_co21.iloc[ii, :]
    #        tt3_2 = tt1.iloc[ii2, :]
    #        tt_co23_2 = tt_co21.iloc[ii2, :]
    #        avr3 = [0 for x in range(len(ii))]
    #        avr3_2 = [0 for x in range(len(ii2))]
    #        avr_co23 = [0 for x in range(len(ii))]
    #        avr_co23_2 = [0 for x in range(len(ii2))]
    #        for t in range(len(ii)):
    #            avr3[t] = np.average(np.array(tt3.iloc[t, :], dtype=float), weights=pond3)
    #            avr_co23[t] = np.average(tt_co23.iloc[t, :], weights=pond3)
    #        for t in range(len(ii2)):
    #            avr3_2[t] = np.average(np.array(tt3_2.iloc[t, :], dtype=float), weights=pond3_2)
    #            avr_co23_2[t] = np.average(np.array(tt_co23_2.iloc[t, :], dtype=float), weights=pond3_2)
#
    #        avr = pd.concat([pd.DataFrame(avr3), pd.DataFrame(avr3_2)], axis=0)
    #        avr_co2 = pd.concat([pd.DataFrame(avr_co23), pd.DataFrame(avr_co23_2)], axis=0)
    #        average_temp.append(avr.reset_index(drop=True))
    #        average_co2.append(avr_co2.reset_index(drop=True))
#
    #    p1 = p1 + len(indi)
    #    if p1 < len(yearday):
    #        p = yearday[p1]
    #    else:
    #        p = 0
#
    #average_tempF = pd.DataFrame(np.concatenate(average_temp)).iloc[:, 0]
    #average_co2F = pd.DataFrame(np.concatenate(average_co2)).iloc[:, 0]
    ## st = np.where(yearday == 214)[0][0]-1
    #st = np.where(yearday == 323)[0][0] - 1
    ## st = np.where(yearday == 294)[0][0] - 1
#
    #yearday2 = yearday.drop(range(st, yearday.shape[0] - 1), axis=0)
    #avr1 = average_tempF.iloc[st: average_tempF.shape[0] - 1]
    #avr2 = average_tempF.drop(range(st, average_tempF.shape[0] - 1), axis=0)
    #y_temp1 = y_temp.iloc[st: y_temp.shape[0] - 1]
    #y_temp2 = y_temp.drop(range(st, y_temp.shape[0] - 1), axis=0)
    #posis2 = posis.drop(range(st, posis.shape[0] - 1), axis=0)
    #ratio1 = np.mean(y_temp1.iloc[:, 0] / avr1)
#
    #p1 = 0
    #yearday2 = np.array(yearday2)
    #p = yearday2[0]
#
    #mean_temp = []
    #while p1 < len(yearday2):
    #    indi = np.where(np.array(yearday2) == p)[0]
    #    if np.sum(posis2.iloc[indi, 0] - posis2.iloc[indi, 0].iloc[0]) == 0 and np.sum(
    #            posis2.iloc[indi, 1] - posis2.iloc[indi, 1].iloc[0]) == 0:
    #        rat = np.array(y_temp2.iloc[indi, 0] / avr2.iloc[indi], dtype=float)
    #        r1 = np.nanmean(rat) / ratio1
    #        y_temp2.iloc[indi, 0] = y_temp2.iloc[indi, 0] / r1
    #    else:
    #        yyy = y_temp2.iloc[indi, 0]
    #        avr22 = avr2.iloc[indi]
    #        ii = np.unique(np.where((posis2.iloc[indi, 0] == posis2.iloc[indi, :].iloc[0, 0]) & (
    #                posis2.iloc[indi, 1] == posis2.iloc[indi, :].iloc[0, 1]))[0])
    #        ii2 = np.unique(np.where((posis2.iloc[indi, 0] != posis2.iloc[indi, :].iloc[0, 0]) | (
    #                posis2.iloc[indi, 1] != posis2.iloc[indi, :].iloc[0, 1]))[0])
#
    #        rat3 = np.array(yyy.iloc[ii] / avr22.iloc[ii], dtype=float)
    #        r13 = np.nanmean(rat3) / ratio1
    #        yyy.iloc[ii] = yyy.iloc[ii] / r13
    #        rat3_2 = np.array(yyy.iloc[ii2] / avr22.iloc[ii2], dtype=float)
    #        r13_2 = np.nanmean(rat3_2) / ratio1
    #        yyy.iloc[ii2] = yyy.iloc[ii2] / r13_2
#
    #        y_temp2.iloc[indi, 0] = yyy.iloc[:]
#
    #    p1 = p1 + len(indi)
    #    if p1 < len(yearday2):
    #        p = yearday2[p1]
    #    else:
    #        p = 0
#
    #yy = pd.concat([y_temp2, y_temp1], axis=0).reset_index(drop=True)
    #############################################################################################
    #y_temp_good = yy
    #y_hum = pd.DataFrame(hum).reset_index(drop=True)
    #kelvin = np.array(y_temp.iloc[:, 0] + 273.15, dtype=float)
    #ee = np.exp(14.2928 - (5291 / kelvin))
    #especifica = 0.622 / (press * 0.1 / (y_hum.iloc[:, 0] * ee) - 1)
    #kelvin = np.array(y_temp_good.iloc[:, 0] + 273.15, dtype=float)
    #ee = np.exp(14.2928 - (5291 / kelvin))
    #y_hum1 = pd.DataFrame((press * especifica * 0.1) / ((0.622 + especifica) * ee))

    ######################################################################################################################
    #####
    #y_temp_final = y_temp_good
    #y_temp_final = y_temp_final.reset_index(drop=True)
    #y_hum_final = y_hum1
    #y_hum_final = y_hum_final.reset_index(drop=True)
    #y_co2_final = y_co2
    #y_co2_final = y_co2_final.reset_index(drop=True)
#
    #X_#inal = X_final.reset_index(drop=True)

    # ui = np.where(np.isfinite(temp_I))[0]
    # for g in range(y_temp_test.shape[0]):
    #    temp_I.iloc[ui[g]] = y_temp_test.iloc[g,:]

    dayofyear = X.iloc[:, 32]
    X_final = X.drop(X.columns[33], axis=1)

    # distances to fixed devices
    posit = X_final.iloc[:, range(3)]

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
         distanceY_02, distanceY_09, distanceY_1a, distanceY_3d, distanceY_B1, distanceY_B2, distanceY_B3, distanceY_B4,
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

    # EliminaciÃ³n de Agosto de la muestra
    #se = [0 for x in range(31)]
    #out = []
    #out2 = []
    #for i in range(31):
    #    se[i] = 214 + i
    #    a2 = dayofyear
    #    ii2 = np.where(a2 == se[i])[0]
    #    if len(ii) > 0:
    #        out2.append(ii2)
#
    #X_final = X_final.drop(np.concatenate(out2), axis=0)
    #y_temp_final = y_temp_final.drop(np.concatenate(out2), axis=0)
    #y_hum_final = y_hum_final.drop(np.concatenate(out2), axis=0)
    #y_co2_final = y_co2_final.drop(np.concatenate(out2), axis=0)
    #dayofyear = pd.DataFrame(dayofyear).drop(np.concatenate(out2), axis=0)
#
    #X_final = X_final.reset_index(drop=True)
    #y_temp_final = y_temp_final.reset_index(drop=True)
    #y_hum_final = y_hum_final.reset_index(drop=True)
    #y_co2_final = y_co2_final.reset_index(drop=True)
    #dayofyear = pd.DataFrame(dayofyear).reset_index(drop=True)
    med_temp = np.nanmean(y_temp)
    med_hum = np.nanmean(y_hum)
    med_co2 = np.nanmean(y_co2)

    x_final = X_final.reset_index(drop=True)
    y_final = y_temp
    #y_final  =y_hum_final
    y_final = y_final.reset_index(drop=True)

    yy1 = np.where((x_final.iloc[:, 0] == 6.9) | (x_final.iloc[:, 0] == 26))[0]
    yy2 = np.where((x_final.iloc[:, 1] == 4) | (x_final.iloc[:, 1] == 14.55))[0]

    yy3 = np.where((x_final.iloc[:, 0] == 46.3) |(x_final.iloc[:, 0] == 28.8))[0]
    yy4 = np.where((x_final.iloc[:, 1] == 7.6) | (x_final.iloc[:, 1] == 10.1))[0]

    zz1 = np.intersect1d(yy1, yy2)
    zz2 = np.intersect1d(yy3, yy4)
    zz1 = np.sort(np.concatenate((zz1, zz2)))


    y_final = y_final.drop(zz1, axis=0)
    y_final = y_final.reset_index(drop=True)

    x_final = x_final.drop(zz1, axis=0)
    x_final = x_final.reset_index(drop=True)



    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    #######################################################################################################
    #temp_obj_T, temp_x_T, temp_obj, temp_x = nsga2_individual(x_final, y_final, X_test, y_test, med_temp, contador)
    temp_obj_T, temp_x_T, temp_obj, temp_x,res= nsga2_individual(x_final, y_final, med_temp, contador)
    #hum_obj_T, hum_x_T, hum_obj, hum_x, res = nsga2_individual(x_final, y_final,xxx, yyy, med_hum, contador)
    #hum_obj_T, hum_x_T, hum_obj, hum_x, res = nsga2_individual(x_final, y_final,med_hum, contador)


    print("Estructuras factibles temperatura", temp_x_T, "y objetivos", temp_obj_T)
    #print("Estructuras factibles humedad", hum_x_T, "y objetivo", hum_obj_T)
    # print("Estructuras factibles co2", co2_x_T, "y objetivo", co2_obj_T)
    print("Estructura optima temperatura", temp_x, "objetivo", temp_obj)
    #print("Estructura optima humedad", hum_x, "objetivo", hum_obj)
    # print("Estructura optima co2", co2_x, "objetivo", co2_obj)
    # return(temp_obj_T, temp_x_T,hum_obj_T, hum_x_T ,co2_obj_T, co2_x_T,temp_obj, temp_x,hum_obj, hum_x ,co2_obj, co2_x)
    # return (temp_obj_T, temp_x_T, temp_obj, temp_x,res)
    return (temp_obj_T, temp_x_T, temp_obj, temp_x, res)


def position(positions):
    dat = positions.drop(['x', 'y'], axis=1)
    changes = []
    for i in range(len(dat)):
        dat1 = dat.iloc[i]
        dd = datetime.datetime(dat1[0], dat1[1], dat1[2], dat1[3], dat1[4], dat1[5])
        changes.append(dd)

    pos_x = [0 for x in range(len(variables_02[0]))]
    pos_y = [0 for x in range(len(variables_02[0]))]
    j = 0
    for h in range(1, len(changes)):
        diff = changes[h] - changes[h - 1]
        days, seconds = diff.days, diff.total_seconds()
        minutes = int(seconds / 60)
        pos_x[j:(j + minutes)] = np.repeat(positions['x'][h - 1], minutes)
        pos_y[j:(j + minutes)] = np.repeat(positions['y'][h - 1], minutes)
        j = j + minutes

    return pos_x, pos_y


def position_meteo(var, l):
    var1 = [0 for x in range(l)]
    k = 0
    for i in range(len(var)):
        vv = np.repeat(var[i], 10)
        var1[k:(k + 10)] = vv
        k = k + 10

    return var1


######################################################################################################################
######################################################################################################################
# Loading of data from the different hosts
variables = []
# time_end = datetime.datetime(2020, 11, 26, 13, 50, 0)
#time_end = datetime.datetime(2021, 1, 20, 23, 0, 0)
time_end = datetime.datetime(2021, 3, 10, 12, 0, 0)

#############################################################################
#############################################################################
# 0x6a52
pos_z = loading_carrito(['vert'], 'vertpantilt', '0x6a52', variables, time_end)
variables = []
variables_52 = loading_carrito(['humidity', 'temperature'], 'sht31d', '0x6a52', variables, time_end)
variables_52 = loading_carrito(['pressure'], 'bme680_bsec', '0x6a52', variables_52, time_end)
variables_52 = loading_carrito(['co2'], 'mhz14', '0x6a52', variables_52, time_end)
names_52 = ['humidity_C', 'temperature_C', 'pressure_C', 'co2_C']

# Data of the boxes
# 0x6a02
variables = []
variables_02 = loading(['humidity', 'temperature'], 'sht31d', '0x6a02', variables, time_end)
variables_02 = loading(['co2'], 'mhz14', '0x6a02', variables_02, time_end)
names_02 = ['humidity_02', 'temperature_02', 'co2_02']
# 0x6a09
variables = []
variables_09 = loading(['humidity', 'temperature'], 'sht31d', '0x6a09', variables, time_end)
variables_09 = loading(['co2'], 'mhz14', '0x6a09', variables_09, time_end)
names_09 = ['humidity_09', 'temperature_09', 'co2_09']
# 0x6a1a
variables = []
variables_1a = loading(['humidity', 'temperature'], 'sht31d', '0x6a1a', variables, time_end)
variables_1a = loading(['co2'], 'mhz14', '0x6a1a', variables_1a, time_end)
names_1a = ['humidity_1a', 'temperature_1a', 'co2_1a']
# 0x6a3d
variables = []
variables_3d = loading(['humidity', 'temperature'], 'sht31d', '0x6a3d', variables, time_end)
variables_3d = loading(['co2'], 'mhz14', '0x6a3d', variables_3d, time_end)
names_3d = ['humidity_3d', 'temperature_3d', 'co2_3d']
# rpiB1
variables = []
variables_B1 = loading(['humidity', 'temperature'], 'sht31d', 'rpiB1', variables, time_end)
variables_B1 = loading(['co2'], 'mhz14', 'rpiB1', variables_B1, time_end)
names_B1 = ['humidity_B1', 'temperature_B1', 'co2_B1']
# rpiB2
variables = []
variables_B2 = loading(['humidity', 'temperature'], 'sht31d', 'rpiB2', variables, time_end)
variables_B2 = loading(['co2'], 'mhz14', 'rpiB2', variables_B2, time_end)
names_B2 = ['humidity_B2', 'temperature_B2', 'co2_B2']
# rpiB3
variables = []
variables_B3 = loading(['humidity', 'temperature'], 'sht31d', 'rpiB3', variables, time_end)
variables_B3 = loading(['co2'], 'mhz14', 'rpiB3', variables_B3, time_end)
names_B3 = ['humidity_B3', 'temperature_B3', 'co2_B3']
# rpiB4
variables = []
variables_B4 = loading(['humidity', 'temperature'], 'sht31d', 'rpiB4', variables, time_end)
variables_B4 = loading(['co2'], 'mhz14', 'rpiB4', variables_B4, time_end)
names_B4 = ['humidity_B4', 'temperature_B4', 'co2_B4']

# Meteo data
variables = []
# variables_meteo = loading(['humidity', 'pressure','radiation','rain','temperature','windeast','windnorth','windspeed'], 'meteo', 'none', variables, time_end)
variables_meteo = loading(['radiation', 'temperature'], 'meteo', 'none', variables, time_end)
names_meteo = ['radiation_M', 'temperature_M']
var_meteo = []
for u in range(len(variables_meteo)):
    var_meteo.append(position_meteo(variables_meteo[u], len(variables_B4[0])))

# Cart positions
positions = pd.read_csv("positions_new.csv", sep=";", decimal=",")
pos_x, pos_y = position(positions)

car = pd.DataFrame(np.array(variables_52).transpose())
z = pd.DataFrame(pos_z[0])
car = car.reset_index(drop=True)

# DATA JOIN
variables = pd.concat(
    [car, pd.DataFrame(pos_x).set_index(car.index),
     pd.DataFrame(pos_y).set_index(car.index),
     z.set_index(car.index),
     pd.DataFrame(np.array(variables_02).transpose()).set_index(car.index),
     pd.DataFrame(np.array(variables_09).transpose()).set_index(car.index),
     pd.DataFrame(np.array(variables_1a).transpose()).set_index(car.index),
     pd.DataFrame(np.array(variables_3d).transpose()).set_index(car.index),
     pd.DataFrame(np.array(variables_B1).transpose()).set_index(car.index),
     pd.DataFrame(np.array(variables_B2).transpose()).set_index(car.index),
     pd.DataFrame(np.array(variables_B3).transpose()).set_index(car.index),
     pd.DataFrame(np.array(variables_B4).transpose()).set_index(car.index),
     pd.DataFrame(np.array(var_meteo).transpose()).set_index(car.index)], axis=1)

pos = ['pos_x', 'pos_y', 'pos_z']
variables.columns = np.concatenate(
    [names_52, pos, names_02, names_09, names_1a, names_3d, names_B1, names_B2, names_B3, names_B4, names_meteo])
names = variables.columns
############################################################################################

############################################################################################
# Parameters definition
date_init = '2020-06-17 11:00:00'
ntest = 4000

manager = multiprocessing.Manager()
dictionary = manager.dict()
contador = manager.list()
contador.append(0)
# General function to train and save the models
# temp_obj_T, temp_x_T,hum_obj_T, hum_x_T ,co2_obj_T, co2_x_T,temp_obj, temp_x,hum_obj, hum_x,co2_obj, co2_x = MTI_train(variables, date_init, ntest, names)
temp_obj_T, temp_x_T, temp_obj, temp_x, res = MTI_train(variables, date_init, ntest, names, contador)

print('El nÃºmero de evaluaciones totales fue', contador)
np.savetxt('temp_obj_total2.txt', temp_obj_T)
np.savetxt('temp_X_total2.txt', temp_x_T)
np.savetxt('temp_obj2.txt', temp_obj)
np.savetxt('temp_X2.txt', temp_x)
# np.savetxt('result.txt',res)

print(f"{round((time() - t) / 3600, 2)} h.")


from itertools import combinations
perm = combinations([1,2,3,4,5,6,7,8],4)
pp=list(perm)