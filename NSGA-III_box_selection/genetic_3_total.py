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
from pymoo.util.running_metric import RunningMetric
from pymoo.visualization.pcp import PCP
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
from pymoo.algorithms.nsga3 import NSGA3
from pymoo.factory import get_problem, get_reference_directions
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from sklearn.metrics import r2_score

import numpy as np
# import os, psutil
from pymoo.util.misc import stack
from pymoo.model.problem import Problem
# from pymoo.util.misc import covert_to_type
from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
from sklearn.metrics import r2_score
from pymoo.factory import get_problem, get_visualization, get_decomposition

import xgboost as xgb
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
    mbe = np.mean(y_true - y_pred) / med
    return (mbe * 100)


#def complex(neurons1, neurons2, neurons3, max_N, max_H):
#    if neurons1 > 0 and neurons2 == 0 and neurons3 == 0:
#        u = 1
#        W = neurons1
#    elif neurons1 > 0 and neurons2 > 0 and neurons3 == 0:
#        u = 2
#        W = np.array([neurons1, neurons2])
#    elif neurons1 > 0 and neurons2 > 0 and neurons3 > 0:
#        u = 3
#        W = np.array([neurons1, neurons2, neurons3])
#
#    # F = 0.5*(u / max_H) + np.sum(W / max_N)
#    F = 0.25 * (u / max_H) + 0.75 * np.sum((neurons1, neurons2, neurons3)) / max_N
#
#    return F


#def ranges(nums):
#    nums = sorted(set(nums))
#    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
#    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
#    return list(zip(edges, edges))


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
    pred_value.columns = names[pred_col]
    pred_value = pd.DataFrame(pred_value)

    pred_value['id'] = range(1, len(pred_value) + 1)
    pred_value.set_index('id', inplace=True)
    final_df = pd.concat([t, pred_value], axis=1)

    return final_df


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


#def nas_function(x):
#    for j in range(x.shape[0]):
#        # Possibility of missing values
#        if any(np.array(np.isnan(x.iloc[j, :]))):
#            ii = np.where(np.isnan(x.iloc[j, :]))[0]
#            x.iloc[j, ii] = 0
#    return (x)


def XGB(X, y_temp, y_hum, y_co2, med_temp, med_hum, med_co2,m_depth_temp, etaz_temp,m_depth_hum, etaz_hum,m_depth_co2, etaz_co2,box1, box2, box3, box4, box5, box6, box7, box8):
    #xgb_params = {'max_depth':6,'eta': 1,'objective': 'squarederror'}

    name1 = tuple([m_depth_temp, etaz_temp,m_depth_hum, etaz_hum,m_depth_co2, etaz_co2, box1, box2, box3, box4, box5, box6, box7, box8])
    try:
        a0, a1 ,a3,a4 = dictionary[name1]
        return a0, a1, a3,a4

    except KeyError:
        pass

    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################

    #possX = [9.8, 40.9, 17.6]
    #possY = [7.8, 11.6, 1.6]
    #possX = np.array([9.8, 12.4, 7.3, 7.3, 28.9, 39.8, 40.9, 19.7, 4, 45.5, 27.9, 28.7, 46.8, 47.7, 36.4, 21.5, 0.3, 0.4,17.6, 17.6, 32.2, 2.3])
    possX = np.array([9.8, 28.9, 40.9, 4, 27.9, 46.8, 21.5, 45.5, 17.6, 2.3])
    #possY = np.array([7.8, 14.7, 14.7, 11.7, 1.6, 1.6, 11.6, 5, 11.2, 5.6, 14.6, 7.8, 2.2, 1.5, 1.6, 7.5, 6.4, 14.6,6.8, 1.6, 4.7, 4.7])
    possY = np.array([7.8, 1.6, 11.6, 11.2, 14.6, 2.2, 7.5, 5.6, 1.6, 4.7])
    errors_temp = [0 for x in range(len(possX))]
    errors_hum = [0 for x in range(len(possX))]
    errors_co2 = [0 for x in range(len(possX))]
    #omplexity = [0 for x in range(len(possX))]
    #r2 = [0 for x in range(len(possX))]
    X_original = pd.DataFrame(X.copy())
    for j in range(len(possX)):
        print(j)
        yy1 =np.where(X_original.iloc[:, 0] == possX[j])
        yy2 = np.where(X_original.iloc[:, 1] == possY[j])
        zz1 = np.intersect1d(yy1, yy2)
        X_val1 = X_original.iloc[zz1]
        X_val1 = X_val1.reset_index(drop=True)
        y_val1_temp,y_val1_hum, y_val1_co2  = y_temp.iloc[zz1],y_hum.iloc[zz1],y_co2.iloc[zz1]
        y_val1_temp, y_val1_hum, y_val1_co2 = y_val1_temp.reset_index(drop=True), y_val1_hum.reset_index(drop=True), y_val1_co2.reset_index(drop=True)
        X_final = X_original.drop(zz1, axis=0)
        X_finaL = X_final.reset_index(drop=True)
        y1_temp, y1_hum, y1_co2 = y_temp.drop(zz1, axis=0),y_hum.drop(zz1, axis=0),y_co2.drop(zz1, axis=0)
        y1_temp, y1_hum, y1_co2 = y1_temp.reset_index(drop=True),y1_hum.reset_index(drop=True),y1_co2.reset_index(drop=True)

        if X_val1.shape[0]>5:
            if j==0:
                yy1 = np.where(X_finaL.iloc[:, 0] == possX[len(possX)-1])[0]
                yy2 = np.where(X_final.iloc[:, 1] == possY[len(possY)-1])[0]
                zz1 = np.intersect1d(yy1, yy2)
                X_test1 = X_final.iloc[zz1]
                X_test1 = X_test1.reset_index(drop=True)
                y_test1_temp, y_test1_hum, y_test1_co2 = y1_temp.iloc[zz1],y1_hum.iloc[zz1],y1_co2.iloc[zz1]
                y_test1_temp, y_test1_hum, y_test1_co2= y_test1_temp.reset_index(drop=True),y_test1_hum.reset_index(drop=True),y_test1_co2.reset_index(drop=True)
                X_final = X_final.drop(zz1, axis=0)
                X_finaL = X_final.reset_index(drop=True)
                y1_temp, y1_hum, y1_co2 = y1_temp.drop(zz1, axis=0),y1_hum.drop(zz1, axis=0),y1_co2.drop(zz1, axis=0)
                y1_temp, y1_hum, y1_co2 = y1_temp.reset_index(drop=True),y1_hum.reset_index(drop=True),y1_co2.reset_index(drop=True)
            else:
                yy1 = np.where(X_finaL.iloc[:, 0] == possX[j-1])[0]
                yy2 = np.where(X_final.iloc[:, 1] == possY[j-1])[0]
                zz1 = np.intersect1d(yy1, yy2)
                X_test1 = X_final.iloc[zz1]
                X_test1 = X_test1.reset_index(drop=True)
                y_test1_temp, y_test1_hum, y_test1_co2 = y1_temp.iloc[zz1], y1_hum.iloc[zz1], y1_co2.iloc[zz1]
                y_test1_temp, y_test1_hum, y_test1_co2 = y_test1_temp.reset_index(drop=True), y_test1_hum.reset_index(
                    drop=True), y_test1_co2.reset_index(drop=True)
                X_final = X_final.drop(zz1, axis=0)
                X_finaL = X_final.reset_index(drop=True)
                y1_temp, y1_hum, y1_co2 = y1_temp.drop(zz1, axis=0), y1_hum.drop(zz1, axis=0), y1_co2.drop(zz1, axis=0)
                y1_temp, y1_hum, y1_co2 = y1_temp.reset_index(drop=True), y1_hum.reset_index(
                    drop=True), y1_co2.reset_index(drop=True)

            if X_test1.shape[0]>5:
                X_final = X_final.drop(X_final.columns[np.array([0, 1, 2])], axis=1)
                X_test1 = X_test1.drop(X_test1.columns[np.array([0, 1, 2])], axis=1)
                X_val1 = X_val1.drop(X_val1.columns[np.array([0, 1, 2])], axis=1)
                temperaturas_train = np.array(X_final.iloc[:, np.array([1, 4, 7, 10, 13, 16, 19, 22, 26, 31, 34, 37, 40, 43, 46, 49, 52])])
                temperaturas_test = np.array(X_test1.iloc[:, np.array([1, 4, 7, 10, 13, 16, 19, 22, 26, 31, 34, 37, 40, 43, 46, 49, 52])])
                temperaturas_val = np.array(X_val1.iloc[:, np.array([1, 4, 7, 10, 13, 16, 19, 22, 26, 31, 34, 37, 40, 43, 46, 49, 52])])

                humedad_train = np.array(X_final.iloc[:, np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 30, 33, 36, 39, 42, 45, 48, 51])])
                humedad_test = np.array(X_test1.iloc[:, np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 30, 33, 36, 39, 42, 45, 48, 51])])
                humedad_val = np.array(X_val1.iloc[:, np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 30, 33, 36, 39, 42, 45, 48, 51])])

                co2_train = np.array(X_final.iloc[:, np.array([2, 5, 8, 11, 14, 17, 20, 23, 32, 35, 38, 41, 44, 47, 50, 53])])
                co2_test = np.array(X_test1.iloc[:, np.array([2, 5, 8, 11, 14, 17, 20, 23, 32, 35, 38, 41, 44, 47, 50, 53])])
                co2_val = np.array(X_val1.iloc[:, np.array([2, 5, 8, 11, 14, 17, 20, 23, 32, 35, 38, 41, 44, 47, 50, 53])])

                diss_train = np.array(X_final.iloc[:, range(X_final.shape[1] - 24, X_final.shape[1])])
                diss_test = np.array(X_test1.iloc[:, range(X_final.shape[1] - 24, X_final.shape[1])])
                diss_val = np.array(X_val1.iloc[:, range(X_final.shape[1] - 24, X_final.shape[1])])
                rad_train = np.array(X_final.iloc[:, np.array([25])])
                rad_test = np.array(X_test1.iloc[:, np.array([25])])
                rad_val = np.array(X_val1.iloc[:, np.array([25])])
                resto_train = X_final.iloc[:, np.array([27, 28, 29])]
                resto_test = X_test1.iloc[:, np.array([27, 28, 29])]
                resto_val = X_val1.iloc[:, np.array([27, 28, 29])]

                scalar_temp = MinMaxScaler(feature_range=(-1, 1))
                scalar_hum = MinMaxScaler(feature_range=(-1, 1))
                scalar_co2 = MinMaxScaler(feature_range=(-1, 1))
                scalardist = MinMaxScaler(feature_range=(-1, 1))
                scalar_rad = MinMaxScaler(feature_range=(-1, 1))
                scalarresto = MinMaxScaler(feature_range=(-1, 1))
                scalar_temp.fit(np.concatenate((np.concatenate(temperaturas_train), np.concatenate(temperaturas_test),np.concatenate(temperaturas_val),
                                                np.array(y1_temp), np.array(y_test1_temp),np.array(y_val1_temp))).reshape(-1, 1))
                scalardist.fit(np.concatenate((np.concatenate(diss_train), np.concatenate(diss_test),np.concatenate(diss_val))).reshape(-1, 1))
                scalar_hum.fit(np.concatenate((np.concatenate(humedad_train), np.concatenate(humedad_test),np.concatenate(humedad_val),
                                               np.array(y1_hum), np.array(y_test1_hum), np.array(y_val1_hum))).reshape(-1, 1))
                scalar_co2.fit(np.concatenate((np.concatenate(co2_train), np.concatenate(co2_test),np.concatenate(co2_val),
                                               np.array(y1_co2), np.array(y_test1_co2),
                                               np.array(y_val1_co2))).reshape(-1, 1))
                scalar_rad.fit(np.concatenate((np.concatenate(rad_train), np.concatenate(rad_test),np.concatenate(rad_val))).reshape(-1, 1))
                scalarresto.fit(pd.concat([resto_train, resto_test,resto_val], axis=0))
                y_scaled_temp = pd.DataFrame(scalar_temp.transform(np.array(y1_temp).reshape(-1, 1)))
                y_scaled_hum = pd.DataFrame(scalar_hum.transform(np.array(y1_hum).reshape(-1, 1)))
                y_scaled_co2 = pd.DataFrame(scalar_co2.transform(np.array(y1_co2).reshape(-1, 1)))
                y_test1_temp = pd.DataFrame(scalar_temp.transform(np.array(y_test1_temp).reshape(-1, 1)))
                y_test1_hum = pd.DataFrame(scalar_hum.transform(np.array(y_test1_hum).reshape(-1, 1)))
                y_test1_co2 = pd.DataFrame(scalar_co2.transform(np.array(y_test1_co2).reshape(-1, 1)))
                #y_val1 = pd.DataFrame(scalar_temp.transform(np.array(y_val1).reshape(-1, 1)))

                temperaturas_train1 = np.zeros((temperaturas_train.shape[0], temperaturas_train.shape[1]))
                temperaturas_test1 = np.zeros((temperaturas_test.shape[0], temperaturas_test.shape[1]))
                temperaturas_val1 = np.zeros((temperaturas_val.shape[0], temperaturas_val.shape[1]))
                humedad_train1 = np.zeros((humedad_train.shape[0], humedad_train.shape[1]))
                humedad_test1 = np.zeros((humedad_test.shape[0], humedad_test.shape[1]))
                humedad_val1 = np.zeros((humedad_val.shape[0], humedad_val.shape[1]))
                for i in range(temperaturas_train.shape[1]):
                    temperaturas_train1[:, i] = scalar_temp.transform(temperaturas_train[:, i].reshape(-1, 1))[:, 0]
                    temperaturas_test1[:, i] = scalar_temp.transform(temperaturas_test[:, i].reshape(-1, 1))[:, 0]
                    temperaturas_val1[:, i] = scalar_temp.transform(temperaturas_val[:, i].reshape(-1, 1))[:, 0]
                    humedad_train1[:, i] = scalar_hum.transform(humedad_train[:, i].reshape(-1, 1))[:, 0]
                    humedad_test1[:, i] = scalar_hum.transform(humedad_test[:, i].reshape(-1, 1))[:, 0]
                    humedad_val1[:, i] = scalar_hum.transform(humedad_val[:, i].reshape(-1, 1))[:, 0]
                temperaturas_train1 = pd.DataFrame(temperaturas_train1)
                temperaturas_test1 = pd.DataFrame(temperaturas_test1)
                temperaturas_val1 = pd.DataFrame(temperaturas_val1)
                humedad_train1 = pd.DataFrame(humedad_train1)
                humedad_test1 = pd.DataFrame(humedad_test1)
                humedad_val1 = pd.DataFrame(humedad_val1)
                co2_train1 = np.zeros((co2_train.shape[0], co2_train.shape[1]))
                co2_test1 = np.zeros((co2_test.shape[0], co2_train.shape[1]))
                co2_val1 = np.zeros((co2_val.shape[0], co2_train.shape[1]))
                for i in range(co2_train.shape[1]):
                    co2_train1[:, i] = scalar_co2.transform(co2_train[:, i].reshape(-1, 1))[:, 0]
                    co2_test1[:, i] = scalar_co2.transform(co2_test[:, i].reshape(-1, 1))[:, 0]
                    co2_val1[:, i] = scalar_co2.transform(co2_val[:, i].reshape(-1, 1))[:, 0]
                co2_train1 = pd.DataFrame(co2_train1)
                co2_test1 = pd.DataFrame(co2_test1)
                co2_val1 = pd.DataFrame(co2_val1)
                diss_train1 = np.zeros((diss_train.shape[0], diss_train.shape[1]))
                diss_test1 = np.zeros((diss_test.shape[0], diss_train.shape[1]))
                diss_val1 = np.zeros((diss_val.shape[0], diss_train.shape[1]))
                for i in range(diss_train.shape[1]):
                    diss_train1[:, i] = scalardist.transform(diss_train[:, i].reshape(-1, 1))[:, 0]
                    diss_test1[:, i] = scalardist.transform(diss_test[:, i].reshape(-1, 1))[:, 0]
                    diss_val1[:, i] = scalardist.transform(diss_val[:, i].reshape(-1, 1))[:, 0]
                rad_train1 = np.zeros((rad_train.shape[0], rad_train.shape[1]))
                rad_test1 = np.zeros((rad_test.shape[0], rad_train.shape[1]))
                rad_val1 = np.zeros((rad_val.shape[0], rad_train.shape[1]))
                for i in range(rad_train.shape[1]):
                    rad_train1[:, i] = scalar_rad.transform(rad_train[:, i].reshape(-1, 1))[:, 0]
                    rad_test1[:, i] = scalar_rad.transform(rad_test[:, i].reshape(-1, 1))[:, 0]
                    rad_val1[:, i] = scalar_rad.transform(rad_val[:, i].reshape(-1, 1))[:, 0]
                diss_train1 = pd.DataFrame(diss_train1)
                diss_test1 = pd.DataFrame(diss_test1)
                diss_val1 = pd.DataFrame(diss_val1)
                rad_train1 = pd.DataFrame(rad_train1)
                rad_test1 = pd.DataFrame(rad_test1)
                rad_val1 = pd.DataFrame(rad_val1)
                resto_train1 = pd.DataFrame(scalarresto.transform(resto_train))
                resto_test1 = pd.DataFrame(scalarresto.transform(resto_test))
                resto_val1 = pd.DataFrame(scalarresto.transform(resto_val))
                temperaturas_trainM= pd.DataFrame(np.array(temperaturas_train1.iloc[:, 8]))
                temperaturas_train_box1 = pd.DataFrame(np.array(temperaturas_train1.iloc[:, np.array([0, 9])]))
                temperaturas_train_box2 = pd.DataFrame(np.array(temperaturas_train1.iloc[:, np.array([1, 10])]))
                temperaturas_train_box3 = pd.DataFrame(np.array(temperaturas_train1.iloc[:, np.array([2, 11])]))
                temperaturas_train_box4 = pd.DataFrame(np.array(temperaturas_train1.iloc[:, np.array([3, 12])]))
                temperaturas_train_box5 = pd.DataFrame(np.array(temperaturas_train1.iloc[:, np.array([4, 13])]))
                temperaturas_train_box6 = pd.DataFrame(np.array(temperaturas_train1.iloc[:, np.array([5, 14])]))
                temperaturas_train_box7 = pd.DataFrame(np.array(temperaturas_train1.iloc[:, np.array([6, 15])]))
                temperaturas_train_box8 = pd.DataFrame(np.array(temperaturas_train1.iloc[:, np.array([7, 16])]))
                temperaturas_testM = pd.DataFrame(np.array(temperaturas_test1.iloc[:, 8]))
                temperaturas_test_box1 = pd.DataFrame(np.array(temperaturas_test1.iloc[:, np.array([0, 9])]))
                temperaturas_test_box2 = pd.DataFrame(np.array(temperaturas_test1.iloc[:, np.array([1, 10])]))
                temperaturas_test_box3 = pd.DataFrame(np.array(temperaturas_test1.iloc[:, np.array([2, 11])]))
                temperaturas_test_box4 = pd.DataFrame(np.array(temperaturas_test1.iloc[:, np.array([3, 12])]))
                temperaturas_test_box5 = pd.DataFrame(np.array(temperaturas_test1.iloc[:, np.array([4, 13])]))
                temperaturas_test_box6 = pd.DataFrame(np.array(temperaturas_test1.iloc[:, np.array([5, 14])]))
                temperaturas_test_box7 = pd.DataFrame(np.array(temperaturas_test1.iloc[:, np.array([6, 15])]))
                temperaturas_test_box8 = pd.DataFrame(np.array(temperaturas_test1.iloc[:, np.array([7, 16])]))
                temperaturas_valM = pd.DataFrame(np.array(temperaturas_val1.iloc[:, 8]))
                temperaturas_val_box1 = pd.DataFrame(np.array(temperaturas_val1.iloc[:, np.array([0, 9])]))
                temperaturas_val_box2 = pd.DataFrame(np.array(temperaturas_val1.iloc[:, np.array([1, 10])]))
                temperaturas_val_box3 = pd.DataFrame(np.array(temperaturas_val1.iloc[:, np.array([2, 11])]))
                temperaturas_val_box4 = pd.DataFrame(np.array(temperaturas_val1.iloc[:, np.array([3, 12])]))
                temperaturas_val_box5 = pd.DataFrame(np.array(temperaturas_val1.iloc[:, np.array([4, 13])]))
                temperaturas_val_box6 = pd.DataFrame(np.array(temperaturas_val1.iloc[:, np.array([5, 14])]))
                temperaturas_val_box7 = pd.DataFrame(np.array(temperaturas_val1.iloc[:, np.array([6, 15])]))
                temperaturas_val_box8 = pd.DataFrame(np.array(temperaturas_val1.iloc[:, np.array([7, 16])]))

                humedad_trainM = pd.DataFrame(np.array(humedad_train1.iloc[:, 8]))
                humedad_train_box1 = pd.DataFrame(np.array(humedad_train1.iloc[:, np.array([0, 9])]))
                humedad_train_box2 = pd.DataFrame(np.array(humedad_train1.iloc[:, np.array([1, 10])]))
                humedad_train_box3 = pd.DataFrame(np.array(humedad_train1.iloc[:, np.array([2, 11])]))
                humedad_train_box4 = pd.DataFrame(np.array(humedad_train1.iloc[:, np.array([3, 12])]))
                humedad_train_box5 = pd.DataFrame(np.array(humedad_train1.iloc[:, np.array([4, 13])]))
                humedad_train_box6 = pd.DataFrame(np.array(humedad_train1.iloc[:, np.array([5, 14])]))
                humedad_train_box7 = pd.DataFrame(np.array(humedad_train1.iloc[:, np.array([6, 15])]))
                humedad_train_box8 = pd.DataFrame(np.array(humedad_train1.iloc[:, np.array([7, 16])]))
                humedad_testM = pd.DataFrame(np.array(humedad_test1.iloc[:, 8]))
                humedad_test_box1 = pd.DataFrame(np.array(humedad_test1.iloc[:, np.array([0, 9])]))
                humedad_test_box2 = pd.DataFrame(np.array(humedad_test1.iloc[:, np.array([1, 10])]))
                humedad_test_box3 = pd.DataFrame(np.array(humedad_test1.iloc[:, np.array([2, 11])]))
                humedad_test_box4 = pd.DataFrame(np.array(humedad_test1.iloc[:, np.array([3, 12])]))
                humedad_test_box5 = pd.DataFrame(np.array(humedad_test1.iloc[:, np.array([4, 13])]))
                humedad_test_box6 = pd.DataFrame(np.array(humedad_test1.iloc[:, np.array([5, 14])]))
                humedad_test_box7 = pd.DataFrame(np.array(humedad_test1.iloc[:, np.array([6, 15])]))
                humedad_test_box8 = pd.DataFrame(np.array(humedad_test1.iloc[:, np.array([7, 16])]))
                humedad_valM = pd.DataFrame(np.array(humedad_val1.iloc[:, 8]))
                humedad_val_box1 = pd.DataFrame(np.array(humedad_val1.iloc[:, np.array([0, 9])]))
                humedad_val_box2 = pd.DataFrame(np.array(humedad_val1.iloc[:, np.array([1, 10])]))
                humedad_val_box3 = pd.DataFrame(np.array(humedad_val1.iloc[:, np.array([2, 11])]))
                humedad_val_box4 = pd.DataFrame(np.array(humedad_val1.iloc[:, np.array([3, 12])]))
                humedad_val_box5 = pd.DataFrame(np.array(humedad_val1.iloc[:, np.array([4, 13])]))
                humedad_val_box6 = pd.DataFrame(np.array(humedad_val1.iloc[:, np.array([5, 14])]))
                humedad_val_box7 = pd.DataFrame(np.array(humedad_val1.iloc[:, np.array([6, 15])]))
                humedad_val_box8 = pd.DataFrame(np.array(humedad_val1.iloc[:, np.array([7, 16])]))

                co2_train_box1 = pd.DataFrame(np.array(co2_train1.iloc[:, np.array([0, 8])]))
                co2_train_box2 = pd.DataFrame(np.array(co2_train1.iloc[:, np.array([1, 9])]))
                co2_train_box3 = pd.DataFrame(np.array(co2_train1.iloc[:, np.array([2, 10])]))
                co2_train_box4 = pd.DataFrame(np.array(co2_train1.iloc[:, np.array([3, 11])]))
                co2_train_box5 = pd.DataFrame(np.array(co2_train1.iloc[:, np.array([4, 12])]))
                co2_train_box6 = pd.DataFrame(np.array(co2_train1.iloc[:, np.array([5, 13])]))
                co2_train_box7 = pd.DataFrame(np.array(co2_train1.iloc[:, np.array([6, 14])]))
                co2_train_box8 = pd.DataFrame(np.array(co2_train1.iloc[:, np.array([7, 15])]))

                co2_test_box1 = pd.DataFrame(np.array(co2_test1.iloc[:, np.array([0, 8])]))
                co2_test_box2 = pd.DataFrame(np.array(co2_test1.iloc[:, np.array([1, 9])]))
                co2_test_box3 = pd.DataFrame(np.array(co2_test1.iloc[:, np.array([2, 10])]))
                co2_test_box4 = pd.DataFrame(np.array(co2_test1.iloc[:, np.array([3, 11])]))
                co2_test_box5 = pd.DataFrame(np.array(co2_test1.iloc[:, np.array([4, 12])]))
                co2_test_box6 = pd.DataFrame(np.array(co2_test1.iloc[:, np.array([5, 13])]))
                co2_test_box7 = pd.DataFrame(np.array(co2_test1.iloc[:, np.array([6, 14])]))
                co2_test_box8 = pd.DataFrame(np.array(co2_test1.iloc[:, np.array([7, 15])]))

                co2_val_box1 = pd.DataFrame(np.array(co2_val1.iloc[:, np.array([0, 8])]))
                co2_val_box2 = pd.DataFrame(np.array(co2_val1.iloc[:, np.array([1, 9])]))
                co2_val_box3 = pd.DataFrame(np.array(co2_val1.iloc[:, np.array([2, 10])]))
                co2_val_box4 = pd.DataFrame(np.array(co2_val1.iloc[:, np.array([3, 11])]))
                co2_val_box5 = pd.DataFrame(np.array(co2_val1.iloc[:, np.array([4, 12])]))
                co2_val_box6 = pd.DataFrame(np.array(co2_val1.iloc[:, np.array([5, 13])]))
                co2_val_box7 = pd.DataFrame(np.array(co2_val1.iloc[:, np.array([6, 14])]))
                co2_val_box8 = pd.DataFrame(np.array(co2_val1.iloc[:, np.array([7, 15])]))

                diss_train_box1 = pd.DataFrame(np.array(diss_train1.iloc[:, np.array([0, 8, 16])]))
                diss_train_box2 = pd.DataFrame(np.array(diss_train1.iloc[:, np.array([1, 9, 17])]))
                diss_train_box3 = pd.DataFrame(np.array(diss_train1.iloc[:, np.array([2, 10, 18])]))
                diss_train_box4 = pd.DataFrame(np.array(diss_train1.iloc[:, np.array([3, 11, 19])]))
                diss_train_box5 = pd.DataFrame(np.array(diss_train1.iloc[:, np.array([4, 12, 20])]))
                diss_train_box6 = pd.DataFrame(np.array(diss_train1.iloc[:, np.array([5, 13, 21])]))
                diss_train_box7 = pd.DataFrame(np.array(diss_train1.iloc[:, np.array([6, 14, 22])]))
                diss_train_box8 = pd.DataFrame(np.array(diss_train1.iloc[:, np.array([7, 15, 23])]))
                diss_test_box1 = pd.DataFrame(np.array(diss_test1.iloc[:, np.array([0, 8, 16])]))
                diss_test_box2 = pd.DataFrame(np.array(diss_test1.iloc[:, np.array([1, 9, 17])]))
                diss_test_box3 = pd.DataFrame(np.array(diss_test1.iloc[:, np.array([2, 10, 18])]))
                diss_test_box4 = pd.DataFrame(np.array(diss_test1.iloc[:, np.array([3, 11, 19])]))
                diss_test_box5 = pd.DataFrame(np.array(diss_test1.iloc[:, np.array([4, 12, 20])]))
                diss_test_box6 = pd.DataFrame(np.array(diss_test1.iloc[:, np.array([5, 13, 21])]))
                diss_test_box7 = pd.DataFrame(np.array(diss_test1.iloc[:, np.array([6, 14, 22])]))
                diss_test_box8 = pd.DataFrame(np.array(diss_test1.iloc[:, np.array([7, 15, 23])]))
                diss_val_box1 = pd.DataFrame(np.array(diss_val1.iloc[:, np.array([0, 8, 16])]))
                diss_val_box2 = pd.DataFrame(np.array(diss_val1.iloc[:, np.array([1, 9, 17])]))
                diss_val_box3 = pd.DataFrame(np.array(diss_val1.iloc[:, np.array([2, 10, 18])]))
                diss_val_box4 = pd.DataFrame(np.array(diss_val1.iloc[:, np.array([3, 11, 19])]))
                diss_val_box5 = pd.DataFrame(np.array(diss_val1.iloc[:, np.array([4, 12, 20])]))
                diss_val_box6 = pd.DataFrame(np.array(diss_val1.iloc[:, np.array([5, 13, 21])]))
                diss_val_box7 = pd.DataFrame(np.array(diss_val1.iloc[:, np.array([6, 14, 22])]))
                diss_val_box8 = pd.DataFrame(np.array(diss_val1.iloc[:, np.array([7, 15, 23])]))

                X_scaled_temp = pd.concat([rad_train1, resto_train1, temperaturas_trainM], axis=1)
                X_test1_temp = pd.concat([rad_test1, resto_test1, temperaturas_testM], axis=1)
                X_val1_temp = pd.concat([rad_val1, resto_val1, temperaturas_valM], axis=1)
                X_scaled_hum = pd.concat([rad_train1, resto_train1, humedad_trainM], axis=1)
                X_test1_hum = pd.concat([rad_test1, resto_test1, humedad_testM], axis=1)
                X_val1_hum = pd.concat([rad_val1, resto_val1, humedad_valM], axis=1)
                X_scaled_co2 = pd.concat([rad_train1, resto_train1], axis=1)
                X_test1_co2 = pd.concat([rad_test1, resto_test1], axis=1)
                X_val1_co2 = pd.concat([rad_val1, resto_val1], axis=1)
                if box1 == 1:
                    X_scaled_temp, X_scaled_hum, X_scaled_co2 = pd.concat([X_scaled_temp, temperaturas_train_box1, diss_train_box1], axis=1),\
                                                                pd.concat([X_scaled_hum, humedad_train_box1, diss_train_box1], axis=1),pd.concat([X_scaled_co2, co2_train_box1, diss_train_box1], axis=1)
                    X_test1_temp, X_test1_hum, X_test1_co2 = pd.concat([X_test1_temp, temperaturas_test_box1, diss_test_box1], axis=1),\
                                                                pd.concat([X_test1_hum, humedad_test_box1, diss_test_box1],axis=1), pd.concat([X_test1_co2, co2_test_box1, diss_test_box1], axis=1)
                    X_val1_temp, X_val1_hum, X_val1_co2 = pd.concat([X_val1_temp, temperaturas_val_box1, diss_val_box1], axis=1), \
                                                             pd.concat([X_val1_hum, humedad_val_box1, diss_val_box1],axis=1), pd.concat([X_val1_co2, co2_val_box1, diss_val_box1], axis=1)
                if box2 == 1:
                    X_scaled_temp, X_scaled_hum, X_scaled_co2 = pd.concat([X_scaled_temp, temperaturas_train_box2, diss_train_box2], axis=1),\
                                                                pd.concat([X_scaled_hum, humedad_train_box2, diss_train_box2],axis=1), pd.concat([X_scaled_co2, co2_train_box2, diss_train_box2], axis=1)
                    X_test1_temp, X_test1_hum, X_test1_co2 = pd.concat([X_test1_temp, temperaturas_test_box2, diss_test_box2], axis=1), \
                                                             pd.concat([X_test1_hum, humedad_test_box2, diss_test_box2],axis=1), pd.concat([X_test1_co2, co2_test_box2, diss_test_box2], axis=1)
                    X_val1_temp, X_val1_hum, X_val1_co2 = pd.concat([X_val1_temp, temperaturas_val_box2, diss_val_box2],axis=1), \
                                                          pd.concat([X_val1_hum, humedad_val_box2, diss_val_box2],axis=1), pd.concat([X_val1_co2, co2_val_box2, diss_val_box2], axis=1)

                if box3 == 1:
                    X_scaled_temp, X_scaled_hum, X_scaled_co2 = pd.concat([X_scaled_temp, temperaturas_train_box3, diss_train_box3], axis=1),\
                                                                pd.concat([X_scaled_hum, humedad_train_box3, diss_train_box3],axis=1), pd.concat([X_scaled_co2, co2_train_box3, diss_train_box3], axis=1)
                    X_test1_temp, X_test1_hum, X_test1_co2 = pd.concat([X_test1_temp, temperaturas_test_box3, diss_test_box3], axis=1), \
                                                             pd.concat([X_test1_hum, humedad_test_box3, diss_test_box3],axis=1), pd.concat([X_test1_co2, co2_test_box3, diss_test_box3], axis=1)
                    X_val1_temp, X_val1_hum, X_val1_co2 = pd.concat([X_val1_temp, temperaturas_val_box3, diss_val_box3],axis=1), \
                                                          pd.concat([X_val1_hum, humedad_val_box3, diss_val_box3],axis=1), pd.concat([X_val1_co2, co2_val_box3, diss_val_box3], axis=1)
                if box4 == 1:
                    X_scaled_temp, X_scaled_hum, X_scaled_co2 = pd.concat([X_scaled_temp, temperaturas_train_box4, diss_train_box4], axis=1),\
                                                                pd.concat([X_scaled_hum, humedad_train_box4, diss_train_box4],axis=1), pd.concat([X_scaled_co2, co2_train_box4, diss_train_box4], axis=1)
                    X_test1_temp, X_test1_hum, X_test1_co2 = pd.concat([X_test1_temp, temperaturas_test_box4, diss_test_box4], axis=1), \
                                                             pd.concat([X_test1_hum, humedad_test_box4, diss_test_box4],axis=1), pd.concat([X_test1_co2, co2_test_box4, diss_test_box4], axis=1)
                    X_val1_temp, X_val1_hum, X_val1_co2 = pd.concat([X_val1_temp, temperaturas_val_box4, diss_val_box4],axis=1), \
                                                          pd.concat([X_val1_hum, humedad_val_box4, diss_val_box4],axis=1), pd.concat([X_val1_co2, co2_val_box4, diss_val_box4], axis=1)

                if box5 == 1:
                    X_scaled_temp, X_scaled_hum, X_scaled_co2 = pd.concat([X_scaled_temp, temperaturas_train_box5, diss_train_box5], axis=1),\
                                                                pd.concat([X_scaled_hum, humedad_train_box5, diss_train_box5],axis=1), pd.concat([X_scaled_co2, co2_train_box5, diss_train_box5], axis=1)
                    X_test1_temp, X_test1_hum, X_test1_co2 = pd.concat([X_test1_temp, temperaturas_test_box5, diss_test_box5], axis=1), \
                                                             pd.concat([X_test1_hum, humedad_test_box5, diss_test_box5],axis=1), pd.concat([X_test1_co2, co2_test_box5, diss_test_box5], axis=1)
                    X_val1_temp, X_val1_hum, X_val1_co2 = pd.concat([X_val1_temp, temperaturas_val_box5, diss_val_box5],axis=1), \
                                                          pd.concat([X_val1_hum, humedad_val_box5, diss_val_box5],axis=1), pd.concat([X_val1_co2, co2_val_box5, diss_val_box5], axis=1)
                if box6 == 1:
                    X_scaled_temp, X_scaled_hum, X_scaled_co2 = pd.concat([X_scaled_temp, temperaturas_train_box6, diss_train_box6], axis=1),\
                                                                pd.concat([X_scaled_hum, humedad_train_box6, diss_train_box6],axis=1), pd.concat([X_scaled_co2, co2_train_box6, diss_train_box6], axis=1)
                    X_test1_temp, X_test1_hum, X_test1_co2 = pd.concat([X_test1_temp, temperaturas_test_box6, diss_test_box6], axis=1), \
                                                             pd.concat([X_test1_hum, humedad_test_box6, diss_test_box6],axis=1), pd.concat([X_test1_co2, co2_test_box6, diss_test_box6], axis=1)
                    X_val1_temp, X_val1_hum, X_val1_co2 = pd.concat([X_val1_temp, temperaturas_val_box6, diss_val_box6],axis=1), \
                                                          pd.concat([X_val1_hum, humedad_val_box6, diss_val_box6],axis=1), pd.concat([X_val1_co2, co2_val_box6, diss_val_box6], axis=1)
                if box7 == 1:
                    X_scaled_temp, X_scaled_hum, X_scaled_co2 = pd.concat([X_scaled_temp, temperaturas_train_box7, diss_train_box7], axis=1),\
                                                                pd.concat([X_scaled_hum, humedad_train_box7, diss_train_box7],axis=1), pd.concat([X_scaled_co2, co2_train_box7, diss_train_box7], axis=1)
                    X_test1_temp, X_test1_hum, X_test1_co2 = pd.concat([X_test1_temp, temperaturas_test_box7, diss_test_box7], axis=1), \
                                                             pd.concat([X_test1_hum, humedad_test_box7, diss_test_box7],axis=1), pd.concat([X_test1_co2, co2_test_box7, diss_test_box7], axis=1)
                    X_val1_temp, X_val1_hum, X_val1_co2 = pd.concat([X_val1_temp, temperaturas_val_box7, diss_val_box7],axis=1), \
                                                          pd.concat([X_val1_hum, humedad_val_box7, diss_val_box7],axis=1), pd.concat([X_val1_co2, co2_val_box7, diss_val_box7], axis=1)
                if box8 == 1:
                    X_scaled_temp, X_scaled_hum, X_scaled_co2 = pd.concat([X_scaled_temp, temperaturas_train_box8, diss_train_box8], axis=1),\
                                                                pd.concat([X_scaled_hum, humedad_train_box8, diss_train_box8],axis=1), pd.concat([X_scaled_co2, co2_train_box8, diss_train_box8], axis=1)
                    X_test1_temp, X_test1_hum, X_test1_co2 = pd.concat([X_test1_temp, temperaturas_test_box8, diss_test_box8], axis=1), \
                                                             pd.concat([X_test1_hum, humedad_test_box8, diss_test_box8],axis=1), pd.concat([X_test1_co2, co2_test_box8, diss_test_box8], axis=1)
                    X_val1_temp, X_val1_hum, X_val1_co2 = pd.concat([X_val1_temp, temperaturas_val_box8, diss_val_box8],axis=1), \
                                                          pd.concat([X_val1_hum, humedad_val_box8, diss_val_box8],axis=1), pd.concat([X_val1_co2, co2_val_box8, diss_val_box8], axis=1)


                miss = X_scaled_temp.apply(lambda x: x.count(), axis=1) - X_scaled_temp.shape[1]
                miss = np.where(miss <= -6)[0]
                if len(miss)>0:
                    X_scaled_temp = X_scaled_temp.drop(miss, axis=0)
                    X_scaled_temp = X_scaled_temp.reset_index(drop=True)
                    y_scaled_temp = y_scaled_temp.drop(miss, axis=0)
                    y_scaled_temp = y_scaled_temp.reset_index(drop=True)
                miss = X_scaled_hum.apply(lambda x: x.count(), axis=1) - X_scaled_hum.shape[1]
                miss = np.where(miss <= -6)[0]
                if len(miss) > 0:
                    X_scaled_hum = X_scaled_hum.drop(miss, axis=0)
                    X_scaled_hum = X_scaled_hum.reset_index(drop=True)
                    y_scaled_hum = y_scaled_hum.drop(miss, axis=0)
                    y_scaled_hum = y_scaled_hum.reset_index(drop=True)
                miss = X_scaled_co2.apply(lambda x: x.count(), axis=1) - X_scaled_co2.shape[1]
                miss = np.where(miss <= -6)[0]
                if len(miss) > 0:
                    X_scaled_co2 = X_scaled_co2.drop(miss, axis=0)
                    X_scaled_co2 = X_scaled_co2.reset_index(drop=True)
                    y_scaled_co2 = y_scaled_co2.drop(miss, axis=0)
                    y_scaled_co2 = y_scaled_co2.reset_index(drop=True)

                miss = X_test1_temp.apply(lambda x: x.count(), axis=1) - X_test1_temp.shape[1]
                miss = np.where(miss <= -6)[0]
                if len(miss) > 0:
                    X_test1_temp = X_test1_temp.drop(miss, axis=0)
                    X_test1_temp = X_test1_temp.reset_index(drop=True)
                    y_test1_temp = y_test1_temp.drop(miss, axis=0)
                    y_test1_temp = y_test1_temp.reset_index(drop=True)
                miss = X_test1_hum.apply(lambda x: x.count(), axis=1) - X_test1_hum.shape[1]
                miss = np.where(miss <= -6)[0]
                if len(miss) > 0:
                    X_test1_hum = X_test1_hum.drop(miss, axis=0)
                    X_test1_hum = X_test1_hum.reset_index(drop=True)
                    y_test1_hum = y_test1_hum.drop(miss, axis=0)
                    y_test1_hum = y_test1_hum.reset_index(drop=True)
                miss = X_test1_co2.apply(lambda x: x.count(), axis=1) - X_test1_co2.shape[1]
                miss = np.where(miss <= -6)[0]
                if len(miss) > 0:
                    X_test1_co2 = X_test1_co2.drop(miss, axis=0)
                    X_test1_co2 = X_test1_co2.reset_index(drop=True)
                    y_test1_co2 = y_test1_co2.drop(miss, axis=0)
                    y_test1_co2 = y_test1_co2.reset_index(drop=True)
                miss = X_val1_temp.apply(lambda x: x.count(), axis=1) - X_val1_temp.shape[1]
                miss = np.where(miss <= -6)[0]
                if len(miss) > 0:
                    X_val1_temp = X_val1_temp.drop(miss, axis=0)
                    X_val1_temp = X_val1_temp.reset_index(drop=True)
                    y_val1_temp = y_val1_temp.drop(miss, axis=0)
                    y_val1_temp = y_val1_temp.reset_index(drop=True)
                miss = X_val1_hum.apply(lambda x: x.count(), axis=1) - X_val1_hum.shape[1]
                miss = np.where(miss <= -6)[0]
                if len(miss) > 0:
                    X_val1_hum = X_val1_hum.drop(miss, axis=0)
                    X_val1_hum = X_val1_hum.reset_index(drop=True)
                    y_val1_hum = y_val1_hum.drop(miss, axis=0)
                    y_val1_hum = y_val1_hum.reset_index(drop=True)
                miss = X_val1_co2.apply(lambda x: x.count(), axis=1) - X_val1_co2.shape[1]
                miss = np.where(miss <= -6)[0]
                if len(miss) > 0:
                    X_val1_co2 = X_val1_co2.drop(miss, axis=0)
                    X_val1_co2 = X_val1_co2.reset_index(drop=True)
                    y_val1_co2 = y_val1_co2.drop(miss, axis=0)
                    y_val1_co2 = y_val1_co2.reset_index(drop=True)



                X_scaled_co2 = X_scaled_co2.reset_index(drop=True)
                X_scaled_temp = X_scaled_temp.reset_index(drop=True)
                X_scaled_hum = X_scaled_hum.reset_index(drop=True)
                y_scaled_co2 = y_scaled_co2.reset_index(drop=True)
                y_scaled_temp = y_scaled_temp.reset_index(drop=True)
                y_scaled_hum = y_scaled_hum.reset_index(drop=True)

                X_scaled_temp.columns = range(X_scaled_temp.shape[1])
                X_scaled_hum.columns = range(X_scaled_hum.shape[1])
                X_scaled_co2.columns = range(X_scaled_co2.shape[1])
                X_test1_temp.columns = range(X_test1_temp.shape[1])
                X_test1_hum.columns = range(X_test1_hum.shape[1])
                X_test1_co2.columns = range(X_test1_co2.shape[1])
                X_val1_temp.columns = range(X_val1_temp.shape[1])
                X_val1_hum.columns = range(X_val1_hum.shape[1])
                X_val1_co2.columns = range(X_val1_co2.shape[1])

                ########################################################################
                for t in range(X_scaled_temp.shape[1]):
                    a = X_scaled_temp.iloc[:, t]
                    b = X_scaled_hum.iloc[:, t]
                    if len(np.where(np.isnan(a))[0]) > 0:
                        a[np.where(np.isnan(a))[0]] = np.repeat(-10, len(np.where(np.isnan(a))[0]))
                        X_scaled_temp.iloc[:, t] = a
                    if len(np.where(np.isnan(b))[0]) > 0:
                        b[np.where(np.isnan(b))[0]] = np.repeat(-10, len(np.where(np.isnan(b))[0]))
                        X_scaled_hum.iloc[:, t] = b
                for t in range(X_scaled_co2.shape[1]):
                    a = X_scaled_co2.iloc[:, t]
                    if len(np.where(np.isnan(a))[0]) > 0:
                        a[np.where(np.isnan(a))[0]] = np.repeat(-10, len(np.where(np.isnan(a))[0]))
                        X_scaled_co2.iloc[:, t] = a
                ####################################################################################
                for t in range(X_test1_temp.shape[1]):
                    a = X_test1_temp.iloc[:, t]
                    b = X_test1_hum.iloc[:, t]
                    if len(np.where(np.isnan(a))[0]) > 0:
                        a[np.where(np.isnan(a))[0]] = np.repeat(-10, len(np.where(np.isnan(a))[0]))
                        X_test1_temp.iloc[:, t] = a
                    if len(np.where(np.isnan(b))[0]) > 0:
                        b[np.where(np.isnan(b))[0]] = np.repeat(-10, len(np.where(np.isnan(b))[0]))
                        X_test1_hum.iloc[:, t] = b
                for t in range(X_test1_co2.shape[1]):
                    a = X_test1_co2.iloc[:, t]
                    if len(np.where(np.isnan(a))[0]) > 0:
                        a[np.where(np.isnan(a))[0]] = np.repeat(-10, len(np.where(np.isnan(a))[0]))
                        X_test1_co2.iloc[:, t] = a
                ###################################################################################
                for t in range(X_val1_temp.shape[1]):
                    a = X_val1_temp.iloc[:, t]
                    b = X_val1_hum.iloc[:, t]
                    if len(np.where(np.isnan(a))[0]) > 0:
                        a[np.where(np.isnan(a))[0]] = np.repeat(-10, len(np.where(np.isnan(a))[0]))
                        X_val1_temp.iloc[:, t] = a
                    if len(np.where(np.isnan(b))[0]) > 0:
                        b[np.where(np.isnan(b))[0]] = np.repeat(-10, len(np.where(np.isnan(b))[0]))
                        X_val1_hum.iloc[:, t] = b
                for t in range(X_val1_co2.shape[1]):
                    a = X_val1_co2.iloc[:, t]
                    if len(np.where(np.isnan(a))[0]) > 0:
                        a[np.where(np.isnan(a))[0]] = np.repeat(-10, len(np.where(np.isnan(a))[0]))
                        X_val1_co2.iloc[:, t] = a


                out_train = np.where(np.isnan(y_scaled_temp))[0]
                if len(out_train) > 0:
                    y_scaled_temp = y_scaled_temp.drop(out_train, axis=0)
                    y_scaled_temp = y_scaled_temp.reset_index(drop=True)
                    X_scaled_temp = X_scaled_temp.drop(out_train, axis=0)
                    X_scaled_temp = X_scaled_temp.reset_index(drop=True)
                out_train = np.where(np.isnan(y_scaled_hum))[0]
                if len(out_train) > 0:
                    y_scaled_hum= y_scaled_hum.drop(out_train, axis=0)
                    y_scaled_hum= y_scaled_hum.reset_index(drop=True)
                    X_scaled_hum= X_scaled_hum.drop(out_train, axis=0)
                    X_scaled_hum= X_scaled_hum.reset_index(drop=True)
                out_train = np.where(np.isnan(y_scaled_co2))[0]
                if len(out_train) > 0:
                    y_scaled_co2= y_scaled_co2.drop(out_train, axis=0)
                    y_scaled_co2= y_scaled_co2.reset_index(drop=True)
                    X_scaled_co2= X_scaled_co2.drop(out_train, axis=0)
                    X_scaled_co2= X_scaled_co2.reset_index(drop=True)
                ################################################################################
                out_T = np.where(np.isnan(y_test1_temp))[0]
                if len(out_T) > 0:
                    y_test1_temp = y_test1_temp.drop(out_T, axis=0)
                    y_test1_temp = y_test1_temp.reset_index(drop=True)
                    X_test1_temp = X_test1_temp.drop(out_T, axis=0)
                    X_test1_temp = X_test1_temp.reset_index(drop=True)
                out_T = np.where(np.isnan(y_test1_hum))[0]
                if len(out_T) > 0:
                    y_test1_hum = y_test1_hum.drop(out_T, axis=0)
                    y_test1_hum = y_test1_hum.reset_index(drop=True)
                    X_test1_hum = X_test1_hum.drop(out_T, axis=0)
                    X_test1_hum = X_test1_hum.reset_index(drop=True)
                out_T = np.where(np.isnan(y_test1_co2))[0]
                if len(out_T) > 0:
                    y_test1_co2 = y_test1_co2.drop(out_T, axis=0)
                    y_test1_co2 = y_test1_co2.reset_index(drop=True)
                    X_test1_co2 = X_test1_co2.drop(out_T, axis=0)
                    X_test1_co2 = X_test1_co2.reset_index(drop=True)
                ###################################################################################
                out_T = np.where(np.isnan(y_val1_temp))[0]
                if len(out_T) > 0:
                    y_val1_temp = y_val1_temp.drop(out_T, axis=0)
                    y_val1_temp = y_val1_temp.reset_index(drop=True)
                    X_val1_temp = X_val1_temp.drop(out_T, axis=0)
                    X_val1_temp = X_val1_temp.reset_index(drop=True)
                    out_T = np.where(np.isnan(y_val1_temp))[0]
                out_T = np.where(np.isnan(y_val1_hum))[0]
                if len(out_T) > 0:
                    y_val1_hum = y_val1_hum.drop(out_T, axis=0)
                    y_val1_hum = y_val1_hum.reset_index(drop=True)
                    X_val1_hum = X_val1_hum.drop(out_T, axis=0)
                    X_val1_hum = X_val1_hum.reset_index(drop=True)
                out_T = np.where(np.isnan(y_val1_co2))[0]
                if len(out_T) > 0:
                    y_val1_co2 = y_val1_co2.drop(out_T, axis=0)
                    y_val1_co2 = y_val1_co2.reset_index(drop=True)
                    X_val1_co2 = X_val1_co2.drop(out_T, axis=0)
                    X_val1_co2 = X_val1_co2.reset_index(drop=True)
                ####################################################################

                X_scaled_temp, X_scaled_hum, X_scaled_co2 = np.array(X_scaled_temp),np.array(X_scaled_hum),np.array(X_scaled_co2)
                X_test1_temp, X_test1_hum, X_test1_co2 = np.array(X_test1_temp),np.array(X_test1_hum),np.array(X_test1_co2)

                y_scaled_temp, y_scaled_hum, y_scaled_co2 = np.array(y_scaled_temp),np.array(y_scaled_hum),np.array(y_scaled_co2)
                y_test1_temp, y_test1_hum, y_test1_co2 = np.array(y_test1_temp),np.array(y_test1_hum),np.array(y_test1_co2)

                X_val1_temp, X_val1_hum, X_val1_co2 = np.array(X_val1_temp), np.array(X_val1_hum), np.array(X_val1_co2)
                y_val1_temp, y_val1_hum, y_val1_co2 = np.array(y_val1_temp), np.array(y_val1_hum), np.array(y_val1_co2)

                evallist_temp = [(X_test1_temp, y_test1_temp)]
                evallist_hum = [(X_test1_hum, y_test1_hum)]
                evallist_co2 = [(X_test1_co2, y_test1_co2)]

                model_temp = xgb.XGBRegressor(max_depth=m_depth_temp, eta=etaz_temp)
                model_hum = xgb.XGBRegressor(max_depth=m_depth_hum, eta=etaz_hum)
                model_co2 = xgb.XGBRegressor(max_depth=m_depth_co2, eta=etaz_co2)

                if len(y_val1_temp)>0 and X_test1_temp.shape[0]>1 and X_scaled_temp.shape[0]>1:
                    model_temp = model_temp.fit(X_scaled_temp, y_scaled_temp,early_stopping_rounds=10,eval_metric="rmse", eval_set=evallist_temp, verbose=False)
                    y_pred_temp = model_temp.predict(X_val1_temp)
                    xgb1_temp = scalar_temp.inverse_transform(y_pred_temp.reshape(len(y_val1_temp), 1))
                    errors_temp[j] = 100 * (np.sqrt(metrics.mean_squared_error(y_val1_temp, xgb1_temp)) / med_temp)
                else:
                    errors_temp[j] = np.nan

                if len(y_val1_hum)>0 and X_test1_hum.shape[0]>1 and X_scaled_hum.shape[0]>1:
                    model_hum = model_hum.fit(X_scaled_hum, y_scaled_hum,early_stopping_rounds=10,eval_metric="rmse", eval_set=evallist_hum, verbose=False)
                    y_pred_hum = model_hum.predict(X_val1_hum)
                    xgb1_hum = scalar_hum.inverse_transform(y_pred_hum.reshape(len(y_val1_hum), 1))
                    errors_hum[j] = 100 * (np.sqrt(metrics.mean_squared_error(y_val1_hum, xgb1_hum)) / med_hum)
                else:
                    errors_hum[j]=np.nan

                if len(y_val1_co2)>0 and X_test1_co2.shape[0]>1 and X_scaled_co2.shape[0]>1:
                    model_co2 = model_co2.fit(X_scaled_co2, y_scaled_co2,early_stopping_rounds=10,eval_metric="rmse", eval_set=evallist_co2, verbose=False)
                    y_pred_co2 = model_co2.predict(X_val1_co2)
                    xgb1_co2 = scalar_co2.inverse_transform(y_pred_co2.reshape(len(y_val1_co2), 1))
                    errors_co2[j] = 100 * (np.sqrt(metrics.mean_squared_error(y_val1_co2, xgb1_co2)) / med_co2)
                else:
                    errors_co2[j]=np.nan

            else:
                errors_co2[j] = np.nan
                errors_temp[j] = np.nan
                errors_hum[j] = np.nan
        else:
            errors_co2[j]=np.nan
            errors_temp[j]=np.nan
            errors_hum[j]=np.nan


    accuracy_temp = np.nanmean(np.array(errors_temp, dtype=float))
    accuracy_hum = np.nanmean(np.array(errors_hum, dtype=float))
    accuracy_co2 = np.nanmean(np.array(errors_co2, dtype=float))
    compx = np.sum([box1, box2, box3, box4, box5, box6, box7, box8])

    dictionary[name1] = accuracy_temp, accuracy_hum, accuracy_co2,compx
    return accuracy_temp, accuracy_hum, accuracy_co2,compx


class MyRepair(Repair):

    def _do(self, problem, pop, **kwargs):
        for k in range(len(pop)):
            x = pop[k].X
            x2 = np.delete(x,[0,1,2,3,4,5])
            if np.sum(x2) ==2:
                a = np.where(x2==0)[0]
                x2[a[0]]=1
            elif np.sum(x2) ==1 :
                a = np.where(x2==0)[0]
                x2[a[0]]=1
                x2[a[1]]=1
            elif np.sum(x2) ==0 :
                a = np.where(x2==0)[0]
                x2[a[0]]=1
                x2[a[1]]=1
                x2[a[2]]=1
            else:
                x2=x2

        x = np.concatenate([x, x2])
        return pop


class MyProblem(Problem):

    def __init__(self, X1, y_temp, y_hum, y_co2, med_temp, med_hum, med_co2, contador, **kwargs):
        self.X1 = X1
        self.y_temp = y_temp
        self.y_hum = y_hum
        self.y_co2 = y_co2
        self.med_temp = med_temp
        self.med_hum = med_hum
        self.med_co2 = med_co2
        self.contador = contador
        super().__init__(n_var=14,
                         n_obj=4,
                         n_constr=1,
                         xl=np.array([0,1,0,1,0,1,0, 0, 0, 0, 0, 0, 0, 0]),
                         xu=np.array([10,10,10,10,10,10, 1, 1, 1, 1, 1, 1, 1,1]),
                         type_var=np.int,
                         elementwise_evaluation=True,
                         **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        X1 = self.X1
        g1 = -np.sum(np.delete(x,[0,1,2,3,4,5])) +3
        out["G"] = g1
        print(x)

        print(
            '\n ############################################## \n ############################# \n ########################## EvaluaciÃ³n ',
            self.contador, '\n #########################')

        f1, f2, f3,f4 = XGB(X1, self.y_temp, self.y_hum, self.y_co2,self.med_temp, self.med_hum, self.med_co2, int(x[0])*10+1,int(x[1])*0.1,int(x[2])*10+1,int(x[3])*0.1,int(x[4])*10+1,int(x[5])*0.1, int(x[6]), int(x[7]),
              int(x[8]), int(x[9]),int(x[10]), int(x[11]), int(x[12]), int(x[13]))

        self.contador[0] += 1
        out["F"] = np.column_stack([f1, f2, f3,f4])



def nsga2_individual(X, y_temp, y_hum, y_co2, med_temp, med_hum, med_co2, contador):
    problem = MyProblem(X, y_temp, y_hum, y_co2, med_temp, med_hum, med_co2, contador)
    ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=10)
    algorithm = NSGA3(pop_size=60,repair=MyRepair(), eliminate_duplicates=True,
                      sampling=get_sampling("int_random"),
                      ref_dirs=ref_dirs,
                      # sampling =g,
                      # crossover=0.9,
                      # mutation=0.1)
                      crossover=get_crossover("int_sbx"),
                      mutation=get_mutation("int_pm"),
                      save_history = True)
    res = minimize(problem,
                   algorithm,
                   termination=('n_gen', 20 ),
                   # ("n_gen", 20),
                   pf=True,
                   verbose=True,
                   save_history=True,
                   seed=7)

    #running = RunningMetric(delta_gen=1,
    #                        n_plots=5,
    #                        only_if_n_plots=True,
    #                        key_press=False,
    #                        do_show=True)
#
    #for algorithm in res.history:
    #    running.notify(algorithm)
#
#
    #n_evals = np.array([e.evaluator.n_eval for e in res.history])
    #opt = np.array([e.opt[0].F for e in res.history])
#
    #PCP().add(res.F).show()
#
    #from pymoo.visualization.heatmap import Heatmap
    #Heatmap().add(res.F).show()
#
#
    #plt.title("Convergence")
    #plt.plot(n_evals, opt, "--")
    #plt.yscale("log")
    #plt.show()
#
    #plot = Scatter()
    #plot.add(res.F, color="red")
    #plot.show()

    obj_T = res.F
    struct_T = res.X
    print(dictionary)
    return (obj_T, struct_T, res)


def MTI_train(DATA, date_init,  names, contador):
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
        elif any(t == np.array([0, 1, 2, 3])):
            a = DATA_C.iloc[:, t]
            y_smooth = a.rolling(window=5, min_periods=4)
            y_smooth = y_smooth.mean()
            DATA_C.iloc[:, t] = y_smooth
        else:
            a = DATA_C.iloc[:, t]
            y_smooth = a.rolling(window=5, min_periods=4)
            y_smooth = y_smooth.mean()
            DATA_C.iloc[:, t] = y_smooth

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

    X = dd
    D1 = DATA_C

    carrito = X.iloc[:, range(4)]
    temp = carrito.iloc[:, 1]
    hum = carrito.iloc[:, 0]
    co2 = carrito.iloc[:, 3]
    press = carrito.iloc[:, 2]
    temp = temp.reset_index(drop=True)
    hum = hum.reset_index(drop=True)
    co2 = co2.reset_index(drop=True)
    press = press.reset_index(drop=True)

    # names = np.delete(names, np.array([0, 1, 2, 3]))
    # Crreción variable altura carrito
    X.iloc[:, 6] = (X.iloc[:, 6] + 550) / 1000
    D1.iloc[:, 6] = (D1.iloc[:, 6] + 550) / 1000

    y_temp = pd.DataFrame(temp).reset_index(drop=True)
    y_hum = pd.DataFrame(hum).reset_index(drop=True)
    y_co2 = pd.DataFrame(co2).reset_index(drop=True)

    X = X.reset_index(drop=True)
    yearday = X.iloc[:, 36]
    posis = pd.concat([X.iloc[:, 4], X.iloc[:, 5]], axis=1)
    tt = pd.concat([X.iloc[:, np.array([4, 7, 10, 13, 16, 19, 22, 25]) + 4]])
    tt_co2 = pd.concat([X.iloc[:, np.array([5, 8, 11, 14, 17, 20, 23, 26]) + 4]])
    pos_cajas_x = np.array([50.1, 0, 47.1, 4, 20.6, 30.6, 44, 11.5])
    pos_cajas_y = np.array([3, 16, 14.4, 0, 9.4, 15.7, 6.3, 7.8])
    pos_cajas_z = np.array([3.2, 2.4, 3.2, 3, 2.2, 2.5, 2.1, 1.9])
    pos_cajas = pd.concat([pd.DataFrame(pos_cajas_x), pd.DataFrame(pos_cajas_y)], axis=1)
    pos_cajasT = pd.concat([pd.DataFrame(pos_cajas_x), pd.DataFrame(pos_cajas_y), pd.DataFrame(pos_cajas_z)], axis=1)

    ###########################################################################################
    ####################################################################################################################
    # Training correction
    ####################################################################################################################
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

    # yy = pd.concat([y_co21, y_co22], axis=0).reset_index(drop=True)
    y_co2 = pd.DataFrame(y_co2_init).reset_index(drop=True)

    D1.iloc[:, 3] = y_co2

    stop = np.where(yearday == 323)[0][0]
    D1 = D1.drop(range(stop), axis=0)
    D1 = D1.reset_index(drop=True)
    # dd= dd.drop(range(stop), axis=0)
    # dd= dd.reset_index(drop=True)
    yearday = yearday.drop(range(stop), axis=0)
    yearday = yearday.reset_index(drop=True)

    carrito = D1.iloc[:, range(4)]
    temp_I = carrito.iloc[:, 1]
    hum_I = carrito.iloc[:, 0]
    co2_I = carrito.iloc[:, 3]
    press_I = carrito.iloc[:, 2]

    X_final = D1
    X_final = X_final.reset_index(drop=True)
    X_final = X_final.drop(X_final.columns[range(4)], axis=1)
    names = np.delete(names, np.array([0, 1, 2, 3]))

    dayofyear = X_final.iloc[:, 32]
    X_final = X_final.drop(X_final.columns[33], axis=1)

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

    med_temp = np.nanmean(temp_I)
    med_hum = np.nanmean(hum_I)
    med_co2 = np.nanmean(co2_I)

    x_final = X_final.reset_index(drop=True)
    y_temp = temp_I
    y_hum = hum_I
    y_co2 = co2_I
    y_temp = y_temp.reset_index(drop=True)
    y_hum = y_hum.reset_index(drop=True)
    y_co2 = y_co2.reset_index(drop=True)

    yy1 = np.where((x_final.iloc[:, 0] == 6.9) | (x_final.iloc[:, 0] == 26))[0]
    yy2 = np.where((x_final.iloc[:, 1] == 4) | (x_final.iloc[:, 1] == 14.55))[0]

    yy3 = np.where((x_final.iloc[:, 0] == 46.3) | (x_final.iloc[:, 0] == 28.8))[0]
    yy4 = np.where((x_final.iloc[:, 1] == 7.6) | (x_final.iloc[:, 1] == 10.1))[0]
    #yy3 = np.where((x_final.iloc[:, 0] == 28.8) & (x_final.iloc[:, 1] == 10.1))[0]

    zz1 = np.intersect1d(yy1, yy2)
    zz2 = np.intersect1d(yy3, yy4)
    zz1 = np.sort(np.concatenate((zz1, zz2)))

    y_temp = y_temp.drop(zz1, axis=0)
    y_hum = y_hum.drop(zz1, axis=0)
    y_co2 = y_co2.drop(zz1, axis=0)
    y_temp = y_temp.reset_index(drop=True)
    y_hum = y_hum.reset_index(drop=True)
    y_co2 = y_co2.reset_index(drop=True)

    x_final = x_final.drop(zz1, axis=0)
    x_final = x_final.reset_index(drop=True)


    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    #######################################################################################################
    temp_obj_T, temp_x_T, res = nsga2_individual(x_final, y_temp, y_hum, y_co2, med_temp, med_hum, med_co2, contador)


    print("Estructuras factibles temperatura", temp_x_T, "y objetivos", temp_obj_T)
    #print("Estructura optima temperatura", temp_x, "objetivo", temp_obj)
    return (temp_obj_T, temp_x_T, res)


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
# time_end = datetime.datetime(2021, 1, 20, 23, 0, 0)
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
variables_meteo = loading(['humidity','radiation','temperature'], 'meteo', 'none', variables, time_end)
names_meteo=['humidity_M','radiation_M','temperature_M']
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
variables.columns = np.concatenate([names_52, pos, names_02, names_09, names_1a, names_3d, names_B1, names_B2, names_B3, names_B4, names_meteo])
names = variables.columns
############################################################################################

############################################################################################
# Parameters definition
date_init = '2020-06-17 11:00:00'
#multiprocessing.Manager()
#multiprocessing.Manager()
#manager = multiprocessing.Manager()
#dictionary = manager.dict()
#contador = manager.list()
dictionary = dict()
contador = list()
contador.append(0)
# General function to train and save the models
# temp_obj_T, temp_x_T,hum_obj_T, hum_x_T ,co2_obj_T, co2_x_T,temp_obj, temp_x,hum_obj, hum_x,co2_obj, co2_x = MTI_train(variables, date_init, ntest, names)
temp_obj_T, temp_x_T, res = MTI_train(variables, date_init, names, contador)

print('El nÃºmero de evaluaciones totales fue', contador)
np.savetxt('temp_obj_total2.txt', temp_obj_T)
np.savetxt('temp_X_total2.txt', temp_x_T)
#np.savetxt('temp_obj2.txt', temp_obj)
#np.savetxt('temp_X2.txt', temp_x)
# np.savetxt('result.txt',res)

print(f"{round((time() - t) / 3600, 2)} h.")

