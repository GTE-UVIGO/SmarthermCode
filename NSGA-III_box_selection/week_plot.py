import os
import random
import sys
from numba import jit, cuda

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
GLOBAL_SEED = 6666

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
from scipy.stats import norm

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
def miss_funct(x1,y1, lim):
    miss = x1.apply(lambda x: x.count(), axis=1) - x1.shape[1]
    miss = np.where(miss < -lim)[0]
    if len(miss) > 0:
        x1 = x1.drop(miss, axis=0)
        x1 = x1.reset_index(drop=True)
        y1= y1.drop(miss, axis=0)
        y1 = y1.reset_index(drop=True)


    return(x1, y1)

def scalar_funct(x,mask):
    for t in range(x.shape[1]):
        a = x.iloc[:, t]
        if len(np.where(np.isnan(a))[0]) > 0:
            a[np.where(np.isnan(a))[0]] = np.repeat(mask, len(np.where(np.isnan(a))[0]))
            x.iloc[:, t] = a

    return(x)

#FUNCTIONS
def MBE(y_true, y_pred,med):
    '''
    Parameters:
        y_true (array): Array of observed values
        y_pred (array): Array of prediction values

    Returns:
        mbe (float): Biais score
    '''
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true = y_true.reshape(len(y_true),1)
    y_pred = y_pred.reshape(len(y_pred),1)
    #mbe = np.sum(y_true-y_pred)/np.sum(y_true)
    mbe = np.mean(y_true-y_pred)/med
    #mbe = mbe/np.mean(y_true)
    #print('MBE = ', mbe)
    return(mbe*100)

def cortes(x, D, zc):
    Y = np.zeros( (zc, round(D/zc)))
    i = 0
    s = 0
    while i <= D:
        if D-i < zc and D-i>0:
            #Y = np.delete(Y,s,1)
            Y= np.delete(Y, Y.shape[1]-1, 1)
            break
        elif D-i ==0:
            break
        else:
            Y[:,s]= x[i:(i+zc)][:,0]
            i= i + zc
            s = s +1
    return(Y)


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

######################################################################################################################
def loading(var_name, sensor_name, host, variables, time_end,diox):
    import influxdb_client
    from influxdb_client import InfluxDBClient, Point, Dialect
    from influxdb_client.client.write_api import SYNCHRONOUS
    #info client

    query_api = client.query_api()

    if host=='none':


        for u in range(len(var_name)):

            q = f'''
                      from(bucket: "sensors/data")|> range(start:2020-06-17T11:00:00.000Z, stop:{time_end}) 
                      |> filter(fn: (r) => r["_measurement"] ==  "{sensor_name}")
                                       |> filter(fn: (r) => r["_field"] == "{var_name[u]}")
                                       |> aggregateWindow(every: 10m, fn: mean, createEmpty: true)
                                       |> yield(name: "{var_name[u]}")

                  '''

            tables = query_api.query(q)

            values = [0 for x in range(len(list(tables[0])))]
            j = 0
            for table in tables:
                #print(table)
                for record in table.records:
                    #print(record.values)
                    values[j] = record.values['_value']
                    j += 1

            variables.append(values)

    else:

        for u in range(len(var_name)):
            if diox==True:

                q = f'''
                          from(bucket: "sensors/data")|> range(start:2020-06-17T11:00:00.000Z, stop:{time_end}) 
                          |> filter(fn: (r) => r["_measurement"] ==  "{sensor_name}")
                                           |> filter(fn: (r) => r["host"] == "{host}")
                                           |> filter(fn: (r) => r["_field"] == "{var_name[u]}")
                                           |> filter(fn: (r) => r["mhz14.autocalibrate"] == "False")
                                           |> aggregateWindow(every: 1m, fn: mean, createEmpty: true)
                                           |> yield(name: "{var_name[u]}")
    
                      '''
            else:
                q = f'''
                          from(bucket: "sensors/data")|> range(start:2020-06-17T11:00:00.000Z, stop:{time_end}) 
                          |> filter(fn: (r) => r["_measurement"] ==  "{sensor_name}")
                                           |> filter(fn: (r) => r["host"] == "{host}")
                                           |> filter(fn: (r) => r["_field"] == "{var_name[u]}")
                                           |> aggregateWindow(every: 1m, fn: mean, createEmpty: true)
                                           |> yield(name: "{var_name[u]}")

                      '''
            tables = query_api.query(q)

            values = [0 for x in range(len(list(tables[0])))]

            for table in tables:
                j = 0
                #print(table)
                for record in table.records:
                    #print(record.values)
                    values[j] = record.values['_value']
                    j += 1


            uu = np.where(np.array(values)==None)[0]
            vv=pd.Series(values)
            vv[uu]=9999
            values = list(vv)

            variables.append(values)


    return variables


#def loading_carrito(var_name, sensor_name, host, variables, time_end):
    #info influx
#
#    influx = influxdb.InfluxDBClient.from_dsn(CONN_STR)
#    influx.switch_database(DB_NAME)
#
#    place = ["sensor_data.autogen",sensor_name]
#    sep='.'
#    place = sep.join(place)
#    place2 = [sensor_name,"address"]
#    sep = '.'
#    place2 = sep.join(place2)
#    time_end_str = time_end.strftime('%Y-%m-%dT%H:%M:%SZ')
#
#    if host=='none':
#        for u in range(len(var_name)):
#
#            var2 = [var_name[u], 'vc']
#            sep = '_'
#            var2 = sep.join(var2)
#            query = f"""
#                            SELECT mean({var_name[u]}) AS {var_name[u]} FROM {place}
#                            WHERE time > '2020-06-17T11:00:00Z' AND time < '{time_end_str}' AND {var2}<3
#                              AND {place2} != '69' GROUP BY time(10m) FILL(9999)
#                        """
#
#            results = influx.query(query)
#
#            point = list(results)[0]
#            values = [0 for x in range(len(point))]
#            #dates = [0 for x in range(len(point))]
#            for t in range(len(point)):
#                values[t] = point[t][var_name[u]]
#                #dates[t] = point[t]['time']
#
#            variables.append(values)
#
#    else:
#
#        for u in range(len(var_name)):
#            query = f"""
#                            SELECT mean({var_name[u]}) AS {var_name[u]} FROM {place}
#                            WHERE time > '2020-06-17T11:00:00Z' AND time < '{time_end_str}'  AND "host"='{host}'
#                              AND {place2} != '69' GROUP BY time(1m) FILL(linear)
#                        """
#
#            results = influx.query(query)
#
#            point = list(results)[0]
#            values = [0 for x in range(len(point))]
#            #dates = [0 for x in range(len(point))]
#            for t in range(len(point)):
#                values[t]=point[t][var_name[u]]
#                #dates[t] = point[t]['time']
#
#            variables.append(values)
#
#    return variables

def nas_function(x):
    for j in range(x.shape[0]):
        # Possibility of missing values
        if any(np.array(np.isnan(x.iloc[j, :]))):
            ii = np.where(np.isnan(x.iloc[j, :]))[0]
            x.iloc[j, ii] = 0
    return(x)

def loading_carrito(var_name, sensor_name, host, variables, time_end, diox):
    import influxdb_client
    from influxdb_client import InfluxDBClient, Point, Dialect
    from influxdb_client.client.write_api import SYNCHRONOUS
    import interpolate

    #info client

    for u in range(len(var_name)):
        if diox==True:
            q = f'''
                      from(bucket: "sensors/data")|> range(start:2020-06-17T11:00:00.000Z, stop:{time_end}) 
                      |> filter(fn: (r) => r["_measurement"] ==  "{sensor_name}")
                                       |> filter(fn: (r) => r["host"] == "{host}")
                                       |> filter(fn: (r) => r["_field"] == "{var_name[u]}")
                                       |> filter(fn: (r) => r["mhz14.autocalibrate"] == "False")
                                       |> aggregateWindow(every: 1m, fn: mean, createEmpty: true)
                                       |> yield(name: "{var_name[u]}")

                  '''
        else:
            q = f'''
                      from(bucket: "sensors/data")|> range(start:2020-06-17T11:00:00.000Z, stop:{time_end}) 
                      |> filter(fn: (r) => r["_measurement"] ==  "{sensor_name}")
                                       |> filter(fn: (r) => r["host"] == "{host}")
                                       |> filter(fn: (r) => r["_field"] == "{var_name[u]}")
                                       |> aggregateWindow(every: 1m, fn: mean, createEmpty: true)
                                       |> yield(name: "{var_name[u]}")
                  '''
        tables = query_api.query(q)

        values = [0 for x in range(len(list(tables[0])))]

        for table in tables:
            #print(table)
            j = 0
            for record in table.records:
                #print(record.values)
                values[j] = record.values['_value']
                j += 1
        values = pd.Series(values).interpolate()
        uu = np.where(np.isnan(np.array(values)))[0]
        vv = pd.Series(values)
        vv[uu] = 9999
        values = list(vv)
        variables.append(values)
    return variables


def MTI_train_reduced(x_original, y_temp, y_hum, y_co2, med_temp, med_hum, med_co2, positX, positY,n_temp,
                       n_hum, n_co2):
    x_original = x_original.reset_index(drop=True)
    y_temp1 = y_temp.reset_index(drop=True)
    y_hum1 = y_hum.reset_index(drop=True)
    y_co21 = y_co2.reset_index(drop=True)


    P1 = np.where(x_original.iloc[:, 0] == positX)
    P2 = np.where(x_original.iloc[:, 1] == positY)
    #yyy = np.intersect1d(P1, P2)[0:1440*3+840]
    yyy = np.intersect1d(P1, P2)
    x_val = x_original.iloc[yyy]
    x_val = x_val.reset_index(drop=True)
    y_temp_val = y_temp1.iloc[yyy]
    y_hum_val = y_hum1.iloc[yyy]
    y_co2_val = y_co21.iloc[yyy]
    y_temp_val = y_temp_val.reset_index(drop=True)
    y_hum_val = y_hum_val.reset_index(drop=True)
    y_co2_val = y_co2_val.reset_index(drop=True)
    x_t = x_original.drop(yyy, axis=0)
    x_t = x_t.reset_index(drop=True)
    y_temp_t = y_temp1.drop(yyy, axis=0)
    y_hum_t = y_hum1.drop(yyy, axis=0)
    y_co2_t = y_co21.drop(yyy, axis=0)
    y_temp_t = y_temp_t.reset_index(drop=True)
    y_hum_t = y_hum_t.reset_index(drop=True)
    y_co2_t = y_co2_t.reset_index(drop=True)
    if x_val.shape[0] > 5:
        yy1 = np.where(x_t.iloc[:, 0] ==17.1)[0]
        yy2 = np.where(x_t.iloc[:, 1] == 14.2)[0]
        zz1 = np.intersect1d(yy1, yy2)
        x_test1 = x_t.iloc[zz1]
        x_test1 = x_test1.reset_index(drop=True)
        y_test_temp, y_test_hum, y_test_co2 = y_temp_t.iloc[zz1], y_hum_t.iloc[zz1], y_co2_t.iloc[zz1]
        y_test_temp, y_test_hum, y_test_co2 = y_test_temp.reset_index(drop=True), y_test_hum.reset_index(
            drop=True), y_test_co2.reset_index(drop=True)
        x_t = x_t.drop(zz1, axis=0)
        x_t = x_t.reset_index(drop=True)
        y_temp_t, y_hum_t, y_co2_t = y_temp_t.drop(zz1, axis=0), y_hum_t.drop(zz1, axis=0), y_co2_t.drop(zz1, axis=0)
        y_temp_t, y_hum_t, y_co2_t = y_temp_t.reset_index(drop=True), y_hum_t.reset_index(
            drop=True), y_co2_t.reset_index(drop=True)

        if x_test1.shape[0] > 5:
            x_test1 = x_test1.drop(x_test1.columns[np.array([0, 1, 2])], axis=1)
            x_t = x_t.drop(x_t.columns[np.array([0, 1, 2])], axis=1)
            x_val = x_val.drop(x_val.columns[np.array([0, 1, 2])], axis=1)
            names = x_t.columns

            temperaturas_train = np.array(
                x_t.copy().iloc[:, np.array([7, 10, 16, 26, 37, 40, 46])])
            temperaturas_test = np.array(
                x_test1.copy().iloc[:, np.array([7, 10, 16, 26, 37, 40, 46])])
            temperaturas_val = np.array(
                x_val.copy().iloc[:, np.array([7, 10, 16, 26, 37, 40, 46])])

            humedad_train = np.array(
                x_t.copy().iloc[:, np.array([6, 9, 15, 24, 36, 39, 45])])
            humedad_test = np.array(
                x_test1.copy().iloc[:, np.array([6, 9, 15, 24, 36, 39, 45])])
            humedad_val = np.array(
                x_val.copy().iloc[:, np.array([6, 9, 15, 24, 36, 39, 45])])

            co2_train = np.array(
                x_t.copy().iloc[:, np.array([8, 11, 17, 38, 41, 47])])
            co2_test = np.array(
                x_test1.copy().iloc[:, np.array([8, 11, 17, 38, 41, 47])])
            co2_val = np.array(
                x_val.copy().iloc[:, np.array([8, 11, 17, 38, 41, 47])])

            diss_train = np.array(x_t.copy().iloc[:, range(x_t.shape[1] - 9, x_t.shape[1])])
            diss_test = np.array(x_test1.copy().iloc[:, range(x_t.shape[1] - 9, x_t.shape[1])])
            # diss_test_2 = np.array(x_test1_2.iloc[:, range(x_t.shape[1] - 24, x_t.shape[1])])
            diss_val = np.array(x_val.copy().iloc[:, range(x_t.shape[1] - 9, x_t.shape[1])])
            rad_train = np.array(x_t.copy().iloc[:, np.array([25])])
            rad_test = np.array(x_test1.copy().iloc[:, np.array([25])])
            # rad_test_2 = np.array(x_test1_2.iloc[:, np.array([25])])
            rad_val = np.array(x_val.copy().iloc[:, np.array([25])])
            resto_train = x_t.copy().iloc[:, np.array([27, 28, 29])]
            resto_test = x_test1.copy().iloc[:, np.array([27, 28, 29])]
            # resto_test_2 = x_test1_2.iloc[:, np.array([27, 28, 29])]
            resto_val = x_val.copy().iloc[:, np.array([27, 28, 29])]

            scalar_temp = MinMaxScaler(feature_range=(0, 1))
            scalar_hum = MinMaxScaler(feature_range=(0, 1))
            scalar_co2 = MinMaxScaler(feature_range=(0, 1))
            scalardist = MinMaxScaler(feature_range=(0, 1))
            scalar_rad = MinMaxScaler(feature_range=(0, 1))
            scalarresto = MinMaxScaler(feature_range=(0, 1))

            scalar_temp.fit(np.concatenate((np.concatenate(temperaturas_train), np.concatenate(temperaturas_test),
                                            np.concatenate(temperaturas_val), np.array(y_temp_t),
                                            np.array(y_test_temp), np.array(y_temp_val))).reshape(-1, 1))
            scalardist.fit(np.concatenate(
                (np.concatenate(diss_train), np.concatenate(diss_test), np.concatenate(diss_val))).reshape(-1, 1))
            scalar_hum.fit(np.concatenate(
                (np.concatenate(humedad_train), np.concatenate(humedad_test), np.concatenate(humedad_val),
                 np.array(y_hum_t), np.array(y_test_hum), np.array(y_hum_val))).reshape(-1, 1))
            scalar_co2.fit(np.concatenate((np.concatenate(co2_train), np.concatenate(co2_test), np.concatenate(co2_val),
                                           np.array(y_co2_t), np.array(y_test_co2), np.array(y_co2_val))).reshape(-1,
                                                                                                                  1))
            scalar_rad.fit(
                np.concatenate((np.concatenate(rad_train), np.concatenate(rad_test), np.concatenate(rad_val))).reshape(
                    -1, 1))
            scalarresto.fit(pd.concat([resto_train, resto_test, resto_val], axis=0))
            y_scaled_temp = pd.DataFrame(scalar_temp.transform(np.array(y_temp_t.copy()).reshape(-1, 1)))
            y_scaled_hum = pd.DataFrame(scalar_hum.transform(np.array(y_hum_t.copy()).reshape(-1, 1)))
            y_scaled_co2 = pd.DataFrame(scalar_co2.transform(np.array(y_co2_t.copy()).reshape(-1, 1)))
            y_test1_temp = pd.DataFrame(scalar_temp.transform(np.array(y_test_temp.copy()).reshape(-1, 1)))
            y_test1_hum = pd.DataFrame(scalar_hum.transform(np.array(y_test_hum.copy()).reshape(-1, 1)))
            y_test1_co2 = pd.DataFrame(scalar_co2.transform(np.array(y_test_co2.copy()).reshape(-1, 1)))
            # y_test1_temp_2 = pd.DataFrame(scalar_temp.transform(np.array(y_test_temp_2).reshape(-1, 1)))
            # y_test1_hum_2 = pd.DataFrame(scalar_hum.transform(np.array(y_test_hum_2).reshape(-1, 1)))
            # y_test1_co2_2 = pd.DataFrame(scalar_co2.transform(np.array(y_test_co2_2).reshape(-1, 1)))

            temperaturas_train1 = np.zeros((temperaturas_train.shape[0], temperaturas_train.shape[1]))
            temperaturas_test1 = np.zeros((temperaturas_test.shape[0], temperaturas_test.shape[1]))
            # temperaturas_test1_2 = np.zeros((temperaturas_test_2.shape[0], temperaturas_test_2.shape[1]))
            temperaturas_val1 = np.zeros((temperaturas_val.shape[0], temperaturas_val.shape[1]))
            humedad_train1 = np.zeros((humedad_train.shape[0], humedad_train.shape[1]))
            humedad_test1 = np.zeros((humedad_test.shape[0], humedad_test.shape[1]))
            # humedad_test1_2 = np.zeros((humedad_test_2.shape[0], humedad_test_2.shape[1]))
            humedad_val1 = np.zeros((humedad_val.shape[0], humedad_val.shape[1]))
            for i in range(temperaturas_train.shape[1]):
                temperaturas_train1[:, i] = scalar_temp.transform(temperaturas_train[:, i].reshape(-1, 1))[:, 0]
                temperaturas_test1[:, i] = scalar_temp.transform(temperaturas_test[:, i].reshape(-1, 1))[:, 0]
                # temperaturas_test1_2[:, i] = scalar_temp.transform(temperaturas_test_2[:, i].reshape(-1, 1))[:, 0]
                temperaturas_val1[:, i] = scalar_temp.transform(temperaturas_val[:, i].reshape(-1, 1))[:, 0]
                humedad_train1[:, i] = scalar_hum.transform(humedad_train[:, i].reshape(-1, 1))[:, 0]
                humedad_test1[:, i] = scalar_hum.transform(humedad_test[:, i].reshape(-1, 1))[:, 0]
                # humedad_test1_2[:, i] = scalar_hum.transform(humedad_test_2[:, i].reshape(-1, 1))[:, 0]
                humedad_val1[:, i] = scalar_hum.transform(humedad_val[:, i].reshape(-1, 1))[:, 0]
            temperaturas_train1 = pd.DataFrame(temperaturas_train1)
            temperaturas_test1 = pd.DataFrame(temperaturas_test1)
            # temperaturas_test1_2 = pd.DataFrame(temperaturas_test1_2)
            temperaturas_val1 = pd.DataFrame(temperaturas_val1)
            humedad_train1 = pd.DataFrame(humedad_train1)
            humedad_test1 = pd.DataFrame(humedad_test1)
            # humedad_test1_2 = pd.DataFrame(humedad_test1_2)
            humedad_val1 = pd.DataFrame(humedad_val1)
            co2_train1 = np.zeros((co2_train.shape[0], co2_train.shape[1]))
            co2_test1 = np.zeros((co2_test.shape[0], co2_train.shape[1]))
            # co2_test1_2 = np.zeros((co2_test_2.shape[0], co2_test_2.shape[1]))
            co2_val1 = np.zeros((co2_val.shape[0], co2_train.shape[1]))
            for i in range(co2_train.shape[1]):
                co2_train1[:, i] = scalar_co2.transform(co2_train[:, i].reshape(-1, 1))[:, 0]
                co2_test1[:, i] = scalar_co2.transform(co2_test[:, i].reshape(-1, 1))[:, 0]
                # co2_test1_2[:, i] = scalar_co2.transform(co2_test_2[:, i].reshape(-1, 1))[:, 0]
                co2_val1[:, i] = scalar_co2.transform(co2_val[:, i].reshape(-1, 1))[:, 0]
            co2_train1 = pd.DataFrame(co2_train1)
            co2_test1 = pd.DataFrame(co2_test1)
            # co2_test1_2 = pd.DataFrame(co2_test1_2)
            co2_val1 = pd.DataFrame(co2_val1)
            diss_train1 = np.zeros((diss_train.shape[0], diss_train.shape[1]))
            diss_test1 = np.zeros((diss_test.shape[0], diss_train.shape[1]))
            # diss_test1_2 = np.zeros((diss_test_2.shape[0], diss_test_2.shape[1]))
            diss_val1 = np.zeros((diss_val.shape[0], diss_train.shape[1]))
            for i in range(diss_train.shape[1]):
                diss_train1[:, i] = scalardist.transform(diss_train[:, i].reshape(-1, 1))[:, 0]
                diss_test1[:, i] = scalardist.transform(diss_test[:, i].reshape(-1, 1))[:, 0]
                # diss_test1_2[:, i] = scalardist.transform(diss_test_2[:, i].reshape(-1, 1))[:, 0]
                diss_val1[:, i] = scalardist.transform(diss_val[:, i].reshape(-1, 1))[:, 0]
            rad_train1 = np.zeros((rad_train.shape[0], rad_train.shape[1]))
            rad_test1 = np.zeros((rad_test.shape[0], rad_train.shape[1]))
            # rad_test1_2 = np.zeros((rad_test_2.shape[0], rad_test_2.shape[1]))
            rad_val1 = np.zeros((rad_val.shape[0], rad_train.shape[1]))
            for i in range(rad_train.shape[1]):
                rad_train1[:, i] = scalar_rad.transform(rad_train[:, i].reshape(-1, 1))[:, 0]
                rad_test1[:, i] = scalar_rad.transform(rad_test[:, i].reshape(-1, 1))[:, 0]
                # rad_test1_2[:, i] = scalar_rad.transform(rad_test_2[:, i].reshape(-1, 1))[:, 0]
                rad_val1[:, i] = scalar_rad.transform(rad_val[:, i].reshape(-1, 1))[:, 0]
            diss_train1 = pd.DataFrame(diss_train1)
            diss_test1 = pd.DataFrame(diss_test1)
            # diss_test1_2 = pd.DataFrame(diss_test1_2)
            diss_val1 = pd.DataFrame(diss_val1)
            rad_train1 = pd.DataFrame(rad_train1)
            rad_test1 = pd.DataFrame(rad_test1)
            # rad_test1_2 = pd.DataFrame(rad_test1_2)
            rad_val1 = pd.DataFrame(rad_val1)
            resto_train1 = pd.DataFrame(scalarresto.transform(resto_train))
            resto_test1 = pd.DataFrame(scalarresto.transform(resto_test))
            # resto_test1_2 = pd.DataFrame(scalarresto.transform(resto_test_2))
            resto_val1 = pd.DataFrame(scalarresto.transform(resto_val))

            X_scaled_temp = pd.concat([temperaturas_train1, diss_train1, rad_train1, resto_train1], axis=1)
            X_test1_temp = pd.concat([temperaturas_test1, diss_test1, rad_test1, resto_test1], axis=1)
            # X_test1_temp_2 = pd.concat([temperaturas_test1_2,diss_test1_2 ,rad_test1_2, resto_test1_2], axis=1)
            X_val1_temp = pd.concat([temperaturas_val1, diss_val1, rad_val1, resto_val1], axis=1)
            X_scaled_hum = pd.concat([humedad_train1, diss_train1, rad_train1, resto_train1], axis=1)
            X_test1_hum = pd.concat([humedad_test1, diss_test1, rad_test1, resto_test1], axis=1)
            # X_test1_hum_2 = pd.concat([humedad_test1_2,diss_test1_2, rad_test1_2, resto_test1_2], axis=1)
            X_val1_hum = pd.concat([humedad_val1, diss_val1, rad_val1, resto_val1], axis=1)
            X_scaled_co2 = pd.concat([co2_train1, diss_train1, rad_train1, resto_train1], axis=1)
            X_test1_co2 = pd.concat([co2_test1, diss_test1, rad_test1, resto_test1], axis=1)
            # X_test1_co2_2 = pd.concat([co2_test1_2,diss_test1_2,rad_test1_2, resto_test1_2], axis=1)
            X_val1_co2 = pd.concat([co2_val1, diss_val1, rad_val1, resto_val1], axis=1)

            y_val1_temp = y_temp_val
            y_val1_hum = y_hum_val
            y_val1_co2 = y_co2_val

            X_scaled_temp, y_scaled_temp = miss_funct(X_scaled_temp, y_scaled_temp, 2)
            X_scaled_hum, y_scaled_hum = miss_funct(X_scaled_hum, y_scaled_hum, 2)
            X_scaled_co2, y_scaled_co2 = miss_funct(X_scaled_co2, y_scaled_co2, 2)
            X_test1_temp, y_test1_temp = miss_funct(X_test1_temp, y_test1_temp, 2)
            # X_test1_temp_2, y_test1_temp_2 =miss_funct(X_test1_temp_2, y_test1_temp_2,6)
            X_test1_hum, y_test1_hum = miss_funct(X_test1_hum, y_test1_hum, 2)
            # X_test1_hum_2, y_test1_hum_2 =miss_funct(X_test1_hum_2, y_test1_hum_2,6)
            X_test1_co2, y_test1_co2 = miss_funct(X_test1_co2, y_test1_co2, 2)
            ##X_test1_co2_2, y_test1_co2_2 =miss_funct(X_test1_co2_2, y_test1_co2_2,6)
            X_val1_temp, y_val1_temp = miss_funct(X_val1_temp, y_val1_temp, 2)
            X_val1_hum, y_val1_hum = miss_funct(X_val1_hum, y_val1_hum, 2)
            X_val1_co2, y_val1_co2 = miss_funct(X_val1_co2, y_val1_co2, 2)

            X_scaled_co2 = X_scaled_co2.reset_index(drop=True)
            X_scaled_temp = X_scaled_temp.reset_index(drop=True)
            X_scaled_hum = X_scaled_hum.reset_index(drop=True)
            X_val1_co2 = X_val1_co2.reset_index(drop=True)
            X_val1_temp = X_val1_temp.reset_index(drop=True)
            X_val1_hum = X_val1_hum.reset_index(drop=True)
            X_test1_co2 = X_test1_co2.reset_index(drop=True)
            X_test1_temp = X_test1_temp.reset_index(drop=True)
            X_test1_hum = X_test1_hum.reset_index(drop=True)
            # X_test1_co2_2 = X_test1_co2_2.reset_index(drop=True)
            # X_test1_temp_2 = X_test1_temp_2.reset_index(drop=True)
            # X_test1_hum_2 = X_test1_hum_2.reset_index(drop=True)

            y_scaled_co2 = y_scaled_co2.reset_index(drop=True)
            y_scaled_temp = y_scaled_temp.reset_index(drop=True)
            y_scaled_hum = y_scaled_hum.reset_index(drop=True)

            X_scaled_temp.columns = range(X_scaled_temp.shape[1])
            X_scaled_hum.columns = range(X_scaled_hum.shape[1])
            X_scaled_co2.columns = range(X_scaled_co2.shape[1])
            X_test1_temp.columns = range(X_test1_temp.shape[1])
            X_test1_hum.columns = range(X_test1_hum.shape[1])
            X_test1_co2.columns = range(X_test1_co2.shape[1])
            # X_test1_temp_2.columns = range(X_test1_temp_2.shape[1])
            # X_test1_hum_2.columns = range(X_test1_hum_2.shape[1])
            # X_test1_co2_2.columns = range(X_test1_co2_2.shape[1])
            X_val1_temp.columns = range(X_val1_temp.shape[1])
            X_val1_hum.columns = range(X_val1_hum.shape[1])
            X_val1_co2.columns = range(X_val1_co2.shape[1])

            ########################################################################
            X_scaled_temp = scalar_funct(X_scaled_temp, -10)
            X_scaled_hum = scalar_funct(X_scaled_hum, -10)
            X_scaled_co2 = scalar_funct(X_scaled_co2, -10)
            X_test1_temp = scalar_funct(X_test1_temp, -10)
            X_test1_hum = scalar_funct(X_test1_hum, -10)
            X_test1_co2 = scalar_funct(X_test1_co2, -10)
            X_val1_temp = scalar_funct(X_val1_temp, -10)
            X_val1_hum = scalar_funct(X_val1_hum, -10)
            X_val1_co2 = scalar_funct(X_val1_co2, -10)

            out_train = np.where(np.isnan(y_scaled_temp))[0]
            if len(out_train) > 0:
                y_scaled_temp = y_scaled_temp.drop(out_train, axis=0)
                y_scaled_temp = y_scaled_temp.reset_index(drop=True)
                X_scaled_temp = X_scaled_temp.drop(out_train, axis=0)
                X_scaled_temp = X_scaled_temp.reset_index(drop=True)
            out_train = np.where(np.isnan(y_scaled_hum))[0]
            if len(out_train) > 0:
                y_scaled_hum = y_scaled_hum.drop(out_train, axis=0)
                y_scaled_hum = y_scaled_hum.reset_index(drop=True)
                X_scaled_hum = X_scaled_hum.drop(out_train, axis=0)
                X_scaled_hum = X_scaled_hum.reset_index(drop=True)
            out_train = np.where(np.isnan(y_scaled_co2))[0]
            if len(out_train) > 0:
                y_scaled_co2 = y_scaled_co2.drop(out_train, axis=0)
                y_scaled_co2 = y_scaled_co2.reset_index(drop=True)
                X_scaled_co2 = X_scaled_co2.drop(out_train, axis=0)
                X_scaled_co2 = X_scaled_co2.reset_index(drop=True)
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
            #######################################################################################################
            out_T = np.where(np.isnan(y_val1_temp))[0]
            if len(out_T) > 0:
                y_val1_temp = y_val1_temp.drop(out_T, axis=0)
                y_val1_temp = y_val1_temp.reset_index(drop=True)
                X_val1_temp = X_val1_temp.drop(out_T, axis=0)
                X_val1_temp = X_val1_temp.reset_index(drop=True)
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

            X_val1_temp, X_val1_hum, X_val1_co2 = np.array(X_val1_temp), np.array(X_val1_hum), np.array(X_val1_co2)
            y_val1_temp, y_val1_hum, y_val1_co2 = np.array(y_val1_temp), np.array(y_val1_hum), np.array(y_val1_co2)
            X_test1_temp, X_test1_hum, X_test1_co2 = np.array(X_test1_temp), np.array(X_test1_hum), np.array(
                X_test1_co2)
            # X_test1_temp_2, X_test1_hum_2, X_test1_co2_2 = np.array(X_test1_temp_2), np.array(X_test1_hum_2), np.array(X_test1_co2_2)
            y_test1_temp, y_test1_hum, y_test1_co2 = np.array(y_test1_temp), np.array(y_test1_hum), np.array(
                y_test1_co2)
            # y_test1_temp_2, y_test1_hum_2, y_test1_co2_2 = np.array(y_test1_temp_2), np.array(y_test1_hum_2), np.array(y_test1_co2_2)
            X_scaled_temp, X_scaled_hum, X_scaled_co2 = np.array(X_scaled_temp), np.array(X_scaled_hum), np.array(
                X_scaled_co2)
            y_scaled_temp, y_scaled_hum, y_scaled_co2 = np.array(y_scaled_temp), np.array(y_scaled_hum), np.array(
                y_scaled_co2)

            evallist_temp = [(X_test1_temp, y_test1_temp)]
            evallist_hum = [(X_test1_hum, y_test1_hum)]
            evallist_co2 = [(X_test1_co2, y_test1_co2)]

            model_temp = xgb.XGBRegressor(n_estimators=n_temp)
            model_hum = xgb.XGBRegressor(n_estimators=n_hum)
            model_co2 = xgb.XGBRegressor(n_estimators=n_co2)

            model_temp.fit(X_scaled_temp, y_scaled_temp, early_stopping_rounds=20, eval_metric="rmse",
                           eval_set=evallist_temp, verbose=False)
            y_pred_temp = model_temp.predict(X_val1_temp)
            xgb1_temp = scalar_temp.inverse_transform(y_pred_temp.reshape(len(y_val1_temp), 1))
            rmse_temp = 100 * (np.sqrt(metrics.mean_squared_error(y_val1_temp, xgb1_temp)) / med_temp)
            nmbe_temp = MBE(y_val1_temp, xgb1_temp, med_temp)

            model_hum.fit(X_scaled_hum, y_scaled_hum, early_stopping_rounds=20, eval_metric="rmse",
                          eval_set=evallist_hum, verbose=False)
            y_pred_hum = model_hum.predict(X_val1_hum)
            xgb1_hum = scalar_hum.inverse_transform(y_pred_hum.reshape(len(y_val1_hum), 1))
            rmse_hum = 100 * (np.sqrt(metrics.mean_squared_error(y_val1_hum, xgb1_hum)) / med_hum)
            nmbe_hum = MBE(y_val1_hum, xgb1_hum, med_hum)

            model_co2.fit(X_scaled_co2, y_scaled_co2, early_stopping_rounds=20, eval_metric="rmse",
                          eval_set=evallist_co2, verbose=False)
            y_pred_co2 = model_co2.predict(X_val1_co2)
            xgb1_co2 = scalar_co2.inverse_transform(y_pred_co2.reshape(len(y_val1_co2), 1))
            rmse_co2 = 100 * (np.sqrt(metrics.mean_squared_error(y_val1_co2, xgb1_co2)) / med_co2)
            nmbe_co2 = MBE(y_val1_co2, xgb1_co2, med_co2)

        #############################################################################################
        else:
            print('Liada')
    else:
        print('Liada')

    return (xgb1_temp, xgb1_hum, xgb1_co2, y_val1_temp, y_val1_hum, y_val1_co2,rmse_temp,rmse_hum,rmse_co2)


def MTI_train_complete(x_original, y_temp, y_hum, y_co2, med_temp, med_hum, med_co2, positX, positY, n_temp, n_hum, n_co2):
    x_original = x_original.reset_index(drop=True)
    y_temp1 = y_temp.reset_index(drop=True)
    y_hum1 = y_hum.reset_index(drop=True)
    y_co21 = y_co2.reset_index(drop=True)


    #nmbe_temp =[0 for x in range(len(positX))]
    #rmse_temp = [0 for x in range(len(positX))]
    #time_temp = [0 for x in range(len(positX))]
    #nmbe_hum =[0 for x in range(len(positX))]
    #rmse_hum = [0 for x in range(len(positX))]
    #time_hum = [0 for x in range(len(positX))]
    #nmbe_co2 =[0 for x in range(len(positX))]
    #rmse_co2 = [0 for x in range(len(positX))]
    #time_co2 = [0 for x in range(len(positX))]
    #u_temp = 0
    #u_hum = 0
    #u_co2 = 0
    #for u in range(len(positX)):
        #print('CROSS NUMBER', u)
    P1= np.where(x_original.iloc[:,0]==positX)
    P2=np.where(x_original.iloc[:,1]==positY)
    yyy = np.intersect1d(P1, P2)
    x_val = x_original.iloc[yyy]
    x_val = x_val.reset_index(drop=True)
    y_temp_val = y_temp1.iloc[yyy]
    y_hum_val = y_hum1.iloc[yyy]
    y_co2_val= y_co21.iloc[yyy]
    y_temp_val = y_temp_val.reset_index(drop=True)
    y_hum_val = y_hum_val.reset_index(drop=True)
    y_co2_val = y_co2_val.reset_index(drop=True)
    x_t = x_original.drop(yyy, axis=0)
    x_t = x_t.reset_index(drop=True)
    y_temp_t = y_temp1.drop(yyy, axis=0)
    y_hum_t = y_hum1.drop(yyy, axis=0)
    y_co2_t = y_co21.drop(yyy, axis=0)
    y_temp_t = y_temp_t.reset_index(drop=True)
    y_hum_t = y_hum_t.reset_index(drop=True)
    y_co2_t = y_co2_t.reset_index(drop=True)
    if x_val.shape[0]>5:
        yy1 = np.where(x_t.iloc[:, 0] == 9.8)[0]
        yy2 = np.where(x_t.iloc[:, 1] == 7.8)[0]
        zz1 = np.intersect1d(yy1, yy2)
        x_test1 = x_t.iloc[zz1]
        x_test1 = x_test1.reset_index(drop=True)
        y_test_temp, y_test_hum, y_test_co2 = y_temp_t.iloc[zz1],y_hum_t.iloc[zz1],y_co2_t.iloc[zz1]
        y_test_temp, y_test_hum, y_test_co2= y_test_temp.reset_index(drop=True),y_test_hum.reset_index(drop=True),y_test_co2.reset_index(drop=True)
        x_t= x_t.drop(zz1, axis=0)
        x_t = x_t.reset_index(drop=True)
        y_temp_t, y_hum_t, y_co2_t = y_temp_t.drop(zz1, axis=0),y_hum_t.drop(zz1, axis=0),y_co2_t.drop(zz1, axis=0)
        y_temp_t, y_hum_t, y_co2_t = y_temp_t.reset_index(drop=True),y_hum_t.reset_index(drop=True),y_co2_t.reset_index(drop=True)


        if x_test1.shape[0]>5:
            x_test1=x_test1.drop(x_test1.columns[np.array([0,1,2])], axis=1)
            x_t = x_t.drop(x_t.columns[np.array([0, 1, 2])], axis=1)
            x_val = x_val.drop(x_val.columns[np.array([0, 1, 2])], axis=1)
            names= x_t.columns
            temperaturas_train = np.array(
                x_t.iloc[:, np.array([1, 4, 7, 10, 13, 16, 19, 22, 26, 31, 34, 37, 40, 43, 46, 49, 52])])
            temperaturas_test = np.array(
                x_test1.iloc[:, np.array([1, 4, 7, 10, 13, 16, 19, 22, 26, 31, 34, 37, 40, 43, 46, 49, 52])])
            temperaturas_val = np.array(
                x_val.iloc[:, np.array([1, 4, 7, 10, 13, 16, 19, 22, 26, 31, 34, 37, 40, 43, 46, 49, 52])])
            humedad_train = np.array(
                x_t.iloc[:, np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 30, 33, 36, 39, 42, 45, 48, 51])])
            humedad_test = np.array(
                x_test1.iloc[:, np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 30, 33, 36, 39, 42, 45, 48, 51])])
            humedad_val = np.array(
                x_val.iloc[:, np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 30, 33, 36, 39, 42, 45, 48, 51])])
            co2_train = np.array(
                x_t.iloc[:, np.array([2, 5, 8, 11, 14, 17, 20, 23, 32, 35, 38, 41, 44, 47, 50, 53])])
            co2_test = np.array(
                x_test1.iloc[:, np.array([2, 5, 8, 11, 14, 17, 20, 23, 32, 35, 38, 41, 44, 47, 50, 53])])
            co2_val = np.array(
                x_val.iloc[:, np.array([2, 5, 8, 11, 14, 17, 20, 23, 32, 35, 38, 41, 44, 47, 50, 53])])
            diss_train = np.array(x_t.iloc[:, range(x_t.shape[1] - 24, x_t.shape[1])])
            diss_test = np.array(x_test1.iloc[:, range(x_t.shape[1] - 24, x_t.shape[1])])
            diss_val = np.array(x_val.iloc[:, range(x_t.shape[1] - 24, x_t.shape[1])])
            rad_train = np.array(x_t.iloc[:, np.array([25])])
            rad_test = np.array(x_test1.iloc[:, np.array([25])])
            rad_val = np.array(x_val.iloc[:, np.array([25])])
            resto_train = x_t.iloc[:, np.array([27, 28, 29])]
            resto_test = x_test1.iloc[:, np.array([27, 28, 29])]
            resto_val = x_val.iloc[:, np.array([27, 28, 29])]

            scalar_temp = MinMaxScaler(feature_range=(0, 1))
            scalar_hum = MinMaxScaler(feature_range=(0, 1))
            scalar_co2 = MinMaxScaler(feature_range=(0, 1))
            scalardist = MinMaxScaler(feature_range=(0, 1))
            scalar_rad = MinMaxScaler(feature_range=(0, 1))
            scalarresto = MinMaxScaler(feature_range=(0, 1))
            scalar_temp.fit(np.concatenate((np.concatenate(temperaturas_train), np.concatenate(temperaturas_test),
                                            np.concatenate(temperaturas_val),np.array(y_temp_t),
                                            np.array(y_test_temp),np.array(y_temp_val))).reshape(-1, 1))
            scalardist.fit(np.concatenate((np.concatenate(diss_train), np.concatenate(diss_test),np.concatenate(diss_val))).reshape(-1, 1))
            scalar_hum.fit(np.concatenate((np.concatenate(humedad_train), np.concatenate(humedad_test), np.concatenate(humedad_val),
                 np.array(y_hum_t), np.array(y_test_hum), np.array(y_hum_val))).reshape(-1, 1))
            scalar_co2.fit(np.concatenate((np.concatenate(co2_train), np.concatenate(co2_test), np.concatenate(co2_val),
                                np.array(y_co2_t), np.array(y_test_co2),np.array(y_co2_val))).reshape(-1, 1))
            scalar_rad.fit(np.concatenate((np.concatenate(rad_train), np.concatenate(rad_test), np.concatenate(rad_val))).reshape(-1, 1))
            scalarresto.fit(pd.concat([resto_train, resto_test, resto_val], axis=0))
            y_scaled_temp = pd.DataFrame(scalar_temp.transform(np.array(y_temp_t).reshape(-1, 1)))
            y_scaled_hum = pd.DataFrame(scalar_hum.transform(np.array(y_hum_t).reshape(-1, 1)))
            y_scaled_co2 = pd.DataFrame(scalar_co2.transform(np.array(y_co2_t).reshape(-1, 1)))
            y_test1_temp = pd.DataFrame(scalar_temp.transform(np.array(y_test_temp).reshape(-1, 1)))
            y_test1_hum = pd.DataFrame(scalar_hum.transform(np.array(y_test_hum).reshape(-1, 1)))
            y_test1_co2 = pd.DataFrame(scalar_co2.transform(np.array(y_test_co2).reshape(-1, 1)))

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
            X_scaled_temp = pd.concat([temperaturas_train1,diss_train1,rad_train1, resto_train1], axis=1)
            X_test1_temp = pd.concat([temperaturas_test1,diss_test1 ,rad_test1, resto_test1], axis=1)
            X_val1_temp = pd.concat([temperaturas_val1, diss_val1,rad_val1, resto_val1], axis=1)
            X_scaled_hum = pd.concat([humedad_train1,diss_train1,rad_train1, resto_train1], axis=1)
            X_test1_hum = pd.concat([humedad_test1,diss_test1, rad_test1, resto_test1], axis=1)
            X_val1_hum = pd.concat([humedad_val1,diss_val1, rad_val1, resto_val1], axis=1)
            X_scaled_co2 = pd.concat([co2_train1, diss_train1,rad_train1, resto_train1], axis=1)
            X_test1_co2 = pd.concat([co2_test1,diss_test1,rad_test1, resto_test1], axis=1)
            X_val1_co2 = pd.concat([co2_val1,diss_val1, rad_val1, resto_val1], axis=1)

            y_val1_temp = y_temp_val
            y_val1_hum = y_hum_val
            y_val1_co2= y_co2_val
            #X_scaled_temp, y_scaled_temp =miss_funct(X_scaled_temp, y_scaled_temp,6)
            #X_scaled_hum, y_scaled_hum =miss_funct(X_scaled_hum, y_scaled_hum,6)
            #X_scaled_co2, y_scaled_co2 =miss_funct(X_scaled_co2, y_scaled_co2,6)
            #X_test1_temp, y_test1_temp =miss_funct(X_test1_temp, y_test1_temp,6)
            #X_test1_hum, y_test1_hum =miss_funct(X_test1_hum, y_test1_hum,6)
            #X_test1_co2, y_test1_co2 =miss_funct(X_test1_co2, y_test1_co2,6)
            #X_val1_temp,y_val1_temp=miss_funct(X_val1_temp,y_val1_temp,6)
            #X_val1_hum, y_val1_hum =miss_funct(X_val1_hum, y_val1_hum,6)
            #X_val1_co2, y_val1_co2 =miss_funct(X_val1_co2, y_val1_co2,6)
            X_scaled_temp, y_scaled_temp = miss_funct(X_scaled_temp, y_scaled_temp, 6)
            X_scaled_hum, y_scaled_hum = miss_funct(X_scaled_hum, y_scaled_hum, 6)
            X_scaled_co2, y_scaled_co2 = miss_funct(X_scaled_co2, y_scaled_co2, 6)
            X_test1_temp, y_test1_temp = miss_funct(X_test1_temp, y_test1_temp, 6)
            X_test1_hum, y_test1_hum = miss_funct(X_test1_hum, y_test1_hum, 6)
            X_test1_co2, y_test1_co2 = miss_funct(X_test1_co2, y_test1_co2, 6)
            X_val1_temp, y_val1_temp = miss_funct(X_val1_temp, y_val1_temp, 6)
            X_val1_hum, y_val1_hum = miss_funct(X_val1_hum, y_val1_hum, 6)
            X_val1_co2, y_val1_co2 = miss_funct(X_val1_co2, y_val1_co2, 6)
            X_scaled_co2 = X_scaled_co2.reset_index(drop=True)
            X_scaled_temp = X_scaled_temp.reset_index(drop=True)
            X_scaled_hum = X_scaled_hum.reset_index(drop=True)
            X_val1_co2 = X_val1_co2.reset_index(drop=True)
            X_val1_temp = X_val1_temp.reset_index(drop=True)
            X_val1_hum = X_val1_hum.reset_index(drop=True)
            X_test1_co2 = X_test1_co2.reset_index(drop=True)
            X_test1_temp = X_test1_temp.reset_index(drop=True)
            X_test1_hum = X_test1_hum.reset_index(drop=True)

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
            X_scaled_temp = scalar_funct(X_scaled_temp, -10)
            X_scaled_hum  = scalar_funct(X_scaled_hum, -10)
            X_scaled_co2  = scalar_funct(X_scaled_co2, -10)
            X_test1_temp = scalar_funct(X_test1_temp, -10)
            X_test1_hum  = scalar_funct(X_test1_hum, -10)
            X_test1_co2  = scalar_funct(X_test1_co2, -10)
            X_val1_temp = scalar_funct(X_val1_temp, -10)
            X_val1_hum  = scalar_funct(X_val1_hum, -10)
            X_val1_co2  = scalar_funct(X_val1_co2, -10)
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
            #######################################################################################################
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

            X_val1_temp, X_val1_hum, X_val1_co2 = np.array(X_val1_temp), np.array(X_val1_hum), np.array(X_val1_co2)
            y_val1_temp, y_val1_hum, y_val1_co2 = np.array(y_val1_temp), np.array(y_val1_hum), np.array(y_val1_co2)
            X_test1_temp, X_test1_hum, X_test1_co2 = np.array(X_test1_temp), np.array(X_test1_hum), np.array(X_test1_co2)
            y_test1_temp, y_test1_hum, y_test1_co2 = np.array(y_test1_temp), np.array(y_test1_hum), np.array(y_test1_co2)
            X_scaled_temp, X_scaled_hum, X_scaled_co2 = np.array(X_scaled_temp), np.array(X_scaled_hum), np.array(X_scaled_co2)
            y_scaled_temp, y_scaled_hum, y_scaled_co2 = np.array(y_scaled_temp), np.array(y_scaled_hum), np.array(y_scaled_co2)
            evallist_temp = [(X_test1_temp, y_test1_temp)]
            evallist_hum = [(X_test1_hum, y_test1_hum)]
            evallist_co2 = [(X_test1_co2, y_test1_co2)]
            model_temp = xgb.XGBRegressor(n_estimators=n_temp)
            model_hum = xgb.XGBRegressor(n_estimators=n_hum)
            model_co2 = xgb.XGBRegressor(n_estimators=n_co2)


            model_temp.fit(X_scaled_temp, y_scaled_temp, early_stopping_rounds=20, eval_metric="rmse",
                                              eval_set=evallist_temp, verbose=False)
            y_pred_temp = model_temp.predict(X_val1_temp)
            xgb1_temp = scalar_temp.inverse_transform(y_pred_temp.reshape(len(y_val1_temp), 1))
            rmse_temp = 100 * (np.sqrt(metrics.mean_squared_error(y_val1_temp, xgb1_temp)) / med_temp)
            nmbe_temp = MBE(y_val1_temp, xgb1_temp, med_temp)


            model_hum.fit(X_scaled_hum, y_scaled_hum, early_stopping_rounds=20, eval_metric="rmse",
                                              eval_set=evallist_hum, verbose=False)
            y_pred_hum = model_hum.predict(X_val1_hum)
            xgb1_hum = scalar_hum.inverse_transform(y_pred_hum.reshape(len(y_val1_hum), 1))
            rmse_hum = 100 * (np.sqrt(metrics.mean_squared_error(y_val1_hum, xgb1_hum)) / med_hum)
            nmbe_hum = MBE(y_val1_hum, xgb1_hum, med_hum)

            model_co2.fit(X_scaled_co2, y_scaled_co2, early_stopping_rounds=20, eval_metric="rmse",
                                              eval_set=evallist_co2, verbose=False)
            y_pred_co2 = model_co2.predict(X_val1_co2)
            xgb1_co2 = scalar_co2.inverse_transform(y_pred_co2.reshape(len(y_val1_co2), 1))
            rmse_co2 = 100 * (np.sqrt(metrics.mean_squared_error(y_val1_co2, xgb1_co2)) / med_co2)
            nmbe_co2 = MBE(y_val1_co2, xgb1_co2, med_co2)

    #############################################################################################
        else:
            print('Liada')
    else:
        print('Liada')

    return(xgb1_temp, xgb1_hum, xgb1_co2, y_val1_temp, y_val1_hum, y_val1_co2, rmse_temp,rmse_hum,rmse_co2)

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


def MTI_total(DATA_C, type):
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

    if type=='complete':
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

        yy1 = np.where((x_final.iloc[:, 0] == 6.9) | (x_final.iloc[:, 0] == 25.5))[0]
        yy2 = np.where((x_final.iloc[:, 1] == 4) | (x_final.iloc[:, 1] == 1.6))[0]
        #
        # yy3 = np.where((x_final.iloc[:, 0] == 46.3) | (x_final.iloc[:, 0] == 28.8))[0]
        # yy4 = np.where((x_final.iloc[:, 1] == 7.6) | (x_final.iloc[:, 1] == 10.1))[0]
        ## yy3 = np.where((x_final.iloc[:, 0] == 28.8) & (x_final.iloc[:, 1] == 10.1))[0]
        #
        zz1 = np.intersect1d(yy1, yy2)
        # zz2 = np.intersect1d(yy3, yy4)
        # zz1 = np.sort(np.concatenate((zz1, zz2)))
        yy3 = np.where((x_final.iloc[:, 0] == 3.9) & (x_final.iloc[:, 1] == 15))[0]
        #zz1 = np.intersect1d(zz1, yy3)
        zz1 = np.concatenate((zz1, yy3))
        zz1 = np.sort(zz1)
        #
        y_temp = y_temp.drop(zz1, axis=0)
        y_hum = y_hum.drop(zz1, axis=0)
        y_co2 = y_co2.drop(zz1, axis=0)
        y_temp = y_temp.reset_index(drop=True)
        y_hum = y_hum.reset_index(drop=True)
        y_co2 = y_co2.reset_index(drop=True)

        x_final = x_final.drop(zz1, axis=0)
        x_final = x_final.reset_index(drop=True)

        #positX = 2.3
        #positY = 4.7
        #positX = 17.6
        #positY = 6.8
        positX = 31.2
        positY = 1.6
        n_temp, n_hum, n_co2, = 200, 240, 60
        pred_temp, pred_hum, pred_co2, real_temp, real_hum, real_co2,rmse_temp,rmse_hum,rmse_co2 = MTI_train_complete(x_final, y_temp, y_hum, y_co2, med_temp, med_hum, med_co2, positX, positY, n_temp, n_hum,n_co2)

    else:
        # distanceX_02 = pd.DataFrame(pos_cajasT.iloc[0, 0] - posit.iloc[:, 0])
        # distanceY_02 = pd.DataFrame(pos_cajasT.iloc[0, 1] - posit.iloc[:, 1])
        # distanceZ_02 = pd.DataFrame(pos_cajasT.iloc[0, 2] - posit.iloc[:, 2])
        # distanceX_09 = pd.DataFrame(pos_cajasT.iloc[1, 0] - posit.iloc[:, 0])
        # distanceY_09 = pd.DataFrame(pos_cajasT.iloc[1, 1] - posit.iloc[:, 1])
        # distanceZ_09 = pd.DataFrame(pos_cajasT.iloc[1, 2] - posit.iloc[:, 2])
        distanceX_1a = pd.DataFrame(pos_cajasT.iloc[2, 0] - posit.iloc[:, 0])
        distanceY_1a = pd.DataFrame(pos_cajasT.iloc[2, 1] - posit.iloc[:, 1])
        distanceZ_1a = pd.DataFrame(pos_cajasT.iloc[2, 2] - posit.iloc[:, 2])
        distanceX_3d = pd.DataFrame(pos_cajasT.iloc[3, 0] - posit.iloc[:, 0])
        distanceY_3d = pd.DataFrame(pos_cajasT.iloc[3, 1] - posit.iloc[:, 1])
        distanceZ_3d = pd.DataFrame(pos_cajasT.iloc[3, 2] - posit.iloc[:, 2])
        # distanceX_B1 = pd.DataFrame(pos_cajasT.iloc[4, 0] - posit.iloc[:, 0])
        # distanceY_B1 = pd.DataFrame(pos_cajasT.iloc[4, 1] - posit.iloc[:, 1])
        # distanceZ_B1 = pd.DataFrame(pos_cajasT.iloc[4, 2] - posit.iloc[:, 2])
        distanceX_B2 = pd.DataFrame(pos_cajasT.iloc[5, 0] - posit.iloc[:, 0])
        distanceY_B2 = pd.DataFrame(pos_cajasT.iloc[5, 1] - posit.iloc[:, 1])
        distanceZ_B2 = pd.DataFrame(pos_cajasT.iloc[5, 2] - posit.iloc[:, 2])
        # distanceX_B3 = pd.DataFrame(pos_cajasT.iloc[6, 0] - posit.iloc[:, 0])
        # distanceY_B3 = pd.DataFrame(pos_cajasT.iloc[6, 1] - posit.iloc[:, 1])
        # distanceZ_B3 = pd.DataFrame(pos_cajasT.iloc[6, 2] - posit.iloc[:, 2])
        # distanceX_B4 = pd.DataFrame(pos_cajasT.iloc[7, 0] - posit.iloc[:, 0])
        # distanceY_B4 = pd.DataFrame(pos_cajasT.iloc[7, 1] - posit.iloc[:, 1])
        # distanceZ_B4 = pd.DataFrame(pos_cajasT.iloc[7, 2] - posit.iloc[:, 2])

        names = X_final.columns
        X_final = pd.concat(
            [X_final, distanceX_1a, distanceX_3d, distanceX_B2,
             distanceY_1a, distanceY_3d, distanceY_B2, distanceZ_1a, distanceZ_3d, distanceZ_B2, ], axis=1)

        nn = ["distanceX_1a", "distanceX_3d", "distanceX_B2",
              "distanceY_1a", "distanceY_3d", "distanceY_B2", "distanceZ_1a", "distanceZ_3d", "distanceZ_B2"]

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

        yy1 = np.where((x_final.iloc[:, 0] == 6.9) | (x_final.iloc[:, 0] == 25.5))[0]
        yy2 = np.where((x_final.iloc[:, 1] == 4) | (x_final.iloc[:, 1] == 1.6))[0]
        #
        # yy3 = np.where((x_final.iloc[:, 0] == 46.3) | (x_final.iloc[:, 0] == 28.8))[0]
        # yy4 = np.where((x_final.iloc[:, 1] == 7.6) | (x_final.iloc[:, 1] == 10.1))[0]
        ## yy3 = np.where((x_final.iloc[:, 0] == 28.8) & (x_final.iloc[:, 1] == 10.1))[0]
        #
        zz1 = np.intersect1d(yy1, yy2)
        # zz2 = np.intersect1d(yy3, yy4)
        # zz1 = np.sort(np.concatenate((zz1, zz2)))
        yy3 = np.where((x_final.iloc[:, 0] == 3.9) & (x_final.iloc[:, 1] == 15))[0]
        zz1 = np.intersect1d(zz1, yy3)
        zz1 = np.sort(zz1)
        #
        y_temp = y_temp.drop(zz1, axis=0)
        y_hum = y_hum.drop(zz1, axis=0)
        y_co2 = y_co2.drop(zz1, axis=0)
        y_temp = y_temp.reset_index(drop=True)
        y_hum = y_hum.reset_index(drop=True)
        y_co2 = y_co2.reset_index(drop=True)

        x_final = x_final.drop(zz1, axis=0)
        x_final = x_final.reset_index(drop=True)

        #positX = 28.7
        #positY = 7.8
        #positX = 27.9
        #positY = 14.6
        #positX = 26
        #positY = 5.5
        positX = 31.2
        positY = 1.6

        n_temp, n_hum, n_co2, = 20, 20, 240
        pred_temp, pred_hum, pred_co2, real_temp, real_hum, real_co2,rmse_temp,rmse_hum,rmse_co2 = MTI_train_reduced(x_final, y_temp, y_hum, y_co2, med_temp, med_hum, med_co2, positX, positY, n_temp, n_hum,
                           n_co2)



    return(pred_temp, pred_hum, pred_co2,real_temp, real_hum, real_co2,rmse_temp,rmse_hum,rmse_co2)

######################################################################################################################
######################################################################################################################
# Loading of data from the different hosts
variables = []
time_end = '2021-05-06T10:00:00.000Z'

#############################################################################
#############################################################################
# 0x6a52
pos_z= loading_carrito(['vert'], 'vertpantilt', '0x6a52', variables, time_end, False)
variables=[]
variables_52 = loading_carrito(['humidity', 'temperature'], 'sht31d', '0x6a52', variables, time_end, False)
variables_52 = loading_carrito(['pressure'], 'bme680_bsec', '0x6a52', variables_52, time_end, False)
variables_52 = loading_carrito(['co2'], 'mhz14', '0x6a52', variables_52, time_end, True)
names_52 = ['humidity_C','temperature_C','pressure_C','co2_C']

#Data of the boxes
# 0x6a02
variables=[]
variables_02 = loading(['humidity', 'temperature'], 'sht31d', '0x6a02', variables, time_end, False)
variables_02 = loading(['co2'], 'mhz14', '0x6a02', variables_02, time_end, True)
names_02 = ['humidity_02','temperature_02','co2_02']
# 0x6a09
variables = []
variables_09 = loading(['humidity', 'temperature'], 'sht31d', '0x6a09', variables, time_end, False)
variables_09 = loading(['co2'], 'mhz14', '0x6a09', variables_09, time_end, True)
names_09 = ['humidity_09','temperature_09','co2_09']
# 0x6a1a
variables = []
variables_1a = loading(['humidity', 'temperature'], 'sht31d', '0x6a1a', variables, time_end, False)
variables_1a = loading(['co2'], 'mhz14', '0x6a1a', variables_1a, time_end, True)
names_1a = ['humidity_1a','temperature_1a','co2_1a']
# 0x6a3d
variables = []
variables_3d = loading(['humidity', 'temperature'], 'sht31d', '0x6a3d', variables, time_end, False)
variables_3d = loading(['co2'], 'mhz14', '0x6a3d', variables_3d, time_end, True)
names_3d = ['humidity_3d','temperature_3d','co2_3d']
# rpiB1
variables = []
variables_B1 = loading(['humidity', 'temperature'], 'sht31d', 'rpiB1', variables, time_end, False)
variables_B1 = loading(['co2'], 'mhz14', 'rpiB1', variables_B1, time_end, True)
names_B1 = ['humidity_B1','temperature_B1','co2_B1']
# rpiB2
variables = []
variables_B2 = loading(['humidity', 'temperature'], 'sht31d', 'rpiB2', variables, time_end, False)
variables_B2 = loading(['co2'], 'mhz14', 'rpiB2', variables_B2, time_end, True)
names_B2 = ['humidity_B2','temperature_B2','co2_B2']
# rpiB3
variables = []
variables_B3 = loading(['humidity', 'temperature'], 'sht31d', 'rpiB3', variables, time_end, False)
variables_B3 = loading(['co2'], 'mhz14', 'rpiB3', variables_B3, time_end, True)
names_B3 = ['humidity_B3','temperature_B3','co2_B3']
# rpiB4
variables = []
variables_B4 = loading(['humidity', 'temperature'], 'sht31d', 'rpiB4', variables, time_end, False)
variables_B4 = loading(['co2'], 'mhz14', 'rpiB4', variables_B4, time_end, True)
names_B4 = ['humidity_B4','temperature_B4','co2_B4']

#Meteo data
variables = []
variables_meteo = loading(['humidity', 'radiation','temperature'], 'meteo', 'none', variables, time_end, False)
#variables_meteo = loading(['humidity', 'pressure','radiation','rain','temperature','windeast','windnorth'], 'meteo', 'none', variables, time_end)
names_meteo=['humidity_M','radiation_M','temperature_M']
var_meteo=[]
for u in range(len(variables_meteo)):
    var_meteo.append(position_meteo(variables_meteo[u],len(variables_B4[0])))

#Cart positions
positions = pd.read_csv("positions_new_new.csv", sep=";", decimal=".")
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

DATA= variables
date_init= '2020-06-17 11:00:00'
date = pd.date_range(date_init, periods=DATA.shape[0], freq='1min')
# TRAINING
# Hour of day
HOUR = [0 for x in range(len(date))]
for h in range(len(date)):
    HOUR[h] = date[h].hour + date[h].minute / 100

# Hour of the year
DAY = [0 for x in range(len(date))]
for h in range(len(date)):
    DAY[h] = date[h].dayofyear*24 + date[h].hour+ date[h].minute / 100

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
                    pd.DataFrame(yearday).set_index(DATA.index),pd.DataFrame(yearday).set_index(DATA.index)], axis=1, ignore_index=True)

temporal = ['hour', 'week', 'day','yearday']
names2 = np.concatenate([names, temporal])
DATA_C.columns = names2


DATA_C = DATA_C.drop(range(21486), axis=0)
DATA_C = DATA_C.reset_index(drop=True)
DATA_C = DATA_C.drop(DATA_C.shape[0]-1, axis=0)
DATA_C = DATA_C.reset_index(drop=True)

#######################################################################################################
pred_temp_complete, pred_hum_complete, pred_co2_complete,real_temp, real_hum, real_co2,rmse_temp_complete,rmse_hum_complete,rmse_co2_complete = MTI_total(DATA_C, 'complete')
pred_temp_reduced, pred_hum_reduced,pred_co2_reduced,real_temp, real_hum, real_co2,rmse_temp_reduced,rmse_hum_reduced,rmse_co2_reduced = MTI_total(DATA_C, 'reduced')



import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from matplotlib import rcParams
rcParams['axes.titlepad'] = 20

pred_temp_complete = pd.DataFrame(pred_temp_complete[0:3420])
pred_hum_complete = pd.DataFrame(pred_hum_complete[0:3420])
pred_co2_complete = pd.DataFrame(pred_co2_complete[0:3420])

pred_temp_reduced = pd.DataFrame(pred_temp_reduced[0:3420])
pred_hum_reduced = pd.DataFrame(pred_hum_reduced[0:3420])
pred_co2_reduced = pd.DataFrame(pred_co2_reduced[0:3420])

real_temp = pd.DataFrame(real_temp[0:3420])
real_hum = pd.DataFrame(real_hum[0:3420])
real_co2 = pd.DataFrame(real_co2[0:3420])

#r_temp = 100 * (np.sqrt(metrics.mean_squared_error(real_temp, pred_temp_complete)) / med_temp)
#p_temp = 100 * (np.sqrt(metrics.mean_squared_error(real_temp, pred_temp_reduced)) / med_temp)
#r_hum = 100 * (np.sqrt(metrics.mean_squared_error(real_hum, pred_hum_complete)) / med_hum)
#p_hum = 100 * (np.sqrt(metrics.mean_squared_error(real_hum, pred_hum_reduced)) / med_hum)
#r_co2 = 100 * (np.sqrt(metrics.mean_squared_error(real_co2, pred_co2_complete)) / med_co2)
#p_co2 = 100 * (np.sqrt(metrics.mean_squared_error(real_co2, pred_co2_reduced)) / med_co2)



#temp1 = [0 for x in range(360)]
#temp2 = [0 for x in range(360)]
#temp3 = [0 for x in range(360)]
#
#for u in range(360):
#    temp1[u] = np.random.uniform(25,28)
#    temp2[u] = np.random.uniform(20,24)
#    temp3[u] = np.random.uniform(23,26)
fig, axes = plt.subplots(3,1, figsize=(17.5,17))
plt.grid(True, linewidth=0.4, color='black')
#fig.suptitle('Errors distribution', fontsize=30
pred  =[pred_temp_complete.iloc[:,0], pred_hum_complete.iloc[:,0], pred_co2_complete.iloc[:,0]]
real = [real_temp.iloc[:,0], real_hum.iloc[:,0], real_co2.iloc[:,0]]
pred_red = [pred_temp_reduced.iloc[:,0], pred_hum_reduced.iloc[:,0], pred_co2_reduced.iloc[:,0]]
colors = ['red', 'blue', 'darkorange']
vars = ['Temperature [ºC]', 'Relative humidity [%]', 'CO$_2$ [ppm]']
sns.set(font_scale=1.2)
#dates = pd.date_range("2020/02/10 00:00", "2020/02/10 05:59", freq="1min").strftime('%Y %d %b %H:%M')
dates = pd.date_range('2021-03-10 15:00:00', periods=len(pred_temp_reduced), freq='1min')
ylims_inf = [14.5, 40, 380]
ylims_sup = [23, 74, 740]
fig.subplots_adjust(hspace=0.4)
#esu = pd.concat([pd.DataFrame(temp1), pd.DataFrame(temp2)])
#res= pd.concat([pd.DataFrame(temp1)[0], pd.DataFrame(temp2)[0], pd.DataFrame(temp3)[0]], axis=0)
#res = res.reset_index(drop=True)
#a = pd.DataFrame(np.concatenate((np.repeat(1, 7200),np.repeat(2, 7200),np.repeat(3, 7200) )))[0]
#a=a.reset_index(drop=True)
#res = pd.concat([res, a ], axis=1)



#res.columns=['Numbers','Group']
for i in range(3):
    #x_cv = np.linspace(np.min(cvs[i]), np.max(cvs[i]), len(cvs[i]))
    sns.lineplot(ax=axes[i],x=dates, y=pred[i], color='blue')
    sns.lineplot(ax=axes[i], x=dates, y=pred_red[i], color='red')
    sns.lineplot(ax=axes[i], x=dates, y=real[i], color='black')
    axes[i].set_ylim(ylims_inf[i],ylims_sup[i])
    axes[i].set_ylabel(vars[i], fontsize=18)
    axes[i].get_yaxis().set_label_coords(-0.04, 0.5)
    axes[i].grid(True, linewidth=0.25, color='black')
    #axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%y'))
    axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b %H:%M'))
    axes[i].xaxis.set_major_locator(mdates.HourLocator(interval=10))
    #axes[i].xaxis.set_major_locator(MaxNLocator(7))
    # beautify the x-labels
    #plt.locator_params(axis='x', nbins=4)
    #hh= matplotlib.dates.HourLocator()
    #mm = matplotlib.dates.MonthLocator()
    #axes[i].xaxis.set_major_locator(hh)
    #plt.gcf().autofmt_xdate()
    axes[i].tick_params(axis='x',labelsize=16)
    axes[i].tick_params(axis='y',labelsize=16)

    if i == 0:
        axes[i].set_title("[Original model = 4.05  %] $\qquad$  $\qquad$ $\qquad$  $\qquad$ $\qquad$[Reduced model = 4.49 %]", fontsize=22)
    elif i==1:
        axes[i].set_title('[Original model = 3.83 %] $\qquad$ $\qquad$  $\qquad$  $\qquad$  $\qquad$[Reduced model = 4.10 %]',
                          fontsize=22)
    elif i==2:
        axes[i].set_xlabel('Time',fontsize=17)
        axes[i].set_title('[Original model = 6.71 %] $\qquad$ $\qquad$ $\qquad$  $\qquad$  $\qquad$[Reduced model = 6.15 %]',
                          fontsize=22)
    #x_nm = np.linspace(np.min(nmbes[i]), np.max(nmbes[i]), len(nmbes[i]))
    #sns.histplot(ax=axes[i, 1], x=nmbes[i], stat='density', color=colors[i])
    #mu, std = norm.fit(nmbes[i])
    #p = norm.pdf(x_nm, mu, std)

#plt.tight_layout()
#custom_lines1 = [Line2D([0], [0], color='black', lw=2, label='Monitored', linestyle='-'),
#            Line2D([0], [0], color='grey', lw=2, label='Complete model', linestyle='-'),
#                Line2D([0], [0], color='blue', lw=2, label='Reduced model', linestyle='-'),
#                 Line2D([0], [0], color='black', lw=2, label='Monitored', linestyle='-')]
#az2=fig.legend(handles = custom_lines1, loc=3,bbox_to_anchor=(0.02,-0.07),  ncol=4,facecolor='white',edgecolor='white', fontsize=17)
#
#custom_lines = [Line2D([0], [0], color='black', lw=2, label='Complete model', linestyle='-'),
#            Line2D([0], [0], color='black', lw=2, label='Reduce model', linestyle='--')]
#az=fig.legend(handles = custom_lines, ncol=2, loc=4,bbox_to_anchor=(0.99,-0.07),facecolor='white',edgecolor='white', fontsize=17)
#
#fig.savefig('sample', bbox_extra_artists=(az,az2), bbox_inches='tight')


plt.tight_layout()
fig.subplots_adjust(hspace=0.5)
fig.set_size_inches(17.5, 16)
custom_lines1 = [Line2D([0], [0], color='black', lw=2, label='Monitored', linestyle='-'),
            Line2D([0], [0], color='blue', lw=2, label='Original model', linestyle='-'),
                Line2D([0], [0], color='red', lw=2, label='Reduced model', linestyle='-')]
az2=fig.legend(handles = custom_lines1, loc=3,bbox_to_anchor=(0.29,-0.07),  ncol=3,facecolor='white',edgecolor='white', fontsize=18)

#custom_lines = [Line2D([0], [0], color='black', lw=2, label='Complete model', linestyle='-'),
#            Line2D([0], [0], color='black', lw=2, label='Reduce model', linestyle='--')]
#az=fig.legend(handles = custom_lines, ncol=2, loc=4,bbox_to_anchor=(0.99,-0.07),facecolor='white',edgecolor='white', fontsize=17)

fig.savefig('sample3', bbox_inches='tight', dpi=100)

