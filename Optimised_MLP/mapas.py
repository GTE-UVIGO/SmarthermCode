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



#temp
#GLOBAL_SEED =6278
#GLOBAL_SEED =963754 #buena
#hum
GLOBAL_SEED =6

#GLOBAL_SEED =73255

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



import pandas as pd
import numpy as np
import joblib
from random import randint
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from time import time
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import seaborn as sns; sns.set()
from tensorflow.compat.v1.keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestRegressor
from tensorflow.compat.v1.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn import metrics
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score




import xgboost as xgb
from random import randint
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

#FUNCTIONS
#FUNCTIONS
def MBE(y_true, y_pred):
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
    mbe = np.mean(y_true-y_pred)/np.mean(y_true)
    return(mbe*100)


def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))

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
            Y[:,s]= x[i:(i+zc)]
            i= i + zc
            s = s +1
    return(Y)

def means(X, sep):
    LISTA=[]
    for i in range(X.shape[1]):
        data = X.iloc[:, i]
        dat = cortes(data, len(data), sep)
        dat = np.nanmean(dat, axis=0)
        LISTA.append(pd.Series(dat))
    L= pd.DataFrame(LISTA).T
    return(L)

######################################################################################################################
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
            for t in range(len(point)):
                values[t] = point[t][var_name[u]]

            variables.append(values)

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


    return variables

def nas_function(x):
    for j in range(x.shape[0]):
        # Possibility of missing values
        if any(np.array(np.isnan(x.iloc[j, :]))):
            ii = np.where(np.isnan(x.iloc[j, :]))[0]
            x.iloc[j, ii] = 0
    return (x)


def MTI_train(x_train, y_train, x_test, y_test, neurons1, neurons2, neurons3, fechas, var):
     X_final = x_train.reset_index(drop=True)
     y_train1 = y_train.reset_index(drop=True)
     X_test1 = x_test.reset_index(drop=True)
     y_test1 = y_test.reset_index(drop=True)

     temperaturas_train = np.array(X_final.iloc[:, np.array([1, 4, 7, 10, 13, 16, 19, 22, 26, 31, 34, 37, 40, 43, 46, 49, 52])])
     temperaturas_test = np.array(X_test1.iloc[:, np.array([1, 4, 7, 10, 13, 16, 19, 22, 26, 31, 34, 37, 40, 43, 46, 49, 52])])

     humedad_train = np.array(X_final.iloc[:, np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 30, 33, 36, 39, 42, 45, 48, 51])])
     humedad_test = np.array(X_test1.iloc[:, np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 30, 33, 36, 39, 42, 45, 48, 51])])

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
     #ohe = OneHotEncoder(sparse=False)

     if var == 'temp':
         scalar_temp.fit(np.concatenate(
             (np.concatenate(temperaturas_train), np.concatenate(temperaturas_test),
              y_train1, y_test1)).reshape(-1, 1))
         scalardist.fit(np.concatenate((np.concatenate(diss_train), np.concatenate(diss_test),
                                        np.array([50.6, -50.6]))).reshape(-1, 1))
         scalar_hum.fit(np.concatenate(
             (np.concatenate(humedad_train), np.concatenate(humedad_test))).reshape(-1, 1))

         scalar_rad.fit(
             np.concatenate((np.concatenate(rad_train), np.concatenate(rad_test))).reshape(-1,1))
         scalarresto.fit(pd.concat([resto_train, resto_test], axis=0))
         y_train1 = pd.DataFrame(scalar_temp.transform(np.array(y_train1).reshape(-1, 1)))
         y_test1 = pd.DataFrame(scalar_temp.transform(np.array(y_test1).reshape(-1, 1)))
     elif var == 'hum':
         scalar_temp.fit(
             np.concatenate((np.concatenate(temperaturas_train), np.concatenate(temperaturas_test))).reshape(-1, 1))
         scalardist.fit(np.concatenate((np.concatenate(diss_train), np.concatenate(diss_test),np.array([50.6, -50.6]) )).reshape(-1, 1))
         scalar_hum.fit(np.concatenate((np.concatenate(humedad_train), np.concatenate(humedad_test), y_train,
                                        y_test)).reshape(-1, 1))
         scalar_rad.fit(np.concatenate((np.concatenate(rad_train), np.concatenate(rad_test))).reshape(-1, 1))
         scalarresto.fit(pd.concat([resto_train, resto_test], axis=0))
         y_train1 = pd.DataFrame(scalar_hum.transform(np.array(y_train1).reshape(-1, 1)))
         y_test1 = pd.DataFrame(scalar_hum.transform(np.array(y_test1).reshape(-1, 1)))


     temperaturas_train1 = np.zeros((temperaturas_train.shape[0], temperaturas_train.shape[1]))
     temperaturas_test1 = np.zeros((temperaturas_test.shape[0], temperaturas_train.shape[1]))
     humedad_train1 = np.zeros((humedad_train.shape[0], humedad_train.shape[1]))
     humedad_test1 = np.zeros((humedad_test.shape[0], humedad_test.shape[1]))
     for i in range(temperaturas_train.shape[1]):
         temperaturas_train1[:, i] = scalar_temp.transform(temperaturas_train[:, i].reshape(-1, 1))[:, 0]
         temperaturas_test1[:, i] = scalar_temp.transform(temperaturas_test[:, i].reshape(-1, 1))[:, 0]
         humedad_train1[:, i] = scalar_hum.transform(humedad_train[:, i].reshape(-1, 1))[:, 0]
         humedad_test1[:, i] = scalar_hum.transform(humedad_test[:, i].reshape(-1, 1))[:, 0]


     temperaturas_train1 = pd.DataFrame(temperaturas_train1)
     temperaturas_test1 = pd.DataFrame(temperaturas_test1)
     humedad_train1 = pd.DataFrame(humedad_train1)
     humedad_test1 = pd.DataFrame(humedad_test1)

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

     if var == 'temp':
         x_train1 = pd.concat([temperaturas_train1, rad_train1, resto_train1, diss_train1], axis=1)
         x_test1 = pd.concat([temperaturas_test1, rad_test1, resto_test1, diss_test1], axis=1)
         miss = x_train1.apply(lambda x: x.count(), axis=1) - 45
         miss = np.where(miss <= -2)[0]
         x_train1 = x_train1.drop(miss, axis=0)
         x_train1 = x_train1.reset_index(drop=True)
         y_train1 = y_train1.drop(miss, axis=0)
         y_train1 = y_train1.reset_index(drop=True)
     elif var == 'hum':
         x_train1 = pd.concat([humedad_train1, rad_train1,resto_train1, diss_train1], axis=1)
         x_test1 = pd.concat([humedad_test1, rad_test1,resto_test1, diss_test1], axis=1)
         miss = x_train1.apply(lambda x: x.count(), axis=1) - 45
         miss = np.where(miss <= -2)[0]
         x_train1 = x_train1.drop(miss, axis=0)
         x_train1 = x_train1.reset_index(drop=True)
         y_train1 = y_train1.drop(miss, axis=0)
         y_train1 = y_train1.reset_index(drop=True)

     x_train1.columns = range(x_train1.shape[1])
     x_test1.columns = range(x_test1.shape[1])

     for t in range(x_train1.shape[1]):
         a = x_train1.iloc[:, t]
         if len(np.where(np.isnan(a))[0]) > 0:
             a[np.where(np.isnan(a))[0]] = np.repeat(-10, len(np.where(np.isnan(a))[0]))
             x_train1.iloc[:, t] = a


     out_train = np.where(np.isnan(y_train1))[0]
     if len(out_train) > 0:
         y_train1 = y_train1.drop(out_train, axis=0)
         y_train1 = y_train1.reset_index(drop=True)
         x_train1 = x_train1.drop(out_train, axis=0)
         x_train1 = x_train1.reset_index(drop=True)

     #######################################################################################
     ##ANN
     ANN_model = Sequential()
     # The Input Layer :
     ANN_model.add(
         Dense(x_train1.shape[1], kernel_initializer='normal', input_dim=x_train1.shape[1], activation='relu'))
     # The Hidden Layers :
     ANN_model.add(Dense(neurons1, kernel_initializer='normal', activation='relu'))
     ANN_model.add(Dense(neurons2, kernel_initializer='normal', activation='relu'))
     # ANN_model.add(Dense(neurons3, kernel_initializer='normal', activation='relu'))
     # The Output Layer :
     ANN_model.add(Dense(1, kernel_initializer='normal', activation='linear'))
     # Compile the network :
     ANN_model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])
     ANN_model.summary()
     # Checkpoitn callback
     ##es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
     mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
     # Train the model
     time_start = time()
     # ANN_model.fit(x_train1, y_train1, epochs=2000, validation_data=(x_test1, y_test1),callbacks=[es,mc])
     ANN_model.fit(x_train1, y_train1, epochs=200)

     ###################################################################################################

     x_test1 = x_test1.drop(x_test1.columns[range(x_test1.shape[1] - 24, x_test1.shape[1])], axis=1)
     xxx = [0 for x in range(int(50.6 / 0.5))]
     j = 0.5
     for o in range(len(xxx)):
         xxx[o] = np.round(j, 2)
         j = j + 0.5

     yyy = [0 for x in range(int(16.3 / 0.5))]
     j = 0.5
     for o in range(len(yyy)):
         yyy[o] = np.round(j, 2)
         j = j + 0.5
     yyy.append(16.5)

     for u in range(x_test1.shape[0]):
         x_testF = x_test1.iloc[u]
         y_testF = y_test1.iloc[u]

         max = np.zeros((len(yyy), len(xxx)))
         pos_cajas_x = np.array([50.1, 0, 47.1, 4, 20.6, 30.6, 44, 11.5])
         pos_cajas_y = np.array([3, 16, 14.4, 0, 9.4, 15.7, 6.3, 7.8])
         pos_cajas_z = np.array([3.2, 2.4, 3.2, 3, 2.2, 2.5, 2.1, 1.9])
         pos_cajas = pd.concat([pd.DataFrame(pos_cajas_x), pd.DataFrame(pos_cajas_y), pd.DataFrame(pos_cajas_z)],
                               axis=1)
         x_testz = np.array(x_testF)
         for i in range(len(xxx)):
             xxx1 = xxx[i]
             print(i)
             ys = np.zeros((len(yyy), x_test1.shape[1] + 24))
             for w in range(len(yyy)):
                 distanceX_02 = pos_cajas.iloc[0, 0] - xxx1
                 distanceY_02 = pos_cajas.iloc[0, 1] - yyy[w]
                 distanceZ_02 = pos_cajas.iloc[0, 2] - 1.45
                 distanceX_09 = pos_cajas.iloc[1, 0] - xxx1
                 distanceY_09 = pos_cajas.iloc[1, 1] - yyy[w]
                 distanceZ_09 = pos_cajas.iloc[1, 2] - 1.45
                 distanceX_1a = pos_cajas.iloc[2, 0] - xxx1
                 distanceY_1a = pos_cajas.iloc[2, 1] - yyy[w]
                 distanceZ_1a = pos_cajas.iloc[2, 2] - 1.45
                 distanceX_3d = pos_cajas.iloc[3, 0] - xxx1
                 distanceY_3d = pos_cajas.iloc[3, 1] - yyy[w]
                 distanceZ_3d = pos_cajas.iloc[3, 2] - 1.45
                 distanceX_B1 = pos_cajas.iloc[4, 0] - xxx1
                 distanceY_B1 = pos_cajas.iloc[4, 1] - yyy[w]
                 distanceZ_B1 = pos_cajas.iloc[4, 2] - 1.45
                 distanceX_B2 = pos_cajas.iloc[5, 0] - xxx1
                 distanceY_B2 = pos_cajas.iloc[5, 1] - yyy[w]
                 distanceZ_B2 = pos_cajas.iloc[5, 2] - 1.45
                 distanceX_B3 = pos_cajas.iloc[6, 0] - xxx1
                 distanceY_B3 = pos_cajas.iloc[6, 1] - yyy[w]
                 distanceZ_B3 = pos_cajas.iloc[6, 2] - 1.45
                 distanceX_B4 = pos_cajas.iloc[7, 0] - xxx1
                 distanceY_B4 = pos_cajas.iloc[7, 1] - yyy[w]
                 distanceZ_B4 = pos_cajas.iloc[7, 2] - 1.45

                 dist1 = np.array([distanceX_02, distanceX_09, distanceX_1a, distanceX_3d, distanceX_B1, distanceX_B2,
                                   distanceX_B3,
                                   distanceX_B4,
                                   distanceY_02, distanceY_09, distanceY_1a, distanceY_3d, distanceY_B1, distanceY_B2,
                                   distanceY_B3,
                                   distanceY_B4,
                                   distanceZ_02, distanceZ_09, distanceZ_1a, distanceZ_3d, distanceZ_B1, distanceZ_B2,
                                   distanceZ_B3,
                                   distanceZ_B4])

                 dist1 = scalardist.transform(dist1.reshape(-1, 1))
                 x_testz2 = np.concatenate((x_testz, dist1[:, 0]))
                 ys[w, :] = x_testz2
             x_test1F = pd.DataFrame(ys)

             for t in range(x_test1F.shape[1]):
                 a = x_test1F.iloc[:, t]
                 if len(np.where(np.isnan(a))[0]) > 0:
                     a[np.where(np.isnan(a))[0]] = np.repeat(-10, len(np.where(np.isnan(a))[0]))
                     x_test1F.iloc[:, t] = a

             y_pred = ANN_model.predict(x_test1F)
             if var == 'temp':
                 max[:, i] = np.array(scalar_temp.inverse_transform(y_pred))[:, 0]
             elif var == 'hum':
                 max[:, i] = np.array(scalar_hum.inverse_transform(y_pred))[:, 0]

         res = pd.DataFrame(max)
         obj_name = [var + str(np.array(fechas)[u])]

         res.to_csv(str(obj_name) + ".csv")
         ##############################################################################################



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
#time_end = datetime.datetime(2020, 11, 26, 13, 50, 0)
time_end = datetime.datetime(2021, 3, 10, 12, 0, 0)

#Data of the moving cart
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
variables_02= loading(['co2'], 'mhz14', '0x6a02', variables_02, time_end)
names_02 = ['humidity_02','temperature_02','co2_02']
# 0x6a09
variables = []
variables_09= loading(['humidity', 'temperature'], 'sht31d', '0x6a09', variables, time_end)
variables_09= loading(['co2'], 'mhz14', '0x6a09', variables_09, time_end)
names_09 = ['humidity_09','temperature_09','co2_09']
# 0x6a1a
variables = []
variables_1a= loading(['humidity', 'temperature'], 'sht31d', '0x6a1a', variables, time_end)
variables_1a = loading(['co2'], 'mhz14', '0x6a1a', variables_1a, time_end)
names_1a = ['humidity_1a','temperature_1a','co2_1a']
# 0x6a3d
variables = []
variables_3d = loading(['humidity', 'temperature'], 'sht31d', '0x6a3d', variables, time_end)
variables_3d = loading(['co2'], 'mhz14', '0x6a3d', variables_3d, time_end)
names_3d = ['humidity_3d','temperature_3d','co2_3d']
# rpiB1
variables = []
variables_B1 = loading(['humidity', 'temperature'], 'sht31d', 'rpiB1', variables, time_end)
variables_B1 = loading(['co2'], 'mhz14', 'rpiB1', variables_B1, time_end)
names_B1 = ['humidity_B1','temperature_B1','co2_B1']
# rpiB2
variables = []
variables_B2 = loading(['humidity', 'temperature'], 'sht31d', 'rpiB2', variables, time_end)
variables_B2 = loading(['co2'], 'mhz14', 'rpiB2', variables_B2, time_end)
names_B2 = ['humidity_B2','temperature_B2','co2_B2']
# rpiB3
variables = []
variables_B3 = loading(['humidity', 'temperature'], 'sht31d', 'rpiB3', variables, time_end)
variables_B3 = loading(['co2'], 'mhz14', 'rpiB3', variables_B3, time_end)
names_B3 = ['humidity_B3','temperature_B3','co2_B3']
# rpiB4
variables = []
variables_B4 = loading(['humidity', 'temperature'], 'sht31d', 'rpiB4', variables, time_end)
variables_B4 = loading(['co2'], 'mhz14', 'rpiB4', variables_B4, time_end)
names_B4 = ['humidity_B4','temperature_B4','co2_B4']

#Meteo data
variables = []
variables_meteo = loading(['humidity','radiation','temperature'], 'meteo', 'none', variables, time_end)
#variables_meteo = loading(['humidity', 'pressure','radiation','rain','temperature','windeast','windnorth'], 'meteo', 'none', variables, time_end)
names_meteo=['humidity_M','radiation_M','temperature_M']
var_meteo=[]
for u in range(len(variables_meteo)):
    var_meteo.append(position_meteo(variables_meteo[u],len(variables_B4[0])))

#Cart positions
positions = pd.read_csv("positions_new.csv", sep=";", decimal=",")
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

a = np.array(DATA_C)
names = DATA_C.columns
for t in np.array([3,9, 12, 15, 18, 21, 24, 27, 30]):
#for t in np.array([3,9, 12, 15, 18, 21, 24, 27, 30]):
    aa = np.where(np.array(a[:, t], dtype=float) > 800)[0]
    #aa2 = np.where(np.array(a[:, t], dtype=float) < 400)[0]

    if len(aa) > 0:
        a[aa, t] = np.repeat(np.nan, len(aa))

    aa = np.where(np.array(a[:, t], dtype=float) < 400)[0]
    if len(aa) > 0:
        a[aa, t] = np.repeat(400 , len(aa))


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
#dd = pd.DataFrame(np.array(DATA_C, dtype=float))
names = DATA_C.columns
#dd = dd.interpolate(method='linear', limit_direction='forward')

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


DATA_C = ts(DATA_C, 1, range(7,31), DATA_C.shape[0], DATA_C.columns)
DATA_C = DATA_C.reset_index(drop=True)


yearday = DATA_C.iloc[:, 36]

stop = np.where(yearday==323)[0][0]
DATA_C= DATA_C.drop(range(stop), axis=0)
DATA_C = DATA_C.reset_index(drop=True)

yearday = yearday.drop(range(stop), axis=0)
yearday=yearday.reset_index(drop=True)


carrito = DATA_C.iloc[:, range(4)]
temp_I = carrito.iloc[:, 1]
hum_I = carrito.iloc[:, 0]
co2_I = carrito.iloc[:, 3]
press_I = carrito.iloc[:,2]


oo5 = np.where((yearday == 57))[0]
i = oo5[np.array([810,840, 870])-60]
oo=i

DATA_C.iloc[:, 6] = (DATA_C.iloc[:, 6] + 550) / 1000
fechas = DATA_C.iloc[oo, 34]
x_test_I = DATA_C.iloc[oo]
x_test = x_test_I.reset_index(drop=True)
x_train = DATA_C.drop(oo, axis=0)
x_train = x_train.reset_index(drop=True)
x_train = x_train.drop(np.array([x_train.shape[0]-2, x_train.shape[0]-1]), axis=0)
x_train = x_train.reset_index(drop=True)


temp_test = temp_I.iloc[oo]
temp_test = temp_test.reset_index(drop=True)
co2_test = co2_I.iloc[oo]
co2_test= co2_test.reset_index(drop=True)
hum_test = hum_I.iloc[oo]
hum_test = hum_test.reset_index(drop=True)


temp_train = temp_I.drop(oo, axis=0)
temp_train = temp_train.reset_index(drop=True)
co2_train = co2_I.drop(oo, axis=0)
co2_train= co2_train.reset_index(drop=True)
hum_train = hum_I.drop(oo, axis=0)
hum_train = hum_train.reset_index(drop=True)


temp_train = temp_train.drop(np.array([temp_train.shape[0]-2, temp_train.shape[0]-1]), axis=0)
temp_train = temp_train.reset_index(drop=True)
hum_train = hum_train.drop(np.array([hum_train.shape[0]-2, hum_train.shape[0]-1]), axis=0)
hum_train = hum_train.reset_index(drop=True)
co2_train = co2_train.drop(np.array([co2_train.shape[0]-2, co2_train.shape[0]-1]), axis=0)
co2_train = co2_train.reset_index(drop=True)

x_train = x_train.drop(x_train.columns[range(4)], axis=1)
x_test = x_test.drop(x_test.columns[range(4)], axis=1)


names = np.delete(names, np.array([0, 1, 2, 3]))
pos_cajas_x = np.array([50.1, 0, 47.1, 4, 20.6, 30.6, 44, 11.5])
pos_cajas_y = np.array([3, 16, 14.4, 0, 9.4, 15.7, 6.3, 7.8])
pos_cajas_z = np.array([3.2, 2.4, 3.2, 3, 2.2, 2.5, 2.1, 1.9])
pos_cajas = pd.concat([pd.DataFrame(pos_cajas_x), pd.DataFrame(pos_cajas_y)], axis=1)
pos_cajasT = pd.concat([pd.DataFrame(pos_cajas_x), pd.DataFrame(pos_cajas_y), pd.DataFrame(pos_cajas_z)], axis=1)
####################################################################################################################

dayofyear = x_train.iloc[:, 32]
dayofyear_F = x_test.iloc[:, 32]
x_train = x_train.drop(x_train.columns[33], axis=1)
x_test = x_test.drop(x_test.columns[33], axis=1)
# distances to fixed devices
posit = x_train.iloc[:, range(3)]
posit_test = x_test.iloc[:, range(3)]

distanceX_02, distanceX_02t = pd.DataFrame(pos_cajasT.iloc[0, 0] - posit.iloc[:, 0]), pd.DataFrame(
    pos_cajasT.iloc[0, 0] - posit_test.iloc[:, 0])
distanceY_02, distanceY_02t = pd.DataFrame(pos_cajasT.iloc[0, 1] - posit.iloc[:, 1]), pd.DataFrame(
    pos_cajasT.iloc[0, 1] - posit_test.iloc[:, 1])
distanceZ_02, distanceZ_02t = pd.DataFrame(pos_cajasT.iloc[0, 2] - posit.iloc[:, 2]), pd.DataFrame(
    pos_cajasT.iloc[0, 2] - posit_test.iloc[:, 2])
distanceX_09, distanceX_09t = pd.DataFrame(pos_cajasT.iloc[1, 0] - posit.iloc[:, 0]), pd.DataFrame(
    pos_cajasT.iloc[1, 0] - posit_test.iloc[:, 0])
distanceY_09, distanceY_09t = pd.DataFrame(pos_cajasT.iloc[1, 1] - posit.iloc[:, 1]), pd.DataFrame(
    pos_cajasT.iloc[1, 1] - posit_test.iloc[:, 1])
distanceZ_09, distanceZ_09t = pd.DataFrame(pos_cajasT.iloc[1, 2] - posit.iloc[:, 2]), pd.DataFrame(
    pos_cajasT.iloc[1, 2] - posit_test.iloc[:, 2])
distanceX_1a, distanceX_1at = pd.DataFrame(pos_cajasT.iloc[2, 0] - posit.iloc[:, 0]), pd.DataFrame(
    pos_cajasT.iloc[2, 0] - posit_test.iloc[:, 0])
distanceY_1a, distanceY_1at = pd.DataFrame(pos_cajasT.iloc[2, 1] - posit.iloc[:, 1]), pd.DataFrame(
    pos_cajasT.iloc[2, 1] - posit_test.iloc[:, 1])
distanceZ_1a, distanceZ_1at = pd.DataFrame(pos_cajasT.iloc[2, 2] - posit.iloc[:, 2]), pd.DataFrame(
    pos_cajasT.iloc[2, 2] - posit_test.iloc[:, 2])
distanceX_3d, distanceX_3dt = pd.DataFrame(pos_cajasT.iloc[3, 0] - posit.iloc[:, 0]), pd.DataFrame(
    pos_cajasT.iloc[3, 0] - posit_test.iloc[:, 0])
distanceY_3d, distanceY_3dt = pd.DataFrame(pos_cajasT.iloc[3, 1] - posit.iloc[:, 1]), pd.DataFrame(
    pos_cajasT.iloc[3, 1] - posit_test.iloc[:, 1])
distanceZ_3d, distanceZ_3dt = pd.DataFrame(pos_cajasT.iloc[3, 2] - posit.iloc[:, 2]), pd.DataFrame(
    pos_cajasT.iloc[3, 2] - posit_test.iloc[:, 2])
distanceX_B1, distanceX_B1t = pd.DataFrame(pos_cajasT.iloc[4, 0] - posit.iloc[:, 0]), pd.DataFrame(
    pos_cajasT.iloc[4, 0] - posit_test.iloc[:, 0])
distanceY_B1, distanceY_B1t = pd.DataFrame(pos_cajasT.iloc[4, 1] - posit.iloc[:, 1]), pd.DataFrame(
    pos_cajasT.iloc[4, 1] - posit_test.iloc[:, 1])
distanceZ_B1, distanceZ_B1t = pd.DataFrame(pos_cajasT.iloc[4, 2] - posit.iloc[:, 2]), pd.DataFrame(
    pos_cajasT.iloc[4, 2] - posit_test.iloc[:, 2])
distanceX_B2, distanceX_B2t = pd.DataFrame(pos_cajasT.iloc[5, 0] - posit.iloc[:, 0]), pd.DataFrame(
    pos_cajasT.iloc[5, 0] - posit_test.iloc[:, 0])
distanceY_B2, distanceY_B2t = pd.DataFrame(pos_cajasT.iloc[5, 1] - posit.iloc[:, 1]), pd.DataFrame(
    pos_cajasT.iloc[5, 1] - posit_test.iloc[:, 1])
distanceZ_B2, distanceZ_B2t = pd.DataFrame(pos_cajasT.iloc[5, 2] - posit.iloc[:, 2]), pd.DataFrame(
    pos_cajasT.iloc[5, 2] - posit_test.iloc[:, 2])
distanceX_B3, distanceX_B3t = pd.DataFrame(pos_cajasT.iloc[6, 0] - posit.iloc[:, 0]), pd.DataFrame(
    pos_cajasT.iloc[6, 0] - posit_test.iloc[:, 0])
distanceY_B3, distanceY_B3t = pd.DataFrame(pos_cajasT.iloc[6, 1] - posit.iloc[:, 1]), pd.DataFrame(
    pos_cajasT.iloc[6, 1] - posit_test.iloc[:, 1])
distanceZ_B3, distanceZ_B3t = pd.DataFrame(pos_cajasT.iloc[6, 2] - posit.iloc[:, 2]), pd.DataFrame(
    pos_cajasT.iloc[6, 2] - posit_test.iloc[:, 2])
distanceX_B4, distanceX_B4t = pd.DataFrame(pos_cajasT.iloc[7, 0] - posit.iloc[:, 0]), pd.DataFrame(
    pos_cajasT.iloc[7, 0] - posit_test.iloc[:, 0])
distanceY_B4, distanceY_B4t = pd.DataFrame(pos_cajasT.iloc[7, 1] - posit.iloc[:, 1]), pd.DataFrame(
    pos_cajasT.iloc[7, 1] - posit_test.iloc[:, 1])
distanceZ_B4, distanceZ_B4t = pd.DataFrame(pos_cajasT.iloc[7, 2] - posit.iloc[:, 2]), pd.DataFrame(
    pos_cajasT.iloc[7, 2] - posit_test.iloc[:, 2])

names = x_train.columns
x_train = pd.concat([x_train, pd.DataFrame(distanceX_02), pd.DataFrame(distanceX_09), pd.DataFrame(distanceX_1a), pd.DataFrame(distanceX_3d),
     pd.DataFrame(distanceX_B1), pd.DataFrame(distanceX_B2), pd.DataFrame(distanceX_B3),
     pd.DataFrame(distanceX_B4),
     pd.DataFrame(distanceY_02), pd.DataFrame(distanceY_09), pd.DataFrame(distanceY_1a), pd.DataFrame(distanceY_3d), pd.DataFrame(distanceY_B1),
     pd.DataFrame(distanceY_B2), pd.DataFrame(distanceY_B3),
     pd.DataFrame(distanceY_B4),
     pd.DataFrame(distanceZ_02), pd.DataFrame(distanceZ_09), pd.DataFrame(distanceZ_1a), pd.DataFrame(distanceZ_3d), pd.DataFrame(distanceZ_B1), pd.DataFrame(distanceZ_B2), pd.DataFrame(distanceZ_B3),
     pd.DataFrame(distanceZ_B4)], axis=1)
x_test = pd.concat(
    [x_test, pd.DataFrame(distanceX_02t), pd.DataFrame(distanceX_09t), pd.DataFrame(distanceX_1at), pd.DataFrame(distanceX_3dt),
     pd.DataFrame(distanceX_B1t), pd.DataFrame(distanceX_B2t), pd.DataFrame(distanceX_B3t),
     pd.DataFrame(distanceX_B4t),
     pd.DataFrame(distanceY_02t), pd.DataFrame(distanceY_09t), pd.DataFrame(distanceY_1at), pd.DataFrame(distanceY_3dt), pd.DataFrame(distanceY_B1t),
     pd.DataFrame(distanceY_B2t), pd.DataFrame(distanceY_B3t),
     pd.DataFrame(distanceY_B4t),
     pd.DataFrame(distanceZ_02t), pd.DataFrame(distanceZ_09t), pd.DataFrame(distanceZ_1at), pd.DataFrame(distanceZ_3dt), pd.DataFrame(distanceZ_B1t), pd.DataFrame(distanceZ_B2t), pd.DataFrame(distanceZ_B3t),
     pd.DataFrame(distanceZ_B4t)], axis=1)
nn = ["distanceX_02", "distanceX_09", "distanceX_1a", "distanceX_3d", "distanceX_B1", "distanceX_B2",
      "distanceX_B3", "distanceX_B4",
      "distanceY_02", "distanceY_09", "distanceY_1a", "distanceY_3d", "distanceY_B1", "distanceY_B2",
      "distanceY_B3", "distanceY_B4",
      "distanceZ_02", "distanceZ_09", "distanceZ_1a", "distanceZ_3d", "distanceZ_B1", "distanceZ_B2",
      "distanceZ_B3", "distanceZ_B4"]
names = np.concatenate([names, nn])
x_train.columns = names
x_test.columns = names

x_train = x_train.drop(['pos_z', 'pos_x', 'pos_y'], axis=1)
x_test = x_test.drop(['pos_z', 'pos_x', 'pos_y'], axis=1)


#MTI_train(x_train, temp_train, x_test, temp_test,30, 20,0,fechas,'temp')
MTI_train(x_train, hum_train, x_test, hum_test,110, 10,0,fechas,'hum')
