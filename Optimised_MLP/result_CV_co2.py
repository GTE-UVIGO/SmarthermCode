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


GLOBAL_SEED = 1995

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
from scipy import interpolate
import joblib
from random import randint
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from time import time
#from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import seaborn as sns; sns.set()
from tensorflow.compat.v1.keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestRegressor
from tensorflow.compat.v1.keras.callbacks import ModelCheckpoint
from sklearn import svm
from sklearn import metrics
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score



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
    mbe = np.mean(y_true-y_pred)/med
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
    pred_value.columns = names[pred_col]
    pred_value = pd.DataFrame(pred_value)

    pred_value['id'] = range(1, len(pred_value) + 1)
    pred_value.set_index('id', inplace=True)
    final_df = pd.concat([t, pred_value], axis=1)

    return final_df

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
            for t in range(len(point)):
                values[t]=point[t][var_name[u]]

            variables.append(values)

    return variables

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
    return(x)


def MTI_train_cv(x_train, y_train, neurons1, neurons2, neurons3, positX, positY,med,var):
    x_train1 = x_train.reset_index(drop=True)
    y_train1 = y_train.reset_index(drop=True)
    error = [0 for x in range(len(positX))]
    nmbe =[0 for x in range(len(positX))]
    rmse = [0 for x in range(len(positX))]


    outs=[0 for x in range(len(positX))]
    for u in range(len(positX)):
        yy= np.where(x_train1.iloc[:,0]==positX[u])
        yy2=np.where(x_train1.iloc[:,1]==positY[u])
        yyy = np.intersect1d(yy, yy2)

        if len(yyy)>10:
            x_test = x_train1.iloc[yyy]
            x_test = x_test.reset_index(drop=True)
            y_test = y_train1.iloc[yyy]
            y_test = y_test.reset_index(drop=True)
            x_t = x_train1.drop(yyy, axis=0)
            x_t = x_t.reset_index(drop=True)
            y_t = y_train1.drop(yyy, axis=0)
            y_t = y_t.reset_index(drop=True)

            X_test1=x_test.drop(x_test.columns[np.array([0,1,2])], axis=1)
            X_final = x_t.drop(x_t.columns[np.array([0, 1, 2])], axis=1)
            names= X_final.columns


            temperaturas_train = np.array(X_final.iloc[:, np.array([1, 4, 7, 10, 13, 16, 19, 22, 26, 31, 34, 37, 40, 43, 46, 49, 52])])
            temperaturas_test = np.array(X_test1.iloc[:, np.array([1, 4, 7, 10, 13, 16, 19, 22, 26, 31, 34, 37, 40, 43, 46, 49, 52])])


            humedad_train = np.array(X_final.iloc[:, np.array([0, 3, 6, 9,12, 15, 18, 21, 24, 30, 33, 36, 39,42, 45, 48, 51])])
            humedad_test = np.array(X_test1.iloc[:, np.array([0, 3, 6, 9,12, 15, 18, 21, 24, 30, 33, 36, 39,42, 45, 48, 51])])


            co2s_train = np.array(X_final.iloc[:, np.array([2, 5, 8, 11, 14, 17, 20, 23, 32, 35, 38, 41, 44, 47, 50, 53])])
            co2s_test = np.array(X_test1.iloc[:, np.array([2, 5, 8, 11,14, 17, 20, 23, 32, 35, 38, 41,44, 47, 50, 53])])

            diss_train = np.array(X_final.iloc[:, range(X_final.shape[1] - 24, X_final.shape[1])])
            diss_test = np.array(X_test1.iloc[:, range(X_final.shape[1] - 24, X_final.shape[1])])

            rad_train = np.array(X_final.iloc[:, np.array([25])])
            rad_test = np.array(X_test1.iloc[:, np.array([25])])


            resto_train = X_final.iloc[:, np.array([27, 28, 29])]
            resto_test = X_test1.iloc[:, np.array([27, 28, 29])]

            scalar_temp = MinMaxScaler(feature_range=(-1, 1))
            scalar_hum = MinMaxScaler(feature_range=(-1, 1))
            scalar_co2 = MinMaxScaler(feature_range=(-1, 1))
            scalardist = MinMaxScaler(feature_range=(-1, 1))
            scalar_rad = MinMaxScaler(feature_range=(-1, 1))
            scalarresto = MinMaxScaler(feature_range=(-1, 1))


            scalar_temp.fit(
                np.concatenate((np.concatenate(temperaturas_train), np.concatenate(temperaturas_test))).reshape(-1, 1))
            scalardist.fit(np.concatenate((np.concatenate(diss_train), np.concatenate(diss_test))).reshape(-1, 1))
            scalar_hum.fit(
                np.concatenate((np.concatenate(humedad_train), np.concatenate(humedad_test))).reshape(-1, 1))
            scalar_co2.fit(np.concatenate((np.concatenate(co2s_train).reshape(-1,1), np.concatenate(co2s_test).reshape(-1,1), np.array(y_t).reshape(-1,1),np.array(y_test).reshape(-1,1))).reshape(-1, 1))
            scalar_rad.fit(np.concatenate((np.concatenate(rad_train), np.concatenate(rad_test))).reshape(-1, 1))
            scalarresto.fit(pd.concat([resto_train, resto_test], axis=0))
            y_t = pd.DataFrame(scalar_co2.transform(np.array(y_t).reshape(-1,1)))
            y_test = pd.DataFrame(scalar_co2.transform(np.array(y_test).reshape(-1,1)))

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


            x_t = pd.concat([co2s_train1, resto_train1, diss_train1], axis=1)
            x_test = pd.concat([co2s_test1,  resto_test1, diss_test1], axis=1)

            miss = x_t.apply(lambda x: x.count(), axis=1) - 43
            miss = np.where(miss <= -2)[0]
            x_t = x_t.drop(miss, axis=0)
            x_t = x_t.reset_index(drop=True)
            y_t = y_t.drop(miss, axis=0)
            y_t = y_t.reset_index(drop=True)

            miss = x_test.apply(lambda x: x.count(), axis=1) - 43
            miss = np.where(miss <= -2)[0]
            x_test = x_test.drop(miss, axis=0)
            x_test = x_test.reset_index(drop=True)
            y_test = y_test.drop(miss, axis=0)
            y_test = y_test.reset_index(drop=True)

            x_t.columns = range(x_t.shape[1])
            x_test.columns =  range(x_test.shape[1])

            for t in range(x_t.shape[1]):
                a = x_t.iloc[:, t]
                if len(np.where(np.isnan(a))[0]) > 0:
                    a[np.where(np.isnan(a))[0]] = np.repeat(-10, len(np.where(np.isnan(a))[0]))
                    x_t.iloc[:, t] = a
            for t in range(x_test.shape[1]):
                a = x_test.iloc[:, t]
                if len(np.where(np.isnan(a))[0]) > 0:
                    a[np.where(np.isnan(a))[0]] = np.repeat(-10, len(np.where(np.isnan(a))[0]))
                    x_test.iloc[:, t] = a
            out_train = np.where(np.isnan(y_t))[0]
            if len(out_train) > 0:
                y_t = y_t.drop(out_train, axis=0)
                y_t = y_t.reset_index(drop=True)
                x_t = x_t.drop(out_train, axis=0)
                x_t = x_t.reset_index(drop=True)

            out_T = np.where(np.isnan(y_test))[0]
            if len(out_T) > 0:
                y_test = y_test.drop(out_T, axis=0)
                y_test = y_test.reset_index(drop=True)
                x_test = x_test.drop(out_T, axis=0)
                x_test = x_test.reset_index(drop=True)

            if x_test.shape[0] > 1:
                ANN_model = Sequential()
                # The Input Layer :
                ANN_model.add(
                    Dense(x_t.shape[1], kernel_initializer='normal', input_dim=x_t.shape[1], activation='relu'))
                # The Hidden Layers :
                ANN_model.add(Dense(neurons1, kernel_initializer='normal', activation='relu'))
                # The Output Layer :
                ANN_model.add(Dense(1, kernel_initializer='normal', activation='linear'))
                # Compile the network :
                ANN_model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])
                ANN_model.summary()
                # Checkpoitn callback
                es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
                mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
                # Train the model
                time_start = time()
                ANN_model.fit(x_t, y_t, epochs=2000, validation_data=(x_test, y_test), callbacks=[es, mc])
                ##################################################################################################
                y_pred = ANN_model.predict(pd.DataFrame(x_test))
                y_pred = np.array(scalar_co2.inverse_transform(y_pred))
                y_test2 = np.array(scalar_co2.inverse_transform(y_test))

                error[u] = 100 * (np.sqrt(metrics.mean_squared_error(y_test2, y_pred)) / med)
                nmbe[u] = MBE(y_test2, y_pred,med)
                rmse[u] = np.sqrt(metrics.mean_squared_error(y_test2, y_pred))
            else:
                error[u] =0
                nmbe[u] = 0
                rmse[u] = 0
    ##############################################################################################
    error=np.array(error)
    print(error)
    nmbe= np.array(nmbe)
    rmse = np.array(rmse)
    error_T = np.mean(error[error!=0])
    sd_error = np.std(error[error!=0])
    nmbe_T = np.mean(nmbe[nmbe!=0])
    sd_nmbe = np.std(nmbe[nmbe!=0])
    rmse_T = np.mean(rmse[rmse!=0])
    sd_rmse = np.std(rmse[rmse!=0])

    print('CV RMSE es', error_T)
    print('Desviación CV RMSE es', sd_error)
    print('NMBE medio es', nmbe_T)
    print('Desviación NMBE es', sd_nmbe)
    print('RMSE medio es', rmse_T)
    print('Desviación RMSE es', sd_rmse)
    print('position fuera por muchos nas',outs)


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


def MTI_total(DATA_C, var):
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
    D1.iloc[:, 3] = y_co2.iloc[:,0]
    stop = np.where(yearday == 323)[0][0]
    D1 = D1.drop(range(stop), axis=0)
    D1 = D1.reset_index(drop=True)
    yearday = yearday.drop(range(stop), axis=0)
    yearday = yearday.reset_index(drop=True)

    carrito = D1.iloc[:, range(4)]
    temp_I = carrito.iloc[:, 1]
    hum_I = carrito.iloc[:, 0]
    co2_I = carrito.iloc[:, 3]
    press_I = carrito.iloc[:, 2]

    x_train = D1
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

    pos_cajas_x = np.array([50.1, 0, 47.1, 4, 20.6, 30.6, 44, 11.5])
    pos_cajas_y = np.array([3, 16, 14.4, 0, 9.4, 15.7, 6.3, 7.8])
    pos_cajas_z = np.array([3.2, 2.4, 3.2, 3, 2.2, 2.5, 2.1, 1.9])
    pos_cajas = pd.concat([pd.DataFrame(pos_cajas_x), pd.DataFrame(pos_cajas_y)], axis=1)
    pos_cajasT = pd.concat([pd.DataFrame(pos_cajas_x), pd.DataFrame(pos_cajas_y), pd.DataFrame(pos_cajas_z)], axis=1)

    dayofyear = x_train.iloc[:, 32]
    x_train = x_train.drop(x_train.columns[33], axis=1)

    # distances to fixed devices
    posit = x_train.iloc[:, range(3)]

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

    names = x_train.columns
    x_train = pd.concat(
        [x_train, distanceX_02, distanceX_09, distanceX_1a, distanceX_3d, distanceX_B1, distanceX_B2, distanceX_B3,
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
    x_train.columns = names


    med_temp = np.nanmean(np.array(temp_I).reshape(-1,1))
    med_hum= np.nanmean(np.array(hum_I).reshape(-1,1))
    med_co2 = np.nanmean(np.array(co2_I).reshape(-1,1))

    y_train = co2_I

    positX = np.array([9.8, 12.4, 7.3, 7.3,28.9, 39.8, 40.9, 19.7, 4, 45.5, 27.9, 28.7, 46.8, 47.7, 36.4, 21.5, 0.3, 0.4,
        17.6, 17.6, 32.2, 2.3])
    positY = np.array([7.8, 14.7, 14.7, 11.7, 1.6, 1.6, 11.6, 5, 11.2, 5.6, 14.6, 7.8, 2.2, 1.5, 1.6, 7.5, 6.4, 14.6,
         6.8, 1.6, 4.7, 4.7])

    ##########################################################################
    yy1 = np.where((x_train.iloc[:, 0] == 6.9) | (x_train.iloc[:, 0] == 26))[0]
    yy2 = np.where((x_train.iloc[:, 1] == 4) | (x_train.iloc[:, 1] == 14.55))[0]

    yy3 = np.where((x_train.iloc[:, 0] == 46.3) | (x_train.iloc[:, 0] == 28.8))[0]
    yy4 = np.where((x_train.iloc[:, 1] == 7.6) | (x_train.iloc[:, 1] == 10.1))[0]

    zz1 = np.intersect1d(yy1, yy2)
    zz2 = np.intersect1d(yy3, yy4)
    zz1 = np.sort(np.concatenate((zz1, zz2)))

    y_train = y_train.drop(zz1, axis=0)
    y_train = y_train.reset_index(drop=True)

    x_train = x_train.drop(zz1, axis=0)
    x_train = x_train.reset_index(drop=True)

    #################################################################################################


    MTI_train_cv(x_train, y_train, 20, 0, 0, positX, positY, med_co2,var)


######################################################################################################################
######################################################################################################################
# Loading of data from the different hosts
variables = []
time_end = datetime.datetime(2021, 3, 10, 12, 0, 0)

#############################################################################
#############################################################################
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
variables_02 = loading(['co2'], 'mhz14', '0x6a02', variables_02, time_end)
names_02 = ['humidity_02','temperature_02','co2_02']
# 0x6a09
variables = []
variables_09 = loading(['humidity', 'temperature'], 'sht31d', '0x6a09', variables, time_end)
variables_09 = loading(['co2'], 'mhz14', '0x6a09', variables_09, time_end)
names_09 = ['humidity_09','temperature_09','co2_09']
# 0x6a1a
variables = []
variables_1a = loading(['humidity', 'temperature'], 'sht31d', '0x6a1a', variables, time_end)
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
variables_meteo = loading(['humidity', 'radiation','temperature'], 'meteo', 'none', variables, time_end)
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

#######################################################################################################
MTI_total(DATA_C, 'co2')