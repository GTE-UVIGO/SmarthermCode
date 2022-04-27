import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn import preprocessing
import seaborn as sns; sns.set()
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn import svm
from sklearn import metrics
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from pickle import load
import pickle
import influxdb
import datetime
import time
from datetime import datetime, timedelta



class TempError(Exception):
    pass

class HumiError(Exception):
    pass

class CO2Error(Exception):
    pass


#Load and make the predictions of the artificial neural netowrks
def predictions_RF(filename,X_test,var):
    names = X_test.columns
    loaded_model = pickle.load(open(filename, 'rb'))
    print("Loaded model from disk")

    y = [filename, 'Scaler-Y']
    temp = [filename, 'Scaler_temp']
    hum = [filename, 'Scaler_hum']
    co2 = [filename, 'Scaler_co2']
    dist = [filename, 'Scaler_dist']
    rad = [filename, 'Scaler_rad']
    resto = [filename, 'Scaler_resto']

    sep = '-'
    scaler_Y = load(open(sep.join(y), 'rb'))
    scaler_temp = load(open(sep.join(temp), 'rb'))
    scaler_hum = load(open(sep.join(hum), 'rb'))
    scaler_co2 = load(open(sep.join(co2), 'rb'))
    scalerdist = load(open(sep.join(dist), 'rb'))
    scaler_rad = load(open(sep.join(rad), 'rb'))
    scalerresto = load(open(sep.join(resto), 'rb'))

    X_test=X_test.transpose()
    temperaturas_test = np.array(X_test.iloc[:, np.array([1, 4, 7, 10, 13, 16, 19, 22, 26, 31, 34, 37, 40, 43, 46, 49, 52])])
    humedad_test = np.array(X_test.iloc[:, np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 30, 33, 36, 39, 42, 45, 48, 51])])
    co2s_test = np.array(X_test.iloc[:, np.array([2, 5, 8, 11, 14, 17, 20, 23, 32, 35, 38, 41, 44, 47, 50, 53])])
    #diss_test = np.array(X_test.iloc[:, range(X_test.shape[1] - 24, X_test.shape[1])])
    rad_test = np.array(X_test.iloc[:, np.array([25])])
    resto_test = X_test.iloc[:, np.array([27, 28, 29])]

    temperaturas_test1 = np.zeros((temperaturas_test.shape[0], temperaturas_test.shape[1]))
    humedad_test1 = np.zeros((humedad_test.shape[0], temperaturas_test.shape[1]))
    for i in range(temperaturas_test1.shape[1]):
        temperaturas_test1[:, i] = scaler_temp.transform(temperaturas_test[:, i].reshape(-1, 1))[:, 0]
        humedad_test1[:, i] = scaler_hum.transform(humedad_test[:, i].reshape(-1, 1))[:, 0]

    temperaturas_test1 = pd.DataFrame(temperaturas_test1)
    humedad_test1 = pd.DataFrame(humedad_test1)

    co2s_test1 = np.zeros((co2s_test.shape[0], co2s_test.shape[1]))
    for i in range(co2s_test1.shape[1]):
        co2s_test1[:, i] = scaler_co2.transform(co2s_test[:, i].reshape(-1, 1))[:, 0]

    co2s_test1 = pd.DataFrame(co2s_test1)

    #diss_test1 = np.zeros((diss_test.shape[0], diss_train.shape[1]))
    #for i in range(diss_test1.shape[1]):
    #    diss_test1[:, i] = scalardist.transform(diss_test[:, i].reshape(-1, 1))[:, 0]
#
    rad_test1 = np.zeros((rad_test.shape[0], rad_test.shape[1]))
    for i in range(rad_test1.shape[1]):
        rad_test1[:, i] = scaler_rad.transform(rad_test[:, i].reshape(-1, 1))[:, 0]

    #diss_test1 = pd.DataFrame(diss_test1)
    rad_test1 = pd.DataFrame(rad_test1)
    resto_test1 = pd.DataFrame(scalerresto.transform(resto_test))

    try:
        if var == 'temp':
            x_test1 = pd.concat([temperaturas_test1, rad_test1, resto_test1], axis=1)

            miss = x_test1.apply(lambda x: x.count(), axis=1) - 21
            if miss[0] <= -6:
                raise TempError(print('Too missing values in temperatures'))


        elif var == 'hum':
            x_test1 = pd.concat([humedad_test1, rad_test1, resto_test1], axis=1)

            miss = x_test1.apply(lambda x: x.count(), axis=1) - 21
            if miss[0] <= -6:
                raise HumiError(print('Too missing values in relative humidity'))

        else:
            x_test1 = pd.concat([co2s_test1, resto_test1], axis=1)

            miss = x_test1.apply(lambda x: x.count(), axis=1) - 19
            if miss[0] <= -6:
                raise CO2Error(print('Too missing values in co2'))

        x_test1.columns = range(x_test1.shape[1])

        for t in range(x_test1.shape[1]):
            a = x_test1.iloc[:, t]
            if len(np.where(np.isnan(a))[0]) > 0:
                a[np.where(np.isnan(a))[0]] = np.repeat(-10, len(np.where(np.isnan(a))[0]))
                x_test1.iloc[:, t] = a

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

        max = np.zeros((len(yyy), len(xxx)))
        pos_cajas_x = np.array([50.1, 0, 47.1, 4, 20.6, 30.6, 44, 11.5])
        pos_cajas_y = np.array([3, 16, 14.4, 0, 9.4, 15.7, 6.3, 7.8])
        pos_cajas_z = np.array([3.2, 2.4, 3.2, 3, 2.2, 2.5, 2.1, 1.9])
        pos_cajas = pd.concat([pd.DataFrame(pos_cajas_x), pd.DataFrame(pos_cajas_y), pd.DataFrame(pos_cajas_z)],
                              axis=1)
        x_testz = np.array(x_test1)
        for i in range(len(xxx)):
            xxx1 = xxx[i]
            print(i)
            ys = np.zeros((len(yyy), x_testz.shape[1] + 24))
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

                dist1 = np.array(
                    [distanceX_02, distanceX_09, distanceX_1a, distanceX_3d, distanceX_B1, distanceX_B2,
                     distanceX_B3,
                     distanceX_B4,
                     distanceY_02, distanceY_09, distanceY_1a, distanceY_3d, distanceY_B1, distanceY_B2,
                     distanceY_B3,
                     distanceY_B4,
                     distanceZ_02, distanceZ_09, distanceZ_1a, distanceZ_3d, distanceZ_B1, distanceZ_B2,
                     distanceZ_B3,
                     distanceZ_B4])
                dist1 = scalerdist.transform(dist1.reshape(-1, 1))
                x_testz2 = np.concatenate((x_testz[0, :], dist1[:, 0]))
                ys[w, :] = x_testz2
            x_test1F = pd.DataFrame(ys)

            # for j in range(x_test1F.shape[0]):
            #     #Possibility of missing values
            #     if any(np.array(np.isnan(x_test1F.iloc[j,:]))):
            #         ii = np.where(np.isnan(x_test1F.iloc[j,:]))[0]
            #         x_test1F.iloc[ii] = -10

            # q=[filename, 'json']
            # sep='.'
            # json_file = open(sep.join(q), 'r')
            # loaded_model_json = json_file.read()
            # json_file.close()
            # loaded_model = model_from_json(loaded_model_json)
            ## load weights into new model
            # q=[filename, 'h5']
            # loaded_model.load_weights(sep.join(q))
            # print("Loaded model from disk")
            #
            # loaded_model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])
            # y_pred = loaded_model.predict(np.array(X_scaled))
            # result = scaler_Y.inverse_transform(y_pred)

            result = loaded_model.predict(np.array(x_test1F))
            if var == 'temp':
                max[:, i] = np.array(scaler_temp.inverse_transform(result.reshape(-1,1)))[:, 0]
            elif var == 'hum':
                max[:, i] = np.array(scaler_hum.inverse_transform(result.reshape(-1,1)))[:, 0]
            else:
                max[:, i] = np.array(scaler_co2.inverse_transform(result.reshape(-1,1)))[:, 0]

        res = pd.DataFrame(max)

    except TempError:
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

        res = np.zeros((len(yyy), len(xxx)))
    except HumiError:
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

        res = np.zeros((len(yyy), len(xxx)))
    except CO2Error:
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

        res = np.zeros((len(yyy), len(xxx)))



    return res

#Load and make the predictions of the random forest models
#def predictions_RF(filename,X_test):
#    y = [filename, 'Scaler-Y']
#    sep = '-'
#    scaler_Y = load(open(sep.join(y), 'rb'))
#
#    for i in range(X_test.shape[0]):
#        #Possibility of missing values
#        if any(np.isnan(X_test.iloc[i,:])):
#            ii = np.where(np.isnan(X_test.iloc[i,:]))[0]
#            X_test[ii] = 0
#
#    scalarX = StandardScaler()
#    scalarX.fit(X_test)
#    X_scaled = pd.DataFrame(scalarX.transform(pd.DataFrame(X_test)))
#
#    # load the model from disk
#    loaded_model = pickle.load(open(filename, 'rb'))
#    result = loaded_model.predict(np.array(X_scaled))
#    print("Loaded model from disk")
#
#    result= scaler_Y.inverse_transform(result)
#    return result

def loading(var_name, sensor_name, host, variables,variablesP, time_end, time_ini):
    #info influx

    influx = influxdb.InfluxDBClient.from_dsn(CONN_STR)
    influx.switch_database(DB_NAME)

    place = ["sensor_data.autogen",sensor_name]
    sep='.'
    place = sep.join(place)
    place2 = [sensor_name,"address"]
    sep = '.'
    place2 = sep.join(place2)
    time_end= pd.Timestamp(time_end.astimezone()).tz_convert(tz="UTC")
    time_end_str = time_end.strftime('%Y-%m-%dT%H:%M:%SZ')
    time_ini = pd.Timestamp(time_ini.astimezone()).tz_convert("UTC")
    time_ini_str = time_ini.strftime('%Y-%m-%dT%H:%M:%SZ')

    if host=='none':
        for u in range(len(var_name)):

            var2 = [var_name[u], 'vc']
            sep = '_'
            var2 = sep.join(var2)

            query = f"""
                        SELECT LAST(*) FROM {place}
                       WHERE time > '{time_ini_str}'
                    AND {place2} != '69' FILL(null)
                    """
            results = influx.query(query)
            if len(results)==0:
                minutes = timedelta(minutes=30)
                time_ini = time_ini -minutes
                #time_ini = time_ini + one_minute
                time_ini_str = time_ini.strftime('%Y-%m-%dT%H:%M:%SZ')

                query = f"""
                                       SELECT LAST(*) FROM {place}
                                      WHERE time > '{time_ini_str}'
                                   AND {place2} != '69' FILL(null)
                                   """
                results = influx.query(query)

            point = list(results)[0]
            values = [0 for x in range(len(point))]
            for t in range(len(point)):
                values[t] = point[t][var_name[u]]


            #val1 = np.nanmean(values)
            #variablesP.append(val1)
#
            #one_minute = timedelta(minutes=1)
            #time_end = time_end + one_minute
            #time_ini = time_ini + one_minute
            #time_end_str = time_end.strftime('%Y-%m-%dT%H:%M:%SZ')
            #time_ini_str = time_ini.strftime('%Y-%m-%dT%H:%M:%SZ')
#
            #query = f"""
            #                SELECT {var_name[u]} FROM {place}
            #                WHERE time > '{time_ini_str}' AND time < '{time_end_str}' AND {var2}<3
            #                AND {place2} != '69' FILL(null)
            #            """
#
            #results = influx.query(query)
            #point = list(results)[0]
            #values = [0 for x in range(len(point))]
            #for t in range(len(point)):
            #    values[t] = point[t][var_name[u]]
#
            #val2 = np.nanmean(values)
            variablesP=[]
            variables.append(values)

    else:

        for u in range(len(var_name)):

            query = f"""
                        SELECT {var_name[u]} FROM {place} 
                            WHERE time >'{time_ini_str}' AND time <'{time_end_str}' AND {place2} != '69' AND "host"='{host}' FILL(null) 
                        """
           #query = f"""
           #            SELECT temperature FROM {place}
           #                WHERE time >'2021-05-03T11:01:00 UTC' AND time <'2021-05-03T11:07:00 UTC' AND {place2} != '69' AND "host"='{host}' FILL(null)
           #            """

            results = influx.query(query)
            if len(results)==0:
                val1 = np.nan
            else:
                point = list(results)[0]
                values = [0 for x in range(len(point))]
                for t in range(len(point)):
                    values[t] = point[t][var_name[u]]

                val1 = np.nanmean(values)

            variablesP.append(val1)


            one_minute = timedelta(minutes=1)
            time_end = time_end + one_minute
            time_ini = time_ini + one_minute
            time_end_str = time_end.strftime('%Y-%m-%dT%H:%M:%SZ')
            time_ini_str = time_ini.strftime('%Y-%m-%dT%H:%M:%SZ')

            query = f"""
                            SELECT {var_name[u]}  FROM {place} 
                            WHERE time > '{time_ini_str}' AND time < '{time_end_str}'  AND "host"='{host}'
                              AND {place2} != '69'  FILL(null)
                        """

            results = influx.query(query)
            if len(results)==0:
                val2 = np.nan
            else:
                point = list(results)[0]
                values = [0 for x in range(len(point))]
                for t in range(len(point)):
                    values[t] = point[t][var_name[u]]

                val2 = np.nanmean(values)

            variables.append(val2)
            #dates = pd.to_datetime(pd.Series(dates), format='%Y-%m-%dT%H:%M:%SZ')

    return variables, variablesP


#Load the data
#def loading(var_name, sensor_name, host, variables, time_end, time_ini):
    #info influx
#    influx = influxdb.InfluxDBClient.from_dsn(CONN_STR)
#    influx.switch_database(DB_NAME)
#    time_end = datetime.datetime.now()
#    time_end_str = time_end.strftime('%Y-%m-%dT%H:%M:%SZ')
#    time_ini_str = time_ini.strftime('%Y-%m-%dT%H:%M:%SZ')
#
#
#    place = ["sensor_data.autogen",sensor_name]
#    sep='.'
#    place = sep.join(place)
#
#    if host=='none':
#        query = f"""
#            SELECT LAST(*) FROM {place}
#            WHERE time > '2020-07-01T20:00:00Z' AND time < '{time_ini_str}'
#        """
#
#        results = influx.query(query)
#        point = list(results)[0][0]
#
#        query = f"""
#                    SELECT LAST(*) FROM {place}
#                    WHERE time > '2020-07-01T20:00:00Z' AND time < '{time_end_str}'
#                """
#
#        results2 = influx.query(query)
#        point2 = list(results)[0][0]
#
#        for u in range(len(var_name)):
#            variables.append(point[var_name[u]])
#
#    else:
#        query = f"""
#            SELECT LAST(*) FROM {place}
#            WHERE time > '2020-07-01T20:00:00Z' AND time < '{time_ini_str}' AND "host"='{host}'
#        """
#
#        results = influx.query(query)
#        point = list(results)[0][0]
#
#        query = f"""
#                    SELECT LAST(*) FROM {place}
#                    WHERE time > '2020-07-01T20:00:00Z' AND time < '{time_ini_str}' AND "host"='{host}'
#                """
#
#        results2 = influx.query(query)
#        point = list(results)[0][0]
#
#        for u in range(len(var_name)):
#            variables.append(point[var_name[u]])
#
#    return variables

#Make the predictions based on 'selection'
def RESULTS (DATE,DATE_past, selection):
    #0x6a02
    variables=[]
    variablesP = []
    variables_02,variablesP_02 =loading(['humidity','temperature'],'sht31d', '0x6a02', variables,variablesP,DATE, DATE_past)
    variables_02, variablesP_02=loading(['co2'],'mhz14' , '0x6a02', variables_02,variablesP_02, DATE, DATE_past)
    #0x6a09
    variables=[]
    variablesP = []
    variables_09,variablesP_09 =loading(['humidity','temperature'],'sht31d', '0x6a09', variables,variablesP,DATE, DATE_past)
    variables_09, variablesP_09=loading(['co2'],'mhz14' , '0x6a09', variables_09,variablesP_09, DATE, DATE_past)
    #0x6a1a
    variables=[]
    variablesP = []
    variables_1a,variablesP_1a =loading(['humidity','temperature'],'sht31d', '0x6a1a', variables,variablesP,DATE, DATE_past)
    variables_1a, variablesP_1a=loading(['co2'],'mhz14' , '0x6a1a', variables_1a,variablesP_1a, DATE, DATE_past)
    #0x6a3d
    variables=[]
    variablesP = []
    variables_3d,variablesP_3d =loading(['humidity','temperature'],'sht31d', '0x6a3d', variables,variablesP,DATE, DATE_past)
    variables_3d, variablesP_3d=loading(['co2'],'mhz14' , '0x6a3d', variables_3d,variablesP_3d, DATE, DATE_past)
    #rpiB1
    variables=[]
    variablesP = []
    variables_B1,variablesP_B1 =loading(['humidity','temperature'],'sht31d', 'rpiB1', variables,variablesP,DATE, DATE_past)
    variables_B1, variablesP_B1=loading(['co2'],'mhz14' , 'rpiB1', variables_B1,variablesP_B1, DATE, DATE_past)
    #rpiB2
    variables=[]
    variablesP = []
    variables_B2,variablesP_B2 =loading(['humidity','temperature'],'sht31d', 'rpiB2', variables,variablesP,DATE, DATE_past)
    variables_B2, variablesP_B2=loading(['co2'],'mhz14' , 'rpiB2', variables_B2,variablesP_B2, DATE, DATE_past)
    #rpiB3
    variables=[]
    variablesP = []
    variables_B3,variablesP_B3 =loading(['humidity','temperature'],'sht31d', 'rpiB3', variables,variablesP,DATE, DATE_past)
    variables_B3, variablesP_B3=loading(['co2'],'mhz14' , 'rpiB3', variables_B3,variablesP_B3, DATE, DATE_past)
    #rpiB4
    variables=[]
    variablesP = []
    variables_B4,variablesP_B4 =loading(['humidity','temperature'],'sht31d', 'rpiB4', variables,variablesP,DATE, DATE_past)
    variables_B4, variablesP_B4=loading(['co2'],'mhz14' , 'rpiB4', variables_B4,variablesP_B4, DATE, DATE_past)
    #Meteo data
    variables = []
    variablesP = []
    variables_meteo,variablesP_meteo = loading(['last_humidity','last_radiation','last_temperature'], 'meteo', 'none', variables,variablesP, DATE, DATE_past)

    variables=pd.DataFrame(np.concatenate([np.array(variables_02).transpose(),np.array(variables_09).transpose(),np.array(variables_1a).transpose(), np.array(variables_3d).transpose(), np.array(variables_B1).transpose(),
                                           np.array(variables_B2).transpose(),np.array(variables_B3).transpose(),np.array(variables_B4).transpose(), np.array(variables_meteo)[:,0].transpose()]))
    variablesP = pd.DataFrame(np.concatenate([np.array(variablesP_02).transpose(),np.array(variablesP_09).transpose(),np.array(variablesP_1a).transpose(), np.array(variablesP_3d).transpose(), np.array(variablesP_B1).transpose(),
                                           np.array(variablesP_B2).transpose(),np.array(variablesP_B3).transpose(),np.array(variablesP_B4).transpose()]))

    #X_test=np.concatenate(variables)
    Hour = DATE.hour + DATE.minute / 100
    dd = pd.date_range(DATE, periods=1, freq='1min')
    #Day= dd.dayofyear*24 + dd.hour + dd.minute / 100
    Day= dd.dayofyear
    Week= DATE.weekday()

    #X_test = pd.concat([pd.DataFrame(pos_x), pd.DataFrame(pos_y), pd.DataFrame(pos_z),pd.DataFrame(np.array([[X_test],]*n)[:,0,:]), pd.DataFrame(np.repeat(Hour, n)), pd.DataFrame(np.repeat(Week, n)), pd.DataFrame(np.repeat(Day, n)) ], axis=1)
    X_test = pd.concat([variables, pd.Series(Hour), pd.Series(Week), pd.Series(Day), variablesP])
    X_test=X_test.reset_index(drop=True)
    if selection=='temperature':
        #Model temperature
        filename='model_temp_trained'
        temp_prediction = predictions_RF(filename, X_test,'temp')
        print('Temperature interpolation made')
        return(np.array(temp_prediction))
    if selection == 'co2':
        #Model CO2
        filename='model_co2_trained'
        co2_prediction = predictions_RF(filename, X_test,'co2')
        print('CO2 interpolation made')
        return(np.array(co2_prediction))
    if selection == 'humidity':
        #Model humidity
        filename='model_hum_trained'
        humidity_prediction = predictions_RF(filename, X_test,'hum')
        print('Humidity interpolation made')
        return(np.array(humidity_prediction))

######################################################################################################################
#Parameters example
#pos_x=np.array([33.7, 5.7, 40,7,23.7, 0])
#pos_y=np.array([3.7, 11, 30, 5,23.7,33.7])
#pos_z=np.array([355, 555,255, 155, 75, 255])
DATE=datetime.now()
DATE = DATE.replace(second=0,microsecond=0)
one_minute = timedelta(minutes=1)
DATE_past = DATE - one_minute
selection=['co2', 'humidity','temperature']


time.sleep(10)
results= []
#General function to  make predictions depending on the 'selection'
for i in range(len(selection)):
    results.append(RESULTS(DATE,DATE_past, selection[i]))