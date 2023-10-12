import numpy as np
from PyEMD import CEEMDAN
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

import os

from datetime import datetime

from scipy.signal import savgol_filter

from pandas import to_datetime as pd_to_datetime
from pandas import merge as pd_merge
from pandas import DataFrame as pd_DataFrame
from pandas import read_csv as pd_read_csv
from pandas import concat as pd_concat
from pandas import to_timedelta as pd_to_timedelta
from pandas import read_excel as pd_read_excel
from pandas import melt as pd_melt

from numpy import unique as np_unique
from numpy import nan as np_nan

from xgboost import XGBRegressor 
from holidays import Spain


from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

from glob import glob

from pickle import dumps





def tratamiento_nulos_por_horas(df, nom_variable_pred, drop_horas):
    df = df.drop(df[df.hora.isin(drop_horas)].index)
    df=df.reset_index(drop=True)

    median_horas = df.groupby(by='hora').median().demanda

    nulos_hora = df[df.isnull().any(axis=1)]
    for i, hora in enumerate(np_unique(df.hora)):
        df.loc[nulos_hora.loc[nulos_hora.hora == hora, nom_variable_pred].index, nom_variable_pred] = median_horas[i]
    return df

def merge_dfs(fecha, df_temperatura, df_demanda):
    df_final = pd_merge(df_temperatura[df_temperatura.Date.dt.date == fecha],
                        df_demanda[df_demanda.datetime.dt.date == fecha], 
                        how='outer',left_on='Date', right_on='datetime').dropna(axis=0)
    return df_final

def get_all_files(directory):
    pattern = directory + '/*'
    files = glob(pattern)
    return files


def lectura_metereologia(path, name_date_metereologia):
    files = get_all_files(path)
    data_appended = []
    for file in files:
        df_data = pd_read_csv(file)
        data_appended.append(df_data)
    df = pd_concat(data_appended, ignore_index=True)
    df[name_date_metereologia] = pd_to_datetime(df[name_date_metereologia],format='%Y-%m-%d')
    return df

def creacion_variables_horarias(df, name_date_metereologia):
    df = df.drop(df[df.hora == '24:00'].index, axis=0)
    df[name_date_metereologia] = df[name_date_metereologia].astype(str)
    df['hora'] = df['hora'].astype(str)
    df[name_date_metereologia] = pd_to_datetime(df[name_date_metereologia] + ' ' + df['hora'], format="%Y-%m-%d %H:%M")
    df['hora'] = df[name_date_metereologia].dt.hour
    df['Date'] = df[name_date_metereologia].dt.date
    return df

def temperature_max_and_avg(df):
    df['TemperatureMax'] = np_nan
    df['TemperatureAvg'] = np_nan
    dias = np_unique(df.Date.dt.date)
    temperatura_max = df.groupby('fecha').max().Temperature.values
    temperatura_mean = df.groupby('fecha').mean().Temperature.values

    for i, dia in enumerate(dias):
        df.loc[df.Date.dt.date == dia, 'TemperatureMax'] = df.loc[df.Date.dt.date == dia, 'TemperatureMax'].fillna(round(temperatura_max[i], 2))
        df.loc[df.Date.dt.date == dia, 'TemperatureAvg'] = df.loc[df.Date.dt.date == dia, 'TemperatureAvg'].fillna(round(temperatura_mean[i], 2))
    return df

def add_holidays(df, zona_festivo, holidays_years):
    festivos = Spain(years=holidays_years, subdiv=zona_festivo).items()
    df_festivos = pd_DataFrame(festivos, columns = ['Date', 'Festivo'])
    df_festivos['Date'] = pd_to_datetime(df_festivos['Date'], format='%Y-%m-%d')
    df = pd_merge(df_festivos,df, how='outer',left_on='Date', right_on='fecha')
    df.loc[df.notnull().all(axis=1),'Festivo'] = 1
    df.loc[df.isnull().any(axis=1), 'Festivo'] = 0
    try:
        df = df.drop('Date_x', axis = 1)
    finally:
        df = df.rename(columns={'Date_y': 'Date'})
        df['Festivo'] = df['Festivo'].astype(int)
        df = df.dropna()
        df = df.reindex(sorted(df.columns), axis=1)
        return df
    
def merge_df_data_metereologia(df_metereologia, df_data):
    df_data.fecha = pd_to_datetime(df_data.datetime.dt.date.values, format="%Y-%m-%d")
    fechas = np_unique(df_metereologia.Date.dt.date.values)
    df = []
    for fecha in fechas:
        df_merged = merge_dfs(fecha, df_metereologia, df_data)
        df.append(df_merged)

    df = pd_concat(df, axis=0)
    df = df.dropna()
    return df

def split_data_ultimos_dias(df, dias_test):
    ultimos_dias = np_unique(df.Date.dt.date)[-dias_test:]
    X_train = df[~df.Date.dt.date.isin(ultimos_dias)]
    X_test = df[df.Date.dt.date.isin(ultimos_dias)]
    return X_train, X_test

def preprocesamiento():
    name_datetime_data = 'Date'
    path_data = 'Data'
    df_data = lectura_datos(path_data)

    # Tratamiento valores nulos
    nom_variable_pred = 'demanda'
    drop_horas = [2,10]
    df_data = tratamiento_nulos_por_horas(df_data, nom_variable_pred, drop_horas)


    # Lectura de los datos metereológicos
    name_datetime_metereologia = 'fecha'
    path_metereologia = 'Tiempo'
    df_metereologia = lectura_metereologia(path_metereologia, name_datetime_metereologia)


    # Eliminacion de columnas irrelevantes y conversiones


    df_metereologia = df_metereologia[['fecha','hora','Temperature']]

    medida_temperatura='kelvin' #celsius
    if medida_temperatura == 'kelvin':
        df_metereologia.Temperature = df_metereologia.Temperature.apply(lambda x: x - 273.15)


    # Tratamiento de datos horarios

    df_metereologia = creacion_variables_horarias(df_metereologia, name_datetime_metereologia)


    # Agregacion de los datos y calculo de la media de cada uno de ellos

    df_metereologia.Date = df_metereologia.Date.astype(str)
    df_metereologia = df_metereologia.groupby(by=['Date', 'hora'], as_index=False).mean()
    df_metereologia.Date = pd_to_datetime(df_metereologia['Date']) + pd_to_timedelta(df_metereologia['hora'], unit='h')
    df_metereologia = df_metereologia.drop('hora', axis = 1)


    # Merge de los datos metereologicos y los datos de demanda. key=Date
    df_final = merge_df_data_metereologia(df_metereologia, df_data)


    # Cálculo y agregación de la Temperatura maxima y media
    #df_final = temperature_max_and_avg(df_final)
    # Añadir festivos
    holidays_years = [2019, 2021, 2022, 2023]
    zona_festivo = 'MC'
    df_final = add_holidays(df_final, zona_festivo, holidays_years)

    return df_final



def applyer_weekend(row):
    if row == 6 or row == 7:
        return 1
    else:
        return 0



def inicializacion_df_variables(df, sort):
    df['hora'] = df['hora'].str.replace('\D+', '', regex=True)
    df = df[df['fecha'] != 'fecha']
    #df['datetime_text'] = df['fecha'] + ' ' + df['hora']
    df['datetime'] = pd_to_datetime(df['fecha'] + ' ' + df['hora'], format="%d/%m/%Y %H")
    df = df.drop_duplicates(subset=['datetime'])
    df['year'] = df['datetime'].dt.year.astype('int32')
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['weekday'] = df['datetime'].dt.weekday+1
    df['hour'] = df['datetime'].dt.hour
    df['weekend']=df['weekday'].apply(applyer_weekend)
    if sort:
        df.sort_values('datetime', inplace=True)
    return df

def lectura_datos(path):
    root_data = path
    files = get_all_files(root_data)
    data_appended = []
    for file in files:
        df_data = pd_read_excel(file)
        df_unpivot = pd_melt(
            df_data,
            id_vars = 'fecha',
            value_vars = df_data.columns.to_list().remove('fecha'),
            var_name = 'hora',
            value_name = 'demanda'
        )
        data_appended.append(df_unpivot[1:])
        df_data_loaded = pd_concat(data_appended, ignore_index=True)
        df_data_loaded = df_data_loaded.dropna(subset=['fecha'])

    df_data_loadedd = inicializacion_df_variables(df_data_loaded, sort=True)
    return df_data_loadedd

def split_data_all(df_train, df_test,nom_variable_pred,sav_filter=True):


    drop_columns_training = ['demanda','Date','datetime','fecha','hora','year']

    X_train = df_train.drop(columns=drop_columns_training)
    X_test = df_test.drop(columns=drop_columns_training)
    
    y_train = df_train[nom_variable_pred]
    if sav_filter:
        y_train = savgol_filter(y_train, 5,2)

    y_test = df_test[nom_variable_pred]
    
    return X_train, y_train, X_test, y_test
    
    
def save_file_with_path(file_path, data_to_save):
    try:
        with open(file_path, 'w') as file:
            file.write(data_to_save)
        print(f"File saved successfully at: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")    
    
    
def train(planta, empresa):
    df_final = preprocesamiento()
    dias_test = 14
    df_train, df_test = split_data_ultimos_dias(df_final, dias_test)

    df_train = df_train.sort_values(by='Date').reset_index(drop=True)
    df_test = df_test.sort_values(by='Date').reset_index(drop=True)

    X_train, y_train, X_test, y_test = split_data_all(df_train, df_test, nom_variable_pred='demanda',sav_filter=True)
    


# # Ajuste de hiperparámtros

    metric_objetive = 'reg:squarederror'
    model = XGBRegressor(verbosity=1)

    type_model = type(model).__name__
    
    # Grid para GridSearchCV
    param_grid = {
        'objective': [metric_objetive],
        'booster': ['gbtree'],
        'n_estimators': [500],
        'max_depth': [6],
        'learning_rate': [0.01],
        'gamma': [0],
        'subsample': [1],
        'colsample_bytree': [1]
        # Add moX_crossperparameters and their respective values to search over
    }


    # #   GridSearchCV con cross-validation
    grid_search = GridSearchCV(estimator=model,param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error', verbose=1,error_score='raise')
    grid_search.fit(X_train, y_train) 


    # # Obtener los mejores hiperparámetros obtenidos por GridSearch

    best_params = grid_search.best_params_
    # best_score = -grid_search.best_score_

    print("Best Hyperparameters:", best_params)
    # print(f"Best MSE Score: {best_score}")


    # # Crear nuevo modelo a partir de los mejores parámetros obtenidos

    best_model = XGBRegressor(
        objective= metric_objetive,#mse_loss
        n_estimators =     best_params['n_estimators'],
        max_depth =        best_params['max_depth'],
        learning_rate =    best_params['learning_rate'],
        gamma =            best_params['gamma'],
        random_state = 42
    )


    # # Aplicar cross-validation al nuevo modelo

    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')


    best_model.fit(X_train, y_train)
    
    
    # Creacion modelo
    
    model_id = dumps(best_model)

    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    error_train = mean_squared_error(y_train, y_train_pred)
    error_test = mean_squared_error(y_test, y_test_pred)
    error_metric = 'RMSE'
    
    fecha_creacion = str(datetime.now())
    modelo = model_id
    hiperparametros = best_params
    version = "0"    
    
    return fecha_creacion, modelo, hiperparametros, version, type_model, error_train, error_test, error_metric


