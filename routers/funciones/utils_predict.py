from pandas import DataFrame as pd_DataFrame

from pandas import to_datetime as pd_to_datetime
from pandas import merge as pd_merge

from numpy import unique as np_unique
from numpy import ndarray as np_ndarray

from holidays import Spain
from pickle import loads

from routers import meteo_api


def dias_festivos(years:  np_ndarray, zona_festivo: str, df: pd_DataFrame) -> pd_DataFrame:
    df['fecha'] = pd_to_datetime(df['fecha'],format='%Y-%m-%d')
    festivos = Spain(years=years, subdiv=zona_festivo).items()
    df_festivos = pd_DataFrame(festivos, columns = ['Date', 'Festivo'])
    df_festivos['Date'] = pd_to_datetime(df_festivos['Date'], format='%Y-%m-%d')

    df = pd_merge(df_festivos, df, how='outer',left_on='Date', right_on='fecha')
    df.loc[df.notnull().all(axis=1),'Festivo'] = 1
    df.loc[df.isnull().any(axis=1), 'Festivo'] = 0
    df = df.drop('Date', axis = 1)
    df['Festivo'] = df['Festivo'].astype(int)
    
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df

def sort_values_per_fecha(df: pd_DataFrame) -> pd_DataFrame:
    df = df.sort_values('fecha').reset_index(drop=True)
    return df

def applyer_weekend(row) -> int:
    if row == 6 or row == 7:
        return 1
    else:
        return 0
    
def add_variables_horarias(df: pd_DataFrame) -> pd_DataFrame:

    df['hora'] = df['fecha'].dt.hour
    df['month'] = df.fecha.dt.month
    df['day'] = df.fecha.dt.day
    df['weekday'] = df.fecha.dt.weekday + 1
    df['weekend'] = df['weekday'].apply(applyer_weekend)
    df['year'] = df['fecha'].dt.year
    
    return df


def predict_14_dias(path_modelo):
    KEY_API = '3D27UYNFFMDRGZ2V5558F6LZC'

    # API meteo
    response_json = meteo_api.call_api_meteo(KEY_API)
    df = meteo_api.data_extraction_meteo(response_json)

    # Variables que necesitamos
    columnas_metereologicas = ['fecha', 'datetime', 'temp']
    df = df[columnas_metereologicas]

    df['fecha'] = pd_to_datetime(df['fecha'],format='%Y-%m-%d')

    # Añadir los festivos
    years_festivos = np_unique(df.fecha.dt.year)
    df = dias_festivos(years_festivos,'MC', df)

    # Tratamiento fechas y ordenar datos
    df['fecha'] = pd_to_datetime(df['fecha'].astype(str) + ' ' + df['datetime'], format="%Y-%m-%d %H:%M")
    df = sort_values_per_fecha(df)

    # Añadir variables horarias
    df = add_variables_horarias(df)

    # Seleccionamos el horario de la tienda
    horario_horas = [0,12,13,14,15,16,17,18,19,20,21,22,23] #############

    # Descartamos aquellas horas que no esten en el horario de la tienda
    df = sort_values_per_fecha(df)
    df = df[df.hora.isin(horario_horas)]
    df = sort_values_per_fecha(df)

    df_fechas = df.fecha.values
    
    # Eliminamos aquellas variables que no queremos para la prediccion
    df = df.drop(['fecha','datetime','year'],axis=1)



    # Recordar que los datos que le pasamos para la prediccion las columnas 
    # tienen que tener el mismo orden y nombre que las columnas de los datos 
    # con las que se entrenó el modelo

    # Todos estos nombres deberian de estar guardados en un fichero aparte
    # ¿Se puede extraer las variables del modelo?

    df = df.rename(columns={'hora': 'hour',
                            'temp': 'Temperature'})

    df = df.reindex(sorted(df.columns), axis=1)
    df = df.sort_index(axis=1)



    # Cargamos el modelo y se realiza las predicciones
    model = loads(path_modelo)

    predicciones = model.predict(df)

    # # En este caso, la demanda no puede ser negativa.
    # # Por tanto hacemos la trasnformación de que toda predicción < 0, sea 0
    predicciones = [x if x >= 0 else 0 for x in predicciones]
    
    # json no soporta las dependencias de tipos de numpy
    predicciones = [float(item) for item in predicciones]
    
    
    
    
    return df_fechas, predicciones
