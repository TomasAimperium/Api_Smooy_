from pandas import DataFrame as pd_DataFrame
from pandas import concat as pd_concat

from requests import get as requests_get


def call_api_meteo(key:str) -> dict:
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/murcia?unitGroup=metric&key={key}&contentType=json"

    response = requests_get(url)

    response_json = response.json()
    return response_json

def data_extraction_meteo(json_file: pd_DataFrame)-> pd_DataFrame:
    
    # Recorrer json para extraer los datos
    for apartado, contenido in json_file.items():
        if (apartado == 'days'):
            if not isinstance(contenido, dict):
                arr = contenido
                
    # Juntar los datos extraidos del json por horas
    df = []
    for i in range(0, len(arr)):
        dia_actual = pd_DataFrame(arr[i]['hours'])
        dia_actual['fecha'] = arr[i]['datetime']
        df.append(dia_actual)
    df = pd_concat(df)
    
    return df