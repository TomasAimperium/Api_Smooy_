from pydantic import BaseModel

from typing import List

from routers.clases.data import Data
from routers.clases.modelo import Modelo
from routers.clases.empresa import Empresa
from routers.clases.planta import Planta

class Predict(BaseModel):
    data_predict: Data
    
class PredictPlanta(BaseModel):
    empresa: Empresa
    planta: Planta
    modelos: dict