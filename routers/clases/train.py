from pydantic import BaseModel

from typing import List

from routers.clases.data import Data
from routers.clases.modelo import ModeloId
from routers.clases.empresa import Empresa
from routers.clases.planta import Planta
from routers.clases.errors import Errors

from bson import ObjectId

class Train(BaseModel):
    fecha_entrenamiento: str
    data_train: dict
    errores: Errors
    modelo: dict #modelo
    

class EntrenamientoPlanta(BaseModel):
    empresa: Empresa
    planta: Planta
    entrenamientos: dict