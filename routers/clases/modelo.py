from pydantic import BaseModel

from bson import ObjectId

class Modelo(BaseModel):
    fecha_creacion: str
    tipo: str
    hiperparametros: dict
    version: str
    serial: bytes
    
class ModeloId(BaseModel):
    _id: ObjectId