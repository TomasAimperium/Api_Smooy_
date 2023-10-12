from pydantic import BaseModel
from routers.clases.modelo import Modelo
from typing import Optional

class Modulo(BaseModel):
    id: int
    nombre_modulo : str
    modelo : Optional[Modelo]