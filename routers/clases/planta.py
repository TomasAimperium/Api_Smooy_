from pydantic import BaseModel

from typing import Optional

from routers.clases.empresa import Empresa
from routers.clases.modulo import Modulo

class Planta(BaseModel):
    #id: int 
    nombre_planta: str
    #empresa: Optional[Empresa]
    #modulo: Optional[Modulo]