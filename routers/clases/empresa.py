from pydantic import BaseModel
from typing import Optional
class Empresa(BaseModel):
    #id: Optional[str] 
    nombre_empresa: str