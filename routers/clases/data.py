from pydantic import BaseModel
from typing import Optional, List
from routers.clases.errors import Errors
    
class Data(BaseModel):
    input: List[float]
    sampling: int
    fecha_inicio: str