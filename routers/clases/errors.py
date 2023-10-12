
from pydantic import BaseModel
from typing import Optional

class Errors(BaseModel):
    train:Optional[float]
    test: float
    metric: str