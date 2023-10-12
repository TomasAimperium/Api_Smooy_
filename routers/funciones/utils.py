from fastapi import HTTPException, status

from typing import List

from routers.clases.planta import Planta
from routers.clases.empresa import Empresa

def buscar_empresa_id(id : str, empresas_list: List[Empresa]):
    empresas = filter(lambda x: x.id == id, empresas_list)
    try:
        return list(empresas)[0]
    except:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Empresa no encontrada")
    
    
    
def plantas_por_empresa(id: int, plantas_list: List[Planta]):
    plantas = []
    for planta in plantas_list:
        if planta.empresa.id == id:
            plantas.append(planta)
    
    return plantas


def buscar_planta_id(plantas, id_planta):
    planta = ""
    for planta_actual in plantas:
        if planta_actual.id == id_planta:
            planta = planta_actual
    if planta == "":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, 
                            detail="Esta planta no existe")
    return planta


def buscar_empresa(nombre_empresa : str, empresas_list: List[Empresa]):
    empresas = filter(lambda x: x.nombre_empresa == nombre_empresa, empresas_list)
    try:
        return list(empresas)[0]
    except:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Empresa no encontrada")