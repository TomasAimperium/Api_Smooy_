from fastapi import APIRouter, status, HTTPException
from routers.funciones.utils_train import *

from routers.clases.empresa import Empresa
from routers.clases.planta import Planta

from routers.empresa import empresas_list

from routers.funciones.utils import buscar_empresa_id, plantas_por_empresa, buscar_planta_id

import copy

router = APIRouter()

plantas_list = [Planta(id = 0,empresa = Empresa(id=0,nombre_empresa='smooy'), nombre_planta = 'Alfonso X'),
                Planta(id = 1,empresa = Empresa(id=0, nombre_empresa='smooy'), nombre_planta = 'SantoDomingo'),
                Planta(id = 2,empresa = Empresa(id=1, nombre_empresa='ainostrum'), nombre_planta = 'Murcia')]


@router.get('/empresa/{id_empresa}/planta/{id_planta}')
async def get_planta(id_empresa: int, id_planta: int):
    buscar_empresa_id(id_empresa, empresas_list)
    plantas = plantas_por_empresa(id_empresa, plantas_list)
    planta = buscar_planta_id(plantas, id_planta)
   
    return planta

@router.get('/empresa/{id}/plantas')
async def get_plantas_empresa(id: int):
    buscar_empresa_id(id, empresas_list)
    plantas = []
    for planta in plantas_list:
        if planta.empresa.id == id:
            plantas.append(planta)
        
    return plantas

    
@router.post('/empresa/{id}/planta')
async def crear_plantas(id: int, nueva_planta: Planta, status_code=status.HTTP_201_CREATED):

    # Comprobar si existe la empresa con id
    empresa = buscar_empresa_id(id, empresas_list)
    
    nueva_planta.empresa = Empresa(id=empresa.id, 
                                   nombre_empresa=empresa.nombre_empresa)
    
    # Todas las plantas de la empresa
    plantas = plantas_por_empresa(empresa.id, plantas_list)
    
    # Comprobar si esta planta existe
    
    
    plantas = list(filter(lambda x: x.nombre_planta == nueva_planta.nombre_planta, plantas))
    
    if len(plantas) != 0:
        raise HTTPException(status_code=status.HTTP_204_NO_CONTENT, detail="La planta ya esta registrada")
    
    # Añadir la planta a la base de datos
    plantas_list.append(nueva_planta)
  
    return nueva_planta








    