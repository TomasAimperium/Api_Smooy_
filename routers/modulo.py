from fastapi import APIRouter, HTTPException, status
from routers.clases.modulo import Modulo
from routers.planta import plantas_list

from routers.funciones.utils import buscar_empresa_id, plantas_por_empresa, buscar_planta_id

from routers.empresa import empresas_list

router = APIRouter()



@router.get("/modulos")
def get_all_modulos():
    modulos = []
    for plt in plantas_list:
        modulos.append(plt.modulo)
    return modulos


@router.post('/{id_empresa}/{id_planta}/modulo')
async def crear_modulo(id_empresa: int, id_planta: int, modulo_nuevo: Modulo):
    
    buscar_empresa_id(id_empresa, empresas_list)
    
    plantas = plantas_por_empresa(id_empresa, plantas_list)
    planta = buscar_planta_id(plantas, id_planta)
 
    
    if planta.modulo != None:
        raise HTTPException(status_code=status.HTTP_204_NO_CONTENT, 
                            detail = f"Esta planta ya tiene el módulo {modulo_nuevo.nombre_modulo} creado")
    
    planta.modulo = modulo_nuevo
    
    return modulo_nuevo