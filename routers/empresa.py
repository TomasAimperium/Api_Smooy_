from fastapi import APIRouter, HTTPException, status

from routers.clases.empresa import Empresa
from routers.funciones.utils import buscar_empresa


from routers.schemas.empresa import empresas_schema, empresa_schema
from host import db

router = APIRouter()


empresas_list = [Empresa(id = 0, nombre_empresa = 'smooy'),
                  Empresa(id = 1, nombre_empresa = 'ainostrum')]

db["empresas"].drop()
db_empresas = db["empresas"]

#db_empresas.insert_one({"nombre_empresa": "smooy"})

def buscar_empresa(nombre_empresa : str):
    #empresas = filter(lambda x: x.nombre_empresa == nombre_empresa, db.empresas)
    
    
    empresas = db_empresas.find_one({"nombre_empresa": nombre_empresa})
    
    if empresas == None:
        return 0
    else:
        return list(empresas)[0]
    

@router.get('/empresas')
async def get_empresas():
    return empresas_schema(db_empresas.find())


@router.post('/empresa', status_code=status.HTTP_201_CREATED)
async def crear_empresa(nueva_empresa: Empresa):
    
    empresa = buscar_empresa(nueva_empresa.nombre_empresa) # Consulta base de datos
    
    empresa = db_empresas.find_one({"nombre_empresa": nueva_empresa.nombre_empresa})
    
    if empresa != None:
        raise HTTPException(status_code=status.HTTP_204_NO_CONTENT, detail='La empresa ya existe')
         
    
    empresa_dict = dict(nueva_empresa)
    del empresa_dict["id"]
    
    id = db_empresas.insert_one(empresa_dict).inserted_id

    new_empresa = empresa_schema(db_empresas.find_one({"_id": id}))

    return Empresa(**new_empresa)