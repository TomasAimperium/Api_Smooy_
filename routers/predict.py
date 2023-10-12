from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

import copy

from bson import ObjectId

from datetime import datetime
from routers.clases.errors import Errors
from routers.clases.data import Data
from routers.clases.modelo import Modelo, ModeloId
from routers.clases.train import EntrenamientoPlanta, Train
from routers.clases.predict import PredictPlanta, Predict

from routers.funciones.utils import buscar_empresa_id, plantas_por_empresa, buscar_planta_id
from routers.funciones.utils_train import train
from routers.funciones.utils_predict import predict_14_dias

from routers.empresa import empresas_list
from routers.planta import plantas_list

from routers.clases.empresa import Empresa
from routers.clases.planta import Planta

from routers.schemas.entrenamiento import entrenamientos_schema, entrenamiento_schema


from host import db

from pydantic import BaseModel

router = APIRouter()

db_train_models = db["train_models"]
db_predict_data = db["predict_data"]
db_modelos = db["modelos"]
db_data_train = db["data_train"]


@router.get("/entrenamientos")
def entrenamientos():
    return entrenamientos_schema(db_train_models.find())


@router.post("/prediccion")
async def predicccion(empresa: Empresa, planta: Planta):
    
    empresa_planta = db_train_models.find_one({"empresa.nombre_empresa": empresa.nombre_empresa,
                               "planta.nombre_planta": planta.nombre_planta})
    if empresa_planta == None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,detail="La planta no tiene modelo")
    
    ultimo_elemento = empresa_planta["entrenamientos"].popitem()
    _, valor = ultimo_elemento

    id_modelo = valor["modelo"]
    
    modelo = db_modelos.find_one({"_id": id_modelo["_id"]})
    
    
    print(f'SERIAL_TYPE: {type(modelo["serial"])}')
    fechas, predicciones = predict_14_dias(modelo["serial"])
    #dict_predicciones = dict(zip(fechas.astype(str), predicciones))
    data_test= Data(input=predicciones, sampling=60, fecha_inicio=str(fechas[0]))
    
    empresa_planta = db_predict_data.find_one({"empresa.nombre_empresa": empresa.nombre_empresa,
                    "planta.nombre_planta": planta.nombre_planta})
    
    new_predict = Predict(data_predict=data_test).dict()
    fecha_prediccion = str(int(datetime.now().timestamp()*1000))
    
    new_predict_one = {str(id_modelo["_id"]):{"_id": id_modelo["_id"],
                   "predicciones": {fecha_prediccion:new_predict}}
    }
  
    if empresa_planta == None:
        print("empresa_planta NONE :(")
    
        test_db = PredictPlanta(empresa=Empresa(nombre_empresa=empresa.nombre_empresa),
                                            planta=Planta(nombre_planta=planta.nombre_planta),
                                            modelos=new_predict_one
                                ).dict()
        db_predict_data.insert_one(dict(test_db))
        return {"message": f"Primeras predicciones de la planta {planta.nombre_planta} y empresa {empresa.nombre_empresa}"}
    
    
    data = empresa_planta["modelos"]
    id_modelo_id = id_modelo["_id"]
    print("MODELOS")
    print(data.keys())
    print("ID_MODELO")
    print(id_modelo_id)
    
    

    if str(id_modelo_id) in data.keys():
        # existe el modelo, añadir prediccion
        test_db = db_predict_data.update_one(empresa_planta, {"$set":{f"modelos.{str(id_modelo_id)}.predicciones.{fecha_prediccion}":new_predict}})
        return {"message: Prediccion añadido al modelo YA EXISTENTE"}
    else:
        # no existe modelo, añadir modelo con prediccion
        test_db = db_predict_data.update_one(empresa_planta, {"$set":{f"modelos.{str(id_modelo_id)}": {"_id": id_modelo["_id"],
                                                                                                        "predicciones": {fecha_prediccion:new_predict}
                                                                                                        }
                                                                      }
                                                              })
        return {"message: Prediccion con NUEVO modelo"}



@router.post("/entrenamiento")
async def entrenamiento(empresa: Empresa, planta: Planta):
    
    ###################################
    ###################################
    data_train = db_data_train.find_one({})
    ###################################
    ###################################
    
    fecha_creacion, serial_modelo, hiperparametros, version, type_model, error_train, error_test, error_metric = train(planta.nombre_planta,
                                                                                                                        empresa.nombre_empresa)
    

    errores = Errors(train=error_train, 
                                test=error_test, 
                                metric=error_metric)

    modelo= Modelo(fecha_creacion=fecha_creacion, 
                                    serial=serial_modelo,
                                    hiperparametros=hiperparametros, 
                                    version=version, 
                                    tipo=type_model).dict()

    id_model = db_modelos.insert_one(dict(modelo)).inserted_id
    
    print(f"ID_MODEL: {id_model}")
    print(f"TYPE_MODEL: {type(id_model)}")

    fecha_entrenamiento = str(int(datetime.now().timestamp()*1000))
    
    # new_train = Train(fecha_entrenamiento=str(datetime.now()),
    #                 data_train=copy.deepcopy(data_train),
    #                 modelo = str(id_model)).dict()
    
    new_train = Train(fecha_entrenamiento=str(datetime.now()),
                    data_train={"_id": data_train["_id"]},
                    errores = errores,
                    modelo = {"_id": id_model}).dict()
    

    empresa_planta = db_train_models.find_one({"empresa.nombre_empresa": empresa.nombre_empresa,
                               "planta.nombre_planta": planta.nombre_planta})
    
    if empresa_planta == None: 
    
        entrenamiento_planta = EntrenamientoPlanta(empresa=Empresa(nombre_empresa=empresa.nombre_empresa),
                                                    planta=Planta(nombre_planta=planta.nombre_planta),
                                                    entrenamientos={fecha_entrenamiento: new_train
                                                                        }).dict()
        
        db_train_models.insert_one(dict(entrenamiento_planta))
        return 0
        
    else:
        #db_train_models.update_one(empresa_planta, {"$set":{":}})
        print("MODIFICADO")
        db_train_models.update_one(empresa_planta, {"$set":{f"entrenamientos.{fecha_entrenamiento}":new_train}})
        

        return 0
    
      
 
    

@router.get('/{id_empresa}/{id_planta}/{id_modulo}/predict')
async def predict(id_empresa: int, id_planta: int):
    
        
    buscar_empresa_id(id_empresa, empresas_list)
    
    plantas = plantas_por_empresa(id_empresa, plantas_list)
    planta = buscar_planta_id(plantas, id_planta)

    if planta.modulo == None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, 
                            detail = f"No tienes contratado el servicio")
        
    
   
        
    if planta.modulo.modelo == None:
        print("No tienes ningun modelo")
        print("Creando modelo de predicción...")  
        
        data_train = Data(input=[34.55, 543.5432, 11113.43], 
                          sampling=60, 
                          fecha_inicio="2019-20-11")
        
        fecha_creacion, serial_modelo, hiperparametros, version, type_model, error_train, error_test, error_metric = train(planta)
        
        data_train.errores = Errors(train=error_train, 
                                   test=error_test, 
                                   metric=error_metric)
        
        planta.modulo.modelo = Modelo(fecha_creacion=fecha_creacion, 
                                      serial=serial_modelo,
                                      hiperparametros=hiperparametros, 
                                      version=version, 
                                      tipo=type_model)
        
        # Buscar la empresa y la planta
        # Añadir el nuevo entrenamiento al apartado entrenamientos
        
        
        train_db = EntrenamientoPlanta(empresa=copy.deepcopy(planta.empresa),
                                       planta=copy.deepcopy(planta),
                                       entrenamientos=[Train(data_train=copy.deepcopy(data_train), 
                                                            modelo=copy.deepcopy(planta.modulo.modelo),
                                                            fecha_entrenamiento = str(datetime.now()))]
                                       )

        
    fechas, predicciones = predict_14_dias(planta.modulo.modelo.serial)
    dict_predicciones = dict(zip(fechas.astype(str), predicciones))
    
    
    data_test= Data(input=predicciones, sampling=60, fecha_inicio=str(fechas[0]))
    
    test_db = PredictPlanta(empresa=copy.deepcopy(planta.empresa),
                                       planta=copy.deepcopy(planta),
                                       predicciones=[Predict(data_predict=copy.deepcopy(data_test), 
                                                            modelo=copy.deepcopy(planta.modulo.modelo),
                                                            fecha_prediccion = str(datetime.now()))]
                                       )
    
    return dict_predicciones