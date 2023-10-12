
def entrenamiento_schema(entrenamiento) -> dict:
    return {"id": str(entrenamiento["_id"]),
            "empresa": schema_empresa(entrenamiento["empresa"]),
            "planta": schema_planta(entrenamiento["planta"]),
            "entrenamientos": schema_trains(entrenamiento["entrenamientos"])}

def schema_empresa(empresa) -> dict:
    return{"nombre_empresa": empresa["nombre_empresa"]}

def schema_planta(planta) -> dict:
    return{"nombre_planta": planta["nombre_planta"]}

def schema_train(entrenamientos) -> dict:
    return {"data_train": schema_data(entrenamientos["data_train"]),
            "modelo": schema_model(entrenamientos["modelo"])
            }

def schema_trains(entrenamientos):
    dictt = {key: schema_train(value) for key, value in entrenamientos.items()}
    return dictt


def schema_data(data) -> dict:
    return {"input": data["input"],
            "sampling": data["sampling"],
            "fecha_inicio": data["fecha_inicio"],
            "errores": schema_errores(data["errores"])}

def schema_errores(errores) -> dict:
    return {"train": errores["train"],
            "test": errores["test"],
            "metric": errores["metric"]}


def schema_model(modelo) -> dict:
    return {"fecha_creacion": modelo["fecha_creacion"],
            #"serial": str(modelo["serial"]),
            "hiperparametros": modelo["hiperparametros"],
            "version": modelo["version"],
            "tipo": modelo["tipo"]}
    
  
    
    
    
    
def entrenamientos_schema(entrenamiento_list) -> list:
    return [entrenamiento_schema(entrenamiento) for entrenamiento in entrenamiento_list]
