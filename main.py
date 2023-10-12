from fastapi import FastAPI
from routers import empresa, planta, modulo, predict

from pymongo import MongoClient
from host import db

app = FastAPI()

#host = "mongodb://localhost:27017/"

print(db.list_collection_names())


#db["train_models"].insert_one = {"id": 1}

# Routers
app.include_router(empresa.router)
app.include_router(planta.router)
app.include_router(modulo.router)
app.include_router(predict.router)



@app.get("/")
async def root():
    return "Bienvenido a la API de prediccion a 14 dias"




