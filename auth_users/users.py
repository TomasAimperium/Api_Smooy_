###################################

from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from passlib.hash import bcrypt

# Configuración de la aplicación y autenticación
app = FastAPI()


# Diccionario para almacenar usuarios (esto es solo para fines didácticos)
users_db = {
    "user1": {
        "username": "user1",
        "password": bcrypt.hash("password1"),  # Contraseña: password1
        "access_level": "add",
        "empresa":"a"
    },
    "user2": {
        "username": "user2",
        "password": bcrypt.hash("password2"),  # Contraseña: password2
        "access_level": "subtract"
    },
    "user3": {
        "username": "user3",
        "password": bcrypt.hash("password3"),  # Contraseña: password3
        "access_level": "all"
    }
}


# Modelo Pydantic para el token de acceso
class Token(BaseModel):
    access_token: str
    token_type: str

# Modelo Pydantic para el usuario
class User(BaseModel):
    username: str
    access_level:str
    empresa:str
    


# Modelo Pydantic para la creación de usuarios
class UserCreate(BaseModel):
    username: str
    password: str

# Configuración de contraseñas
SECRET_KEY = "SECRET_KEY"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Funciones de utilidad para el manejo de tokens
def create_access_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = users_db.get(form_data.username)
    if user is None or not bcrypt.verify(form_data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Credenciales incorrectas")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user["username"], "access_level": user["access_level"]}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}



# Función para verificar el token de acceso
async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="No se pudo validar las credenciales")
        user = users_db.get(username)
        if user is None:
            raise HTTPException(status_code=401, detail="Usuario no encontrado")
        return User(username=user["username"], access_level=user["access_level"],empresa=user['empresa'])
    except JWTError:
        raise HTTPException(status_code=401, detail="No se pudo validar las credenciales")




@app.get("/{empresa}/add", response_model=int)
async def add(empresa: str,current_user: User = Depends(get_current_user)):

    if current_user.empresa != empresa:
        raise HTTPException(status_code=403, detail="No tienes permiso para esta operación.")

    if current_user.access_level != "add" and current_user.access_level != "all":
        raise HTTPException(status_code=403, detail="No tienes permiso para esta operación.")
    
    result = 5 + 1
    return result

@app.get("/subtract", response_model=int)
async def subtract(current_user: User = Depends(get_current_user)):
    if current_user.access_level != "subtract" and current_user.access_level != "all":
        raise HTTPException(status_code=403, detail="No tienes permiso para esta operación.")
    
    result = 5 - 1
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# curl -X POST "http://localhost:8000/token" -d "grant_type=password" -d "username={username}" -d "password={password}"

