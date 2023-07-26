from typing import List
import datetime, secrets, os

from fastapi import Depends, FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR,'images/')

import models, schemas, service, init_data
from database import SessionLocal, engine

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/user", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = service.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    return service.create_user(db=db, user=user)

@app.post("/user/login", response_model=schemas.User)
def login_user(user: schemas.UserLogin, db: Session = Depends(get_db)):
    db_user = service.validate_user(db, email=user.email, password=user.password)
    if not db_user:
        raise HTTPException(status_code=400, detail="Invalid credentials")
    return db_user

@app.get("/user", response_model=List[schemas.User])
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = service.get_users(db, skip=skip, limit=limit)
    return users

@app.get("/user/{user_id}", response_model=schemas.User)
def read_user(user_id: int, db: Session = Depends(get_db)):
    db_user = service.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

@app.post("/clotheset", response_model=schemas.ClothesetBase)
def create_clotheset(clotheset: schemas.ClothesetBase, db: Session = Depends(get_db)):
    return service.create_clothset(db=db, clotheset=clotheset)

@app.get("/clotheset/{clotheset_id}")
def read_clotheset_by_id(clotheset_id: int, db: Session = Depends(get_db)):
    db_clotheset = service.get_clotheset(db, clotheset_id=clotheset_id)
    if db_clotheset is None:
        raise HTTPException(status_code=404, detail="Clotheset not found")
    return db_clotheset

@app.get("/clotheset/user/{user_id}")
def read_clothesets_by_user(user_id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    clothesets = service.get_clothesets_by_user(db, user_id=user_id, skip=skip, limit=limit)
    return clothesets

@app.get("/clotheset/recommend/{user_id}")
def read_recommend_clothesets(user_id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db), temperture: int = 0):
    clothesets = service.get_recommend_clothesets(db, user_id=user_id, skip=skip, limit=limit)
    return clothesets

@app.patch("/clotheset/{clotheset_id}")
def update_clotheset(clotheset_id: int, clotheset: schemas.ClothesetUpdate, db: Session = Depends(get_db)):
    db_clotheset = service.update_clotheset(db, clotheset_id=clotheset_id, clotheset=clotheset)
    return db_clotheset

@app.delete("/clotheset/{clotheset_id}")
def delete_clotheset(clotheset_id: int, db: Session = Depends(get_db)):
    db_clotheset = service.delete_clotheset(db, clotheset_id=clotheset_id)
    return db_clotheset

@app.post("/image")
async def upload_image(file:  UploadFile = File(...)):
    currentTime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    saved_file_name = ''.join([currentTime, secrets.token_hex(16)])
    print(saved_file_name)
    file_location = os.path.join(STATIC_DIR,"/images",saved_file_name)
    contents = await file.read()
    with open(file_location, "wb+") as file_object:
        file_object.write(contents)
    return saved_file_name

@app.get('/image/{file_name}')
async def get_image(file_name:str):
    return FileResponse(os.join([STATIC_DIR,"/images",file_name]))

@app.get('/init')
def init(db: Session = Depends(get_db)):
    init_data.init_date(db)
    return "init"

