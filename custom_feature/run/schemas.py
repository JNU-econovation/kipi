from pydantic import BaseModel

class UserBase(BaseModel):
    email: str

class UserCreate(UserBase):
    password: str

class UserLogin(UserBase):
    password: str

class User(UserBase):
    id: int
    is_active: bool

    class Config:
        orm_mode = True

class ClothesetBase(BaseModel):
    user_id: int
    img_path: str
    name: str

class Clotheset(ClothesetBase):
    id: int
    fac: str
    feature_vec: str
    temperture: int

    class Config:
        orm_mode = True

class ClothesetCreate(ClothesetBase):
    fac: str
    feature_vec: str
    temperture: int

class ClothesetUpdate(BaseModel):
    name: str
    fac: str