from sqlalchemy.orm import Session
import models, schemas

def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()

def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.User).offset(skip).limit(limit).all()

def validate_user(db: Session, email: str, password: str):
    user = get_user_by_email(db, email)
    if not user:
        return False
    if not password == user.hashed_password:
        return False
    return user

def create_user(db: Session, user: schemas.UserCreate):
    db_user = models.User(email=user.email, hashed_password=user.password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_clotheset(db: Session, clotheset_id: int):
    return db.query(models.Clotheset).filter(models.Clotheset.id == clotheset_id).first()

def get_clothesets(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Clotheset).offset(skip).limit(limit).all()

def get_clothesets_by_user(db: Session, user_id: int, skip: int = 0, limit: int = 100):
    return db.query(models.Clotheset).filter(models.Clotheset.user_id == user_id).offset(skip).limit(limit).all()

def get_cloth(db: Session, cloth_id: int):
    return db.query(models.Cloth).filter(models.Cloth.id == cloth_id).first()

def create_clothset(db: Session, clotheset: schemas.ClothesetBase):
    db_clotheset = models.Clotheset(name=clotheset.name, img_path=clotheset.img_path, user_id=clotheset.user_id, fac="undefined", feature_vec="", temperture=0)
    db.add(db_clotheset)
    db.commit()
    db.refresh(db_clotheset)
    return db_clotheset

def add_cloth_info(db: Session,img_path:str, name: str, clotheset_id: int, fac: str,  feature_vec: str, temperture: int):
    db_clotheset = get_clotheset(db, clotheset_id)
    db_clotheset.img_path = img_path
    db_clotheset.name = name
    db_clotheset.fac = fac
    db_clotheset.feature_vec = feature_vec
    db_clotheset.temperture = temperture
    db.add(db_clotheset)
    db.commit()
    db.refresh(db_clotheset)
    return db_clotheset

def delete_clotheset(db: Session, clotheset_id: int):
    db_clotheset = get_clotheset(db, clotheset_id)
    db.delete(db_clotheset)
    db.commit()
    return db_clotheset