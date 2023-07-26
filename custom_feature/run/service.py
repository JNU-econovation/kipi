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

def get_recommend_clothesets(db: Session, user_id: int, skip: int = 0, limit: int = 100, temperture: int = 0):
    from get_recommend_info import getRecommendCloth
    closet_id_arr = getRecommendCloth(db.query(models.Clotheset).filter(models.Clotheset.user_id == user_id).offset(skip).limit(limit).all(), temperture)
    return db.query(models.Clotheset).filter(models.Clotheset.id in closet_id_arr).all()

def get_cloth(db: Session, cloth_id: int):
    return db.query(models.Cloth).filter(models.Cloth.id == cloth_id).first()

def create_clothset(db: Session, clotheset: schemas.ClothesetBase):
    from get_cross_info import getCrossInfo
    (fac, feature) = getCrossInfo(clotheset.img_path)
    db_clotheset = models.Clotheset(name=clotheset.name, img_path=clotheset.img_path, user_id=clotheset.user_id, fac=fac, feature_vec=feature, temperture=0)
    db.add(db_clotheset)
    db.commit()
    db.refresh(db_clotheset)
    return db_clotheset

def update_clotheset(db: Session, clotheset_id: int, clotheset: schemas.ClothesetUpdate):
    db_clotheset = get_clotheset(db, clotheset_id)
    db_clotheset.name = clotheset.name
    db_clotheset.fac = clotheset.fac
    db.commit()
    db.refresh(db_clotheset)
    return db_clotheset

def delete_clotheset(db: Session, clotheset_id: int):
    db_clotheset = get_clotheset(db, clotheset_id)
    db.delete(db_clotheset)
    db.commit()
    return db_clotheset