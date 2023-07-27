from sqlalchemy import Boolean, Column, Integer, String

from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)

class Clotheset(Base):
    __tablename__ = "clothesets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    img_path = Column(String, nullable=True)
    user_id = Column(Integer, index=True)
    fac = Column(String, nullable=True)
    feature_vec = Column(String, nullable=True)
    temperture = Column(Integer, nullable=True)