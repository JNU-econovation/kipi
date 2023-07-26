import json 
import service, models, schemas
from database import engine
from sqlalchemy.orm import Session

models.Base.metadata.create_all(bind=engine)

def init_date(db: Session):
    count = 0
    try :
      user_data = schemas.UserCreate(email="hello@naver.com", password="1234")
      service.create_user(db, user_data)
    except:
      db.rollback()
      print("이미 유저가 존재합니다.")
    with open('init_data.json', 'r+') as save_file:
        json_data = json.load(save_file);
        for fac in json_data:
            for index, id in enumerate(json_data[fac]):
                try:
                  print(f"add {id} to db {fac} {count}")
                  clothes_base = service.create_clothset(db, schemas.ClothesetBase(user_id=1, name=f"{fac}_{index}", img_path=f"/custom_feature/img/clothes/{id}.jpg"))
                  service.add_cloth_info(db, f"/custom_feature/img/clothes/{id}.jpg", f"{index}", clothes_base.id, fac, ','.join(map(str, json_data[fac][id])), 0)
                  count += 1
                  db.rollback()
                except Exception as e:
                  print(e)
                  print("이미 데이터가 존재합니다.")
