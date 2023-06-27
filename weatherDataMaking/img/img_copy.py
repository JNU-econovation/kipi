import shutil
import os
import pandas as pd

df = pd.read_csv('data/sampled_train_data.csv')
col_names = df.columns.to_list()
print(col_names)

for i in col_names:
    os.makedirs('img/'+i, exist_ok=True)
    for j in df[i].dropna().astype('Int64').to_list():
        source_path = '/Volumes/My Book/downloads/K-Fashion 이미지/Training/원천데이터/'+i+'/'+str(j)+'.jpg'
        destination = 'img/'+i+'/'+str(j)+'.jpg'
        shutil.copyfile(source_path, destination)
        # print(destination)
    print(f'{i} is done..!')
