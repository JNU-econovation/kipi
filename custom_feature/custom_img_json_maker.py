import os
import json

fac_list = ['outer', 'pants', 'onepiece', 'upper']

base_path = './img/prepro_clothes/'
input_path = "../img/prepro_clothes/"

for fac in fac_list:
    ret_list = []

    for f_name in os.listdir(os.path.join(base_path+fac)):
        ret_list.append({"file_name":os.path.join(input_path+fac+'/'+f_name)}, {"labels"})
        
    with open(f'./data/prepro_data/{fac}_img.json', 'w') as f:
        json.dump(ret_list, f, indent=4)
    print(f'{fac} is done!')
