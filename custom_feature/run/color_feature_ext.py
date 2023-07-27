import extcolors
from PIL import Image
import json
import pickle

import torch

def color_feature(fac):
    with open(f'../data/prepro_data/{fac}/category_feature_name.pkl', 'rb') as f:
        c = pickle.load(f)
    file_names = []
    for i in range(len(c)):
        file_names.extend(c[i])

    colors_list = []

    for i,file_name in enumerate(file_names):
        colors, pixel_count = extcolors.extract_from_path(file_name.replace)
        colors_list.append(torch.tensor(list(colors[0][0])))
        if i%100 == 0:
            print(f'{i}th/{len(file_names)}th progress...')
            
    
    return colors_list

if __name__ == '__main__':
    fac_list = ['upper', 'onepiece', 'pants', 'outer']
    for fac in fac_list:
        with open(f'../data/prepro_data/{fac}/color_feature.pkl','wb') as f:
                pickle.dump(color_feature(fac),f)